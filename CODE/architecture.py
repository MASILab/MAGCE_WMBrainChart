"""
Architecture for a purely data-driven GAMLSS model

INPUTS (K+3)
- age
- sex
- One-hot vector of dataset (K)
- Tract metric (used at end)


"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

expected_datasets = ['ABCD', 'ADNI', 'AOMICID1000', 'AOMICPIOP1', 'AOMICPIOP2',
    'Ageility', 'BANDA', 'BIOCARD', 'BLSA', 'BSNIP1', 'BSNIP2', 'CALM',
    'CAMCAN', 'CASCIO', 'CUTTING', 'Calgary', 'DLBS', 'EBDT', 'HABSHD',
    'HBCD', 'HBNnew', 'HCP', 'HCPA', 'HCPBaby', 'HCPD', 'Humphreys',
    'IBIS', 'ICBM', 'Lexical', 'MAP', 'MARS', 'MASiVar', 'MORGAN',
    'NACC', 'NKInew', 'PING', 'QTAB', 'ROS', 'SCAN', 'SWU',
    'TempleSocial', 'UCLA_LA5c', 'UKBB', 'UPennRisk', 'UTAustin579',
    'VMAP_2.0', 'VMAP_JEFFERSON', 'VMAP_TAP', 'WRAP', 'dHCPinfant']

def apply_age_transformation(ages, reverse=False, norm=False):
    """
    Given an age vector in years, apply a transformation to it:
    - convert to days
    - make the age years since conception (adding 280 days)
    - apply natural log transform
    """
    if not reverse:
        ages_days = ages * 365.25
        ages_conception = ages_days + 280
        ages_log = torch.log(ages_conception)
        if norm:
            return apply_age_norm(ages_log, mean=9.6, std=1)
        return ages_log
    else:
        if norm:
            ages = apply_age_norm(ages, mean=9.6, std=1, reverse=True)
        ages_exp = torch.exp(ages)
        ages_conception = ages_exp - 280
        ages_years = ages_conception / 365.25
        return ages_years

def apply_age_norm(ages, mean=40, std=25, reverse=False):
    """
    Applies a normalization to the ages
    """
    if not reverse:
        return (ages - mean) / std
    else:
        return (ages * std) + mean

class MAGCE(nn.Module):
    """
    Micro- and mAcrostructural Growth Charts for whitE matter (MAGCE)

    Note that the input_dim argument should be K+2, where K is the number of datasets

    cbase is the base offset to ensure that the 'reasonable' centiles being learned are non-negative:

    Inversion of the centile net function yields
        T = c - (1/a)*ln((1/centile)-1)
    
    Thus, we enforce a constraint that when centile=0.01, T > 0:
        c - (1/a)*ln(1/0.01 - 1) > 0
        c - (1/a)*ln(100 - 1) > 0
        c > (1/a)*4.5951
        c > 4.5951/a
    
    So we learn the offset deltac that is strictly positive

    For the softplus, we learn a scaled softplus to allow for larger alpha and c values if possible
    """
    #def __init__(self, input_dim=52, hidden_dim=128, dropout_p=0.2, cbase=4.59, softplus_scale=5.0):
    def __init__(self, input_dim=52, hidden_dim=128, dropout_p=0.2, cbase=0, softplus_scale=1.0):
        super(MAGCE, self).__init__()
        #prefix network - takes in all but the last element of the input vector
            #outputs 'alpha' value 
        self.pref_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),#nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),#nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 2), #replaced 1 with 2 to allow for c to be learnable as a function of age and sex
            #sigmoid for alpha prime to alpha
                #REMOVED: replaced with softplus instead to allow larger alpha values than 1
            #nn.Sigmoid()
        )

        self.register_buffer('normalizer_scalar', torch.tensor(-1.5))
        self.register_buffer('c_base', torch.tensor(cbase))
        self.register_buffer('softplus_scale', torch.tensor(softplus_scale))

        final_layer = self.pref_net[-1]
        with torch.no_grad():
            # Set Alpha bias (index 0) to 5.0
            final_layer.bias[0] = 8.0
            # Set C bias (index 1) to 0.0 (Optional)
            final_layer.bias[1] = 0.0

        #centile network - a logistic functiontakes in last element of input vector
            #outputs a single scalar that is a centile value
            #formula is 1 / (1 + exp(-alpha * (T - c)))
                #where alpha is from prefix network, c is a learned parameter, and T is the last element of input vector
        
            #moved to be output by pref_net
        #self.c = nn.Parameter(torch.tensor(0.0))  # learnable centile parameter

    def centile_net(self, T, alpha, c):
        #centile = 1 / (1 + torch.exp(-alpha * (T - c)))
        centile = torch.sigmoid(alpha * (T - c))
        return torch.clamp(centile, min=1e-6, max=1-1e-6) #stability issues
        #return centile

    def softplus(self, x):
        return nn.functional.softplus(x, beta=self.softplus_scale) #better stability in backprop?
        #return torch.log(1 + self.softplus_scale*torch.exp(x))

    def set_normalizing_scalar(self, data):
        #get the normalizing number so that the data are on a more manageable scale
            #if this doesnt work, a normalizer that goes between 0 and 1 (softly)
                #and then if that doesnt work, try a strict 0 and 1 (min-max to 0-1)
        if self.normalizer_scalar == -1:
            # Prevent accidental re-calculation if called multiple times
            print("ERROR: Normalizing scalar is already registered. Skipping calculation...")
            return
        scale_exp = torch.floor(torch.log10(torch.max(torch.tensor(data)))) #exponent for scaling factor
        normalizer_scalar = 10 ** scale_exp
        self.register_buffer('normalizer_scalar', normalizer_scalar)

    def get_centile_curves(self, ages, sexes, centile=0.5, return_alpha_c=False):
        """
        Given the ages, sexes, and desired centile value (0 to 1), returns the corresponding tract metric values
        """
        #run the prefix network to get alphas and cs
        #first, make the dataset agnostic input
        batch_size = ages.shape[0]
            #input features -2 to account for age and sex
        dataset_num = self.pref_net[0].in_features - 2
        dataset_agnostic_input = torch.zeros(batch_size, dataset_num).to(ages.device)
        dataset_agnostic_input.fill_(1.0 / dataset_num)  # uniform distribution across datasets
        
        #stack the ages and sexes the dataset tensors together
        prefix_input = torch.cat((ages.unsqueeze(1), sexes.unsqueeze(1), dataset_agnostic_input), dim=1)

        # run the prefix network to get alphas and cs
        alphaprime_dc = self.pref_net(prefix_input)  # shape: (batch_size, 2)
        alpha = self.softplus(alphaprime_dc[:, 0])  # ensure alpha is positive
        dc = self.softplus(alphaprime_dc[:, 1])  # c is positive as well
        
        # calculate the centile metric values
        T = (dc + self.c_base) - (torch.log( (torch.tensor(1)/centile) - torch.tensor(1)) / alpha)
        T *= self.normalizer_scalar  # rescale T back to original scale
        #T = (torch.log(centile / (1 - centile)) + c) / alpha  # inverse of the centile function
        if return_alpha_c:
            return T, alpha, dc + self.c_base
        return T
    
    def get_median_centile(self, x, return_gt=False, original_scale=False):
        """
        For each individual participant, return the corresponding median centile

        Input is just the same X as forward()
        """
        prefix = x[:, :-1]
        #no need to extract T unless explicity returning it
        alphaprime_c = self.pref_net(prefix)  # shape: (batch_size, 2)
        alpha = self.softplus(alphaprime_c[:, 0])  # ensure alpha is positive
        alpha = torch.clamp(alpha, min=0.4, max=10.0) #for stability
        dc = self.softplus(alphaprime_c[:, 1])  # ensure that deltac is positive as well
        #calculate the centile metric values
        centile = torch.tensor(0.5)
        T = (dc + self.c_base) - (torch.log( (torch.tensor(1)/centile) - torch.tensor(1)) / alpha)
        if original_scale:
            T *= self.normalizer_scalar  # rescale T back to original scale
        if return_gt:
            gt_T = x[:, -1]
            if not original_scale:
                gt_T = gt_T / self.normalizer_scalar
            return T, gt_T
        return T

    def forward(self, x, return_alpha_c=False, return_site_offsets=False, training=True):
        #make sure the normalizing scalar is set
        assert self.normalizer_scalar != -1.5, "Normalizing scalar is not set. Please call set_normalizing_scalar() before forward()." 
        
        #split input into prefix and T
        prefix = x[:, :-1]
        T = x[:, -1] / self.normalizer_scalar  # rescale T
        alphaprime_c = self.pref_net(prefix)  # shape: (batch_size, 2)
        alpha = self.softplus(alphaprime_c[:, 0])  # ensure alpha is positive
        alpha = torch.clamp(alpha, min=0.4, max=10.0) #for stability
        #print(f"alpha max: {torch.max(alpha).item()}, min: {torch.min(alpha).item()}")
        deltac = self.softplus(alphaprime_c[:, 1])  # ensure that deltac is positive as well
        centile = self.centile_net(T, alpha, deltac+self.c_base)
        if return_alpha_c:
            return centile, alpha, deltac + self.c_base
        return centile

class DecoupledMAGCE(MAGCE):
    """
    Version of MAGCE where the site and biological effects are decoupled

    1.) BioNet (pref_net): Learns the overall curve from age+sex
    2.) SiteNet: Learns the site-specific offsets from one-hot dataset vector
    """
    def __init__(self, input_dim=52, input_dim_bio=2, hidden_dim=128, dropout_p=0.2, cbase=0, softplus_scale=1.0, site_dropout_p=0):
        # Initialize parent with dummy input_dim (we will overwrite the pref_net anyway)
        super(DecoupledMAGCE, self).__init__(input_dim=input_dim, hidden_dim=hidden_dim, 
                                             dropout_p=dropout_p, cbase=cbase, softplus_scale=softplus_scale)

        #Bio net
        self.bio_dim = input_dim_bio
        self.pref_net = nn.Sequential(
            nn.Linear(input_dim_bio, hidden_dim),
            nn.ELU(),#nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),#nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 2), # outputs alpha and c prime
        )

        #Site net
        self.site_c_net = nn.Linear(input_dim - input_dim_bio, 1)  # outputs site-specific c offset
        self.site_alpha_net = nn.Linear(input_dim - input_dim_bio, 1)  # outputs site-specific alpha offset
        #initialize to zero offset and nearly no scaling
        self.site_c_net.weight.data.fill_(0.0)
        self.site_c_net.bias.data.fill_(0.0)
        self.site_alpha_net.weight.data.fill_(0.0)
        self.site_alpha_net.bias.data.fill_(0.0)

        #initialize site_net as frozen
        #self._freeze_site_nets()

        #initialize the bias term of the pref_net to have a reasonable starting alpha and c
        with torch.no_grad():
            self.pref_net[-1].bias[0] = 5.0
            self.pref_net[-1].bias[1] = 1.0

        #site dropout percent
        self.site_dropout_p = site_dropout_p

    def _freeze_site_nets(self, freeze_c_net=True, freeze_alpha_net=True, unfreeze=False):
            """
            Freeze/unfreeze the site nets
            """
            
            # Freeze site_c_net
            if freeze_c_net:
                if not unfreeze:
                    for param in self.site_c_net.parameters():
                        param.requires_grad = False
                    self.site_c_net.eval() # Crucial: disable BN/Dropout updates
                else:
                    for param in self.site_c_net.parameters():
                        param.requires_grad = True
                    self.site_c_net.train() # Enable BN/Dropout updates

            # Freeze site_alpha_net
            if freeze_alpha_net:
                if not unfreeze:
                    for param in self.site_alpha_net.parameters():
                        param.requires_grad = False
                    self.site_alpha_net.eval()
                else:
                    for param in self.site_alpha_net.parameters():
                        param.requires_grad = True
                    self.site_alpha_net.train()

    def _calc_centile_tract_values(self, centile, alpha, c):
        """
        Torch-safe way of calculating the centiles
        """
        log_odds = -torch.logit(centile, eps=1e-7)
        T = c - (log_odds / alpha)
        return T

    def get_centile_curves(self, ages, sexes, centile=0.5, return_alpha_c=False):
        """
        Given the ages, sexes, and desired centile value (0 to 1), returns the corresponding tract metric values

        Dataset agnostic curves, where we are not using the dataset shift and scaling
        """
        #run the prefix network to get alphas and cs
        #first, make the dataset agnostic input
        batch_size = ages.shape[0]

        #stack the ages and sexes tensors together
        prefix_input = torch.cat((ages.unsqueeze(1), sexes.unsqueeze(1)), dim=1)

        # run the prefix network to get alphas and cs
        alphaprime_c = self.pref_net(prefix_input)  # shape: (batch_size, 2)
        alpha = self.softplus(alphaprime_c[:, 0])  # ensure alpha is positive
        c = self.softplus(alphaprime_c[:, 1])  # c is positive as well
        
        #calculate the mean centile curve across all datasets

        # calculate the centile metric values
        #T = (c + self.c_base) - (torch.log( (torch.tensor(1)/centile) - torch.tensor(1)) / alpha)
        T = self._calc_centile_tract_values(torch.Tensor([centile]).to(alpha.device), alpha, c + self.c_base)
        T *= self.normalizer_scalar  # rescale T back to original scale
        #T = (torch.log(centile / (1 - centile)) + c) / alpha  # inverse of the centile function
        if return_alpha_c:
            return T, alpha, c + self.c_base
        return T

    def get_median_centile(self, x, return_gt=False, original_scale=False):
        """
        For each individual participant, return the corresponding median centile

        Input is just the same X as forward()
        """
        bio_prefix = x[:, :self.bio_dim]
        site_prefix = x[:, self.bio_dim:-1]
        #no need to extract T unless explicity returning it
        alphaprime_c = self.pref_net(bio_prefix)  # shape: (batch_size, 2)
        alphaprime_c = torch.clamp(alphaprime_c, min=-30.0, max=30.0) #clamp for stability
        alpha = self.softplus(alphaprime_c[:, 0])  # ensure alpha is positive
        alpha = torch.clamp(alpha, min=0.4, max=10.0) #for stability
        c = self.softplus(alphaprime_c[:, 1])  # ensure that c is positive as well
        #get the site offsets
        c_site_offset = self.site_c_net(site_prefix).squeeze(1)
        alpha_site_offset = self.site_alpha_net(site_prefix).squeeze(1)
        alpha = alpha + alpha_site_offset
        c = c + c_site_offset + self.c_base
        #calculate the centile metric values
        centile = torch.tensor(0.5)
        #T = c - (torch.log( (torch.tensor(1)/centile) - torch.tensor(1)) / alpha)
        T = self._calc_centile_tract_values(centile, alpha, c)
        if original_scale:
            T *= self.normalizer_scalar  # rescale T back to original scale
        if return_gt:
            gt_T = x[:, -1]
            if not original_scale:
                gt_T = gt_T / self.normalizer_scalar
            return T, gt_T
        return T

    def forward(self, x, return_alpha_c=False, return_site_offsets=False, training=True):
        assert self.normalizer_scalar != -1.5

        #split inputs into bio and site
        bio_prefix = x[:, :self.bio_dim]
        site_prefix = x[:, self.bio_dim:-1]
        T = x[:, -1] / self.normalizer_scalar  # rescale T
        #run bio net
        alphaprime_c = self.pref_net(bio_prefix)  # shape: (batch_size, 2)
        #alphaprime_c = torch.clamp(alphaprime_c, min=-30.0, max=30.0) #clamp for stability
        alpha_bio = self.softplus(alphaprime_c[:, 0])  # ensure alpha is positive
        #alpha_bio = torch.clamp(alpha_bio, min=0.4, max=10.0) #for stability
        c_bio = self.softplus(alphaprime_c[:, 1])  # ensure that c is positive as well
        #run site net
        c_site_offset = self.site_c_net(site_prefix).squeeze(1)
        alpha_site_offset = self.site_alpha_net(site_prefix).squeeze(1)
        #apply site dropout if specified
        if self.site_dropout_p > 0 and training:
            mask = torch.bernoulli(torch.ones_like(c_site_offset), 1 - self.site_dropout_p).to(x.device)
            c_site_offset = c_site_offset * mask
            alpha_site_offset = alpha_site_offset * mask
        else:
            mask = torch.ones_like(c_site_offset).to(x.device)
        alpha = alpha_bio + alpha_site_offset
        if any(torch.isnan(alpha)):
            pass
        c = c_bio + c_site_offset + self.c_base
        centile = self.centile_net(T, alpha, c)
        if return_alpha_c:
            if not return_site_offsets:
                return centile, alpha, c
            else:
                c_unique_offsets = c_site_offset[mask.to(torch.bool)].unique(dim=0)
                alpha_unique_offsets = alpha_site_offset[mask.to(torch.bool)].unique(dim=0)
                return centile, alpha, c, c_unique_offsets, alpha_unique_offsets
        if return_site_offsets:
            c_unique_offsets = c_site_offset.unique(dim=0)
            alpha_unique_offsets = alpha_site_offset.unique(dim=0)
            return centile, c_unique_offsets, alpha_unique_offsets
        return centile


class MAGCEDataset(Dataset):
    """
    Input is a CSV file with the following columns:
    - age: as a float
    - sex: 0 for female, 1 for male
    - dataset: string indicating dataset
    - "tract_metric": the metric value for the tract (name is dependent on the tract and metric being used)

    Have a separate train and validation MAGCEDataset 
    """
    def __init__(self, df, age_transform=False, age_norm=True):

        #drop diagnosis column if it exists
        df = df.drop(columns=[c for c in ['diagnosis', 'subject', 'fold'] if c in df.columns])

        #save the original dataset labels
        self.dataset_labels = df['dataset']

        #one-hot encode dataset
        #df = pd.get_dummies(df, columns=['dataset'], prefix='dataset') #replaced with the bottom two to allow for consistent ordering and missing datasets
        df['dataset'] = pd.Categorical(df['dataset'], categories=expected_datasets)
        df = pd.get_dummies(df, columns=['dataset'], prefix='dataset')
        #get the non-age, non-sex, non-dataset column
        metric_col = [x for x in df.columns if not x.startswith('dataset') and x not in ['age', 'sex']]
        assert len(metric_col) == 1, f"more than one metric column found: {metric_col}"
        metric_col = metric_col[0]
        #reorder the dataframe so that it goes age, sex, datasets..., metric
        col_order = ['age', 'sex'] + [x for x in df.columns if x.startswith('dataset')] + [metric_col]
        df = df[col_order]

        #assert not (age_transform and age_norm), "Cannot apply both age transformation and age normalization simultaneously."
        if age_transform:
            #apply age transformation
            df['age'] = apply_age_transformation(torch.tensor(df['age'].values.astype("float32")), norm=age_norm).numpy()
        elif age_norm:
            #normalize the age (mean 40, std 25)
            df['age'] = apply_age_norm(df['age'])


        ###### Need to edit when I figure out the best way to do the k-fold splits ######
        self.X = df.values.astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx])
        dataset_name = self.dataset_labels.iloc[idx]
        return X, dataset_name


## LOSS FUNCTION
"""
Given a set C of centiles, the loss function L will be the following:

L = L_m + L_f

L_m = weighted-wasserstein of males between centiles and uniform(0,1) of same size + weighted-wasserstein for each dataset
L_f = ... for females instead
""" 

class PDFLoss(nn.Module):
    """
    Loss function that is based on the derived PDF from the centile curves that come from the CDF of the MAGCE model

    Essentially a maximum likelihood estimation loss function
    """

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, alphas, centiles, cs, Ts):
        """
        Given the CDF of the logistic function:

        CDF = 1 / (1 + exp(-alpha * (T - c)))

        The resulting PDF is the derivative with respect to T:

        z = -alpha * (T - c)
        PDF = (alpha * exp(z)) / (a + exp(z))^2

        Alternatively, this is also the same as:

        PDF = alpha * CDF * (1 - CDF)

        Since CDF = 1 / (1 + exp(z)), and 1 - CDF = exp(z) / (1 + exp(z))

        This loss function maximizes the log likelihood of the PDF values at the true T values

        HOWEVER, there seems to be stability issues from the backpropagation. Thus, we:
        - calculate the log of the PDF algebraically (different equation)
        - add epsilon to avoid log(0)
        - mean over the batch and return negative log likelihood

        ln(PDF) = ln(alpha) + ln(CDF) + ln(1 - CDF)
        -----------
        ln(1-CDF) = ln(e^z / (1+e^z))
        ln(1-CDF) = z - ln(1+e^z)
        ln(1-CDF) = z - softplus(z)
        -----------
        ln(CDF) = ln(1 / (1+e^z))
        ln(CDF) = -ln(1+e^z)
        ln(CDF) = -softplus(z)
        -----------
        ln(PDF) = ln(alpha) - softplus(z) + z - softplus(z)
        ln(PDF) = ln(alpha) + z - 2*softplus(z)
        """

        # zs = -alphas * (Ts - cs)
        # pdfs = (alphas * torch.exp(zs)) / ((1 + torch.exp(zs)) ** 2)
        # #maximize the log likelihood
        # log_pdfs = torch.log(pdfs + self.epsilon)
        # return -torch.mean(log_pdfs)

        #pdfs = alphas * centiles * (1 - centiles)
        #log_pdfs = torch.log(pdfs + self.epsilon)

        zs = -alphas * (Ts - cs)
        log_pdfs = torch.log(alphas + self.epsilon) + zs - 2 * nn.functional.softplus(zs)
        if torch.any(torch.isnan(log_pdfs)):
            pass
        return -torch.mean(log_pdfs)

class MedianLoss(nn.Module):
    """
    MSE loss function for learning the median trajectory

    Input is the predicted medians and the true values
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_medians, true_values):
        return self.mse_loss(predicted_medians, true_values)

class RangeHingeLoss(nn.Module):
    """
    Squared hinge loss function for learning the range between extreme centile curves

    Provided the predicted centiles and the L/U bounds for the specific ages, sexes, and datasets, computes
    a loss to push the bounds of the centiles curves to remain close to the data
    """
    def __init__(self):
        super().__init__()
        # self.lower_bound = lower_bound
        # self.upper_bound = upper_bound
    
    def forward(self, centiles, lower_bounds, upper_bounds):
        lower_loss = torch.clamp(centiles - self.lower_bound, min=0) ** 2

class CentileLoss(nn.Module):
    """
    Wasserstein-based loss function for learning centile distributions

    reference ages are the set of ages that we will use to weight the wasserstein distances

    kappa is the bandwidth for the gaussian kernel used to weight the wasserstein age weightings

    wass_power is the power to which the wasserstein distance is raised (default is 2 for squared wasserstein)
    """
    def __init__(self, ref_min=0, ref_max=101, ref_step=2, kappa=0.85, wass_power=1,
                 log_ages=False, age_transform=False, dataset_weighting=False, age_varying_kernel=False):
        #super(CentileLoss, self).__init__()
        super().__init__()

        assert ref_max > ref_min, "ref_max must be greater than ref_min"
        if not log_ages:
            self.reference_ages_tensor = torch.arange(ref_min, ref_max, ref_step).float()
        else:
            #calculate the number of ages that would be included if linearly spaced with ref_step steps
            assert ref_min > 0, "ref_min must be greater than 0 for log spacing"
            num_ages = int((ref_max - ref_min) / ref_step)
            self.reference_ages_tensor = torch.logspace(torch.log10(torch.tensor(ref_min).float()), torch.log10(torch.tensor(ref_max).float()), steps=num_ages)

        self.register_buffer('reference_ages', self.reference_ages_tensor) #this will ensure it is moved to GPU
        self.kappa = kappa
        self.age_varying_kernel = age_varying_kernel
        self.register_buffer('wass_power', torch.tensor(wass_power))
        self.age_transform = age_transform
        self.dataset_weighting = dataset_weighting

    def get_age_weighting(self, ages):
        """
        Get the weightings for each reference age

        Weightings are gaussian kernel weightings
        """
        if self.age_varying_kernel:
            #calculate variable kappa based on age
            kappas = torch.tensor(1) - torch.exp(-(0.5*self.reference_ages + 0.1)) #kappa increases to a max value of 1
        else:    
            #just return a standard kappa for every age
            kappas = self.kappa
        return torch.exp(-0.5 * ((ages.unsqueeze(1) - self.reference_ages.unsqueeze(0)) / kappas) ** 2)
    
    def calc_wasserstein(self, centiles, min_val=0.01, max_val=0.99):
        uniform = torch.linspace(min_val, max_val, steps=centiles.shape[0]).to(centiles.device)
        #calculate the ordered wassserstein distance
        centiles_sorted, _ = torch.sort(centiles)
        wass = torch.abs(centiles_sorted - uniform.unsqueeze(0)) ** self.wass_power
        return wass

    def forward(self, centiles, ages, sexes, datasets):
        """
        First, we need to compute the weightings for each element

        Split into M/F

        Calculate the global loss (across all datasets) for M/F

        Then calculate the dataset-specific losses for M/F
        EDIT: do them separately so that large datasets are not upweighted too much

        Sum together and return

        If there is only one dataset, then just return the global loss and dont recalculate
        """
        if self.age_transform:
            ages = apply_age_transformation(ages, reverse=True)
        weights = self.get_age_weighting(ages) # shape: (batch_size, num_reference_ages)
        dataset_weights = torch.ones_like(centiles)
        if self.dataset_weighting:
            #weight by inverse of dataset size
            unique_datasets = sorted(list(set(datasets)))
            dataset_to_id = {name: i for i, name in enumerate(unique_datasets)}
            #get the counts for each dataset
            dataset_ids = torch.tensor([dataset_to_id[d] for d in datasets])
            id_list, counts = torch.unique_consecutive(dataset_ids, return_counts=True)
            dataset_count_dict = {ud.item(): c.item() for ud, c in zip(id_list, counts)}
            dataset_weights = torch.tensor([1.0 / dataset_count_dict[d.item()] for d in dataset_ids]).float()
            dataset_weights = dataset_weights / torch.sum(dataset_weights) * len(datasets)
        dataset_weights = dataset_weights.to(centiles.device)

        #dataset specific loss
        if len(set(datasets)) > 1:
            #initialize to size of centiles
            dataset_arr = torch.zeros(centiles.shape[0]).to(centiles.device)
            for dataset in set(datasets):
                #get the indices for this dataset
                dataset_indices = torch.tensor([i for i,x in enumerate(datasets) if x==dataset])
                dataset_mask = torch.tensor([1 if x==dataset else 0 for x in datasets]).bool().to(centiles.device)
                #Male
                dataset_arr[(dataset_mask) & (sexes == 1)] = self.calc_wasserstein(centiles[(dataset_mask) & (sexes == 1)])
                #Female
                dataset_arr[(dataset_mask) & (sexes == 0)] = self.calc_wasserstein(centiles[(dataset_mask) & (sexes == 0)])
            dataset_loss = dataset_arr.unsqueeze(1) * weights
            dataset_loss = dataset_loss * dataset_weights.unsqueeze(1)
            dataset_loss = torch.sum(dataset_loss, dim=1)

            total_loss = dataset_loss
        #across all datasets
        else:
            #initialize to size of centiles
            global_arr = torch.zeros(centiles.shape[0]).to(centiles.device)
            #Male
            global_arr[sexes==1] = self.calc_wasserstein(centiles[sexes==1])
            #Female
            global_arr[sexes==0] = self.calc_wasserstein(centiles[sexes==0])
            #apply weights: should result in an NxM matrix where N is batch size and M is number of reference ages
            global_loss = global_arr.unsqueeze(1) * weights
            global_loss = global_loss * dataset_weights.unsqueeze(1)
            global_loss = torch.sum(global_loss, dim=1)
            total_loss = global_loss
        return torch.mean(total_loss)
