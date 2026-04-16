import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append("/home-local/kimm58/WMLifespan/code/scripts/ModelConfidence/DeepLearning")
import architecture
import torch
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_with_dropout(model, data, iters=100, dropout_p=0.2):
    model.eval()
    #enable dropout 
    for m in model.pref_net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout_p
            m.train()
    preds_list = []
    trajectories_list = []
    with torch.no_grad():
        for iter in range(iters):
            iteration_seed = torch.randint(0, 2**32, (1,)).item() #seeding for dropout randomness - keeping the same across train/val and centile/lifespan
            torch.manual_seed(iteration_seed)
            preds = model(data, training=False)
            #stack the preds / append to the growing list/DF
            preds_list.append(preds.unsqueeze(0))
            #do the centile trajectory inference here as well
            trajectories, ages = infer_centile_curves(model, seednum=iteration_seed)
            trajectories_list.append(trajectories)
    preds_stack = torch.cat(preds_list, dim=0)
    trajectories_stack = torch.cat(trajectories_list, dim=1)
    return preds_stack, trajectories_stack

def infer_centile_curves(model, seednum=None):
    assert seednum is not None, "Please provide a seed number for reproducibility."
    ages = torch.linspace(0.1, 90, steps=1000)
    ages_orig = ages.clone()
    ages = architecture.apply_age_transformation(ages, norm=True).to(DEVICE)
    
    #fill an array called ss with 0.5
    ss = torch.full_like(ages, 0.5).to(DEVICE)
    torch.manual_seed(seednum)
    centile_med = model.get_centile_curves(ages, ss, centile=0.5)
    torch.manual_seed(seednum)
    centile_025 = model.get_centile_curves(ages, ss, centile=0.025)
    torch.manual_seed(seednum)
    centile_975 = model.get_centile_curves(ages, ss, centile=0.975)
    centiles = torch.vstack([centile_025, centile_med, centile_975]).T
        #
        #check the shape
        #
    return centiles, ages_orig

    # ms = torch.ones_like(ages).to(DEVICE)  #male
    # fs = torch.zeros_like(ages).to(DEVICE)  #female
    # torch.manual_seed(seednum)
    # m_centile_med = model.get_centile_curves(ages, ms, centile=0.5)
    # torch.manual_seed(seednum)
    # f_centile_med = model.get_centile_curves(ages, fs, centile=0.5)
    # torch.manual_seed(seednum)
    # m_centile_025 = model.get_centile_curves(ages, ms, centile=0.025)
    # torch.manual_seed(seednum)
    # f_centile_025 = model.get_centile_curves(ages, fs, centile=0.025)
    # torch.manual_seed(seednum)
    # m_centile_975 = model.get_centile_curves(ages, ms, centile=0.975)
    # torch.manual_seed(seednum)
    # f_centile_975 = model.get_centile_curves(ages, fs, centile=0.975)
    # colname_prefix = ['m_centile_.025', 'm_centile_.5', 'm_centile_.975',
    #             'f_centile_.025', 'f_centile_.5', 'f_centile_.975']


def run_model_inference(model_dict):
    csv = model_dict['csv']
    df_orig = pd.read_csv(csv)
    df_res = df_orig.copy()
    traj_df = pd.DataFrame()
    traj_df['ages'] = torch.linspace(0.1, 90, steps=1000).numpy()
    dataset = architecture.MAGCEDataset(df_orig, age_transform=True, age_norm=True)
    dataset_X = torch.tensor(dataset.X).to(DEVICE)
    #get the fold from the model file
    for fold in sorted(df_orig['fold'].unique()):
        # here were the args: --pdf_loss --decoupled --site_dropout_prob 0.2 --equal_3_batch --age_transform --lambda_mse 0.0
            #this also includes age normalization (since the argument to not do it is not included)
        #load in the model
        model_file = model_dict[str(fold)]
        model = architecture.DecoupledMAGCE(input_dim=dataset.X.shape[1]-1, site_dropout_p=0.2).to(DEVICE)
        model.load_state_dict(torch.load(model_file, weights_only=True))
        model.eval()
        
        #model inference without dropout
        with torch.no_grad():
            centiles_nodropout = model(dataset_X, training=False)
            trajectory_nodropout, _ = infer_centile_curves(model, seednum=142) #seednum doesnt really matter here because there is no dropout
        #model inference and centile trajectory inference with dropout (dropout percentage = 5%)
        centiles_dropout, trajectories_dropout = infer_with_dropout(model, dataset_X, iters=100, dropout_p=0.05)
        all_centiles = torch.vstack([centiles_nodropout, centiles_dropout])
        all_trajectories = torch.cat([trajectory_nodropout, trajectories_dropout], dim=1)
        #traj_data = torch.cat([ages.unsqueeze(1).to(DEVICE), all_trajectories], dim=1).detach().cpu().numpy()
        centiles_colnames = [f'centile_fold{fold}_no_dropout'] + [f'centile_fold{fold}_dropout_iter{iter}' for iter in range(centiles_dropout.shape[0])]
        centiles_trajectory_colnames = [f'trajectory_fold{fold}_no_dropout_0.025centile', f'trajectory_fold{fold}_no_dropout_0.5centile', f'trajectory_fold{fold}_no_dropout_0.975centile'] 
        for iter in range(centiles_dropout.shape[0]):
            centiles_trajectory_colnames.append(f'trajectory_fold{fold}_dropout_iter{iter}_0.025centile')
            centiles_trajectory_colnames.append(f'trajectory_fold{fold}_dropout_iter{iter}_0.5centile')
            centiles_trajectory_colnames.append(f'trajectory_fold{fold}_dropout_iter{iter}_0.975centile')
        #add to the dataframe
        new_cent_cols = pd.DataFrame(all_centiles.detach().cpu().numpy().T, index=df_res.index, columns=centiles_colnames)
        df_res = pd.concat([df_res, new_cent_cols], axis=1)
        #also create the dataframe for the centile trajectories
        traj_df_new_cols = pd.DataFrame(all_trajectories.detach().cpu().numpy(), index=traj_df.index, columns=centiles_trajectory_colnames)
        traj_df = pd.concat([traj_df, traj_df_new_cols], axis=1)
        #delete the model from memory
        del model
        

    return df_res, traj_df

root = Path("/fs5/p_masi/kimm58/WMLifespan/data/ModelConfidence/DeepLearning")
dataset_root = root / "kfold_datasets"
model_root = root / "trained_models"
outdir = root / "model_inference"

#go through all the datasets and get the corresponding model files
model_dict = {}
for csv in dataset_root.glob('*.csv'):
    metric = csv.name.split('_CN')[0]
    model_dir = model_root / metric
    assert  model_dir.exists(), print(f"Model directory for metric {metric} does not exist.")
    
    #get the models
    models = list(model_dir.glob('*.pt'))
    if len(models) != 5:
        print(f"Warning: Expected 5 models for metric {metric}, found {len(models)}. Trimming to only 5...")
        #figure out the duplicate folds
        folds = [model.stem.split('_fold')[1][0] for model in models]
        fold_counts = pd.Series(folds).value_counts()
        duplicate_folds = fold_counts[fold_counts > 1].index.tolist()
        #get the models that are duplicated
        for dup in duplicate_folds:
            dup_models = [model for model in models if f"_fold{dup}" in model.stem]
            #see which one has the largest epoch
            epochs = [int(model.stem.split('_epoch')[1]) for model in dup_models]
            max_epoch_idx = epochs.index(max(epochs))
            #keep the one with the largest epoch
            to_remove = []
            for i, model in enumerate(dup_models):
                if i != max_epoch_idx:
                    to_remove.append(model)
        #remove the duplicates from the models list
        for rem in to_remove:
            models.remove(rem)
    model_dict[metric] = {}
    for model in models:
        fold = model.stem.split('_fold')[1][0]
        model_dict[metric][fold] = model_dir / model
    model_dict[metric]['csv'] = csv

for metric in tqdm(model_dict.keys()):
    submodel_dict = model_dict[metric]
    centile_df, traj_df = run_model_inference(submodel_dict)
    #save the dataframes
    out_centile_file = outdir / f"{metric}_centiles.csv"
    out_traj_file = outdir / f"{metric}_trajectories.csv"
    centile_df.to_csv(out_centile_file, index=False)
    traj_df.to_csv(out_traj_file, index=False)