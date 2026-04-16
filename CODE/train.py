
import architecture
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import argparse
from pathlib import Path
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNINGRATE = 0.0001

#set rand to 42
torch.manual_seed(42)


## WANDB logging helpers
def log_centile_histograms(model, epoch, data_type, centile_data_dict, dataset_names, fold, df_info, args, age_transform=False, tolerance=1e-6):
    """
    Logs complex histograms to Weights & Biases.
    
    Args:
        epoch (int): Current epoch number.
        data_type (str): 'Train' or 'Val'
        centile_data_dict (dict): Contains 'centiles', 'ages', 'sexes', 'datasets' (tensors/arrays).
        dataset_names (list): List of unique dataset names for grouping.
    """

    def log_wandb_histogram(log_dict, name, data, tolerance):
        if np.ptp(data) > tolerance:
            try:
                log_dict[name] = wandb.Histogram(data)
            except ValueError:
                log_dict[name] = wandb.Histogram(data, num_bins=1)
        else:
            log_dict[name] = wandb.Histogram(data, num_bins=1)


    def plot_line(ax, x, y, linestyle, color, width):
        ax.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy(), linestyle=linestyle, color=color, linewidth=width)

    model.eval()
    centiles = centile_data_dict['centiles']
    sexes = centile_data_dict['sexes']
    datasets = centile_data_dict['datasets'] # This is expected to be a tensor of numerical IDs
    
    log_dict = {}

    # 1. Histograms of Centiles by Sex (Sex=1: Male, Sex=0: Female)
    # Ensure data is moved to CPU before converting to NumPy for logging
    #log_dict[f'{data_type}/Centiles_Sex_Male'] = wandb.Histogram(centiles[sexes == 1].cpu().numpy())
    #log_dict[f'{data_type}/Centiles_Sex_Female'] = wandb.Histogram(centiles[sexes == 0].cpu().numpy())
    log_wandb_histogram(log_dict, f'{data_type}/Centiles_Sex_Male', centiles[sexes == 1].cpu().numpy(), tolerance)
    log_wandb_histogram(log_dict, f'{data_type}/Centiles_Sex_Female', centiles[sexes == 0].cpu().numpy(), tolerance)

    # 2. Histograms of Centiles by Dataset (Across both sexes)
    # NOTE: Assumes dataset_names is a list of the string names, and datasets tensor holds their corresponding IDs.
    for dataset in dataset_names: 
        # Find indices corresponding to the current dataset ID
        # We assume the ID corresponds to the index in the dataset_names list for simplicity
        ds_indices = (datasets == dataset)
        
        plot_centiles = centiles[ds_indices].cpu().numpy()

        # Log combined centiles for this dataset
        if ds_indices.any():
            # if np.ptp(plot_centiles) > tolerance:
            #     log_dict[f'{data_type}/Centiles_Dataset_{dataset}'] = wandb.Histogram(
            #         centiles[ds_indices].cpu().numpy()
            #     )
            # else:
            #     log_dict[f'{data_type}/Centiles_Dataset_{dataset}'] = wandb.Histogram(
            #         centiles[ds_indices].cpu().numpy(), num_bins=1
            #     )
            log_wandb_histogram(log_dict, f'{data_type}/Centiles_Dataset_{dataset}', plot_centiles, tolerance)


    # 3. Histograms of raw Ages (helpful for checking distribution)
    #log_dict[f'{data_type}/Ages'] = wandb.Histogram(centile_data_dict['ages'].cpu().numpy())
    log_wandb_histogram(log_dict, f'{data_type}/Ages', centile_data_dict['ages'].cpu().numpy(), tolerance)

    #4. Histograms of Alpha values
    #log_dict[f'{data_type}/Alpha_Values'] = wandb.Histogram(centile_data_dict['alpha'].cpu().numpy())
    log_wandb_histogram(log_dict, f'{data_type}/Alpha_Values', centile_data_dict['alpha'].cpu().numpy(), tolerance)

    #5. Histograms of c values
    #log_dict[f'{data_type}/C_Values'] = wandb.Histogram(centile_data_dict['c'].cpu().numpy())
    log_wandb_histogram(log_dict, f'{data_type}/C_Values', centile_data_dict['c'].cpu().numpy(), tolerance)
    
    #6. Plot of centile curves
    ages = torch.linspace(df_info['age'].min(), df_info['age'].max(), steps=1000).to(DEVICE)
    ages_orig = ages.clone()
    if args.age_transform:
        ages = architecture.apply_age_transformation(ages, norm=not args.no_age_norm)
    elif not args.no_age_norm:
        ages = architecture.apply_age_norm(ages)
    ms = torch.ones_like(ages).to(DEVICE)  #male
    fs = torch.zeros_like(ages).to(DEVICE)  #female
    m_centile_med = model.get_centile_curves(ages, ms, centile=0.5)
    f_centile_med = model.get_centile_curves(ages, fs, centile=0.5)
    m_centile_025 = model.get_centile_curves(ages, ms, centile=0.025)
    f_centile_025 = model.get_centile_curves(ages, fs, centile=0.025)
    m_centile_975 = model.get_centile_curves(ages, ms, centile=0.975)
    f_centile_975 = model.get_centile_curves(ages, fs, centile=0.975)
    f, ax = plt.subplots(figsize=(10, 7))
    plot_line(ax, ages_orig, m_centile_med, linestyle='-', color='blue', width=2)
    plot_line(ax, ages_orig, f_centile_med, linestyle='-', color='red', width=2)
    plot_line(ax, ages_orig, m_centile_025, linestyle='--', color='blue', width=2)
    plot_line(ax, ages_orig, f_centile_025, linestyle='--', color='red', width=2)
    plot_line(ax, ages_orig, m_centile_975, linestyle='--', color='blue', width=2)
    plot_line(ax, ages_orig, f_centile_975, linestyle='--', color='red', width=2)
    metric_col = [x for x in df_info.columns if x not in ['diagnosis', 'subject', 'fold', 'age', 'sex', 'dataset']]
    assert len(metric_col) == 1, f"Expected exactly one metric column in df_info. Got {len(metric_col): {metric_col}}."
    metric_col = metric_col[0]
    sns.scatterplot(x='age', y=metric_col, data=df_info, alpha=0.05, s=10, ax=ax, hue='sex', palette={0:'red', 1:'blue'}, legend=False)
    #set the ylim to the min and max of the metric column
    ax.set_ylim(df_info[metric_col].min(), df_info[metric_col].max())
    #make the ticks bigger
    ax.tick_params(axis='both', which='major', labelsize=14)
    # log the figure
    log_dict[f'{data_type}/Centile_Curves'] = wandb.Image(f)

    #7. Logging of dataset-specific shifts and scales
    if args.decoupled:
        #log_dict[f'{data_type}/Alpha_Values_Site'] = wandb.Histogram(centile_data_dict['site_alpha'].cpu().numpy())
        #log_dict[f'{data_type}/C_Values_Site'] = wandb.Histogram(centile_data_dict['site_c'].cpu().numpy())
        log_wandb_histogram(log_dict, f'{data_type}/Alpha_Values_Site', centile_data_dict['site_alpha'].cpu().numpy(), tolerance)
        log_wandb_histogram(log_dict, f'{data_type}/C_Values_Site', centile_data_dict['site_c'].cpu().numpy(), tolerance)

    # Final logging step
    wandb.log(log_dict, step=epoch)

    plt.close(f)

def train_one_epoch(model, dataloader, optimizer, loss_fn, median_loss, lambda_mse=1.0):
    model.train()
    total_loss = 0.0
    # Data collection arrays for logging
    all_centiles = []
    all_sexes = []
    all_ages = []
    all_datasets = []
    all_alphas = []
    all_cs = []
    for X_batch, dataset_labels in dataloader:
        X_batch = X_batch.to(DEVICE)
        
        #now, loop through each dataset in the batch
        for dataset_idx in range(X_batch.shape[1]-3): ## minus 3 for the age, sex, and Tract metric
            subbatch_idxs = torch.where(X_batch[:, dataset_idx+2] == 1)[0]
            X_subbatch = X_batch[subbatch_idxs, :]
            dataset_labels_subbatch = [dataset_labels[i] for i in subbatch_idxs.cpu().numpy()]
            optimizer.zero_grad()
            preds, alphas, cs = model(X_subbatch, return_alpha_c=True)
            #get the ages, sexes, and dataset information
            loss = loss_fn(preds, X_subbatch[:, 0], X_subbatch[:, 1], dataset_labels_subbatch)
            med_centile, gt_T = model.get_median_centile(X_batch, return_gt=True)
            med_loss = median_loss(med_centile, gt_T)
            loss += med_loss
            loss.backward()
            optimizer.step()
        
        #now the entire batch
        optimizer.zero_grad()
        preds, alphas, cs = model(X_batch, return_alpha_c=True)
        #get the ages, sexes, and dataset information
        loss = loss_fn(preds, X_batch[:, 0], X_batch[:, 1], dataset_labels)
        med_centile, gt_T = model.get_median_centile(X_batch, return_gt=True)
        med_loss = median_loss(med_centile, gt_T)
        loss += med_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

        all_centiles.append(preds.detach().cpu())
        all_sexes.append(X_batch[:, 1].detach().cpu())
        all_ages.append(X_batch[:, 0].detach().cpu())
        all_datasets.append(np.array(dataset_labels))
        all_alphas.append(alphas.detach().cpu())
        all_cs.append(cs.detach().cpu())

    centile_data_dict = {
        'centiles': torch.cat(all_centiles).flatten(),
        'sexes': torch.cat(all_sexes).flatten(),
        'ages': torch.cat(all_ages).flatten(),
        'datasets': np.array(all_datasets).flatten(),
        'alpha': torch.cat(all_alphas).flatten(),
        'c': torch.cat(all_cs).flatten()
    }

    return centile_data_dict, total_loss #/ len(dataloader.dataset)

def old_train_one_epoch(model, dataloader, optimizer, loss_fn, median_loss, lambda_mse=1.0):
    """
    Updated to run through each dataframe as well
    """
    model.train()
    total_loss = 0.0
    # Data collection arrays for logging
    all_centiles = []
    all_sexes = []
    all_ages = []
    all_datasets = []
    all_alphas = []
    all_cs = []
    for X_batch, dataset_labels in dataloader:
        X_batch = X_batch.to(DEVICE)
        optimizer.zero_grad()
        preds, alphas, cs = model(X_batch, return_alpha_c=True)
        #metric_predictions = model.get_centile_curves(X_batch[:, 0], X_batch[:, 1], centile=preds)
        #get the ages, sexes, and dataset information
        loss = loss_fn(preds, X_batch[:, 0], X_batch[:, 1], dataset_labels)
        #get the median centile
        med_centile, gt_T = model.get_median_centile(X_batch, return_gt=True)
        med_loss = median_loss(med_centile, gt_T)
        total_loss = loss + lambda_mse*med_loss
        total_loss.backward()
        optimizer.step()
        total_loss += total_loss.item() * X_batch.size(0)

        all_centiles.append(preds.detach().cpu())
        all_sexes.append(X_batch[:, 1].detach().cpu())
        all_ages.append(X_batch[:, 0].detach().cpu())
        all_datasets.append(np.array(dataset_labels))
        all_alphas.append(alphas.detach().cpu())
        all_cs.append(cs.detach().cpu())

    centile_data_dict = {
        'centiles': torch.cat(all_centiles).flatten(),
        'sexes': torch.cat(all_sexes).flatten(),
        'ages': torch.cat(all_ages).flatten(),
        'datasets': np.array(all_datasets).flatten(),
        'alpha': torch.cat(all_alphas).flatten(),
        'c': torch.cat(all_cs).flatten()
    }

    return centile_data_dict, total_loss #/ len(dataloader.dataset)

def eval_one_epoch(model, dataloader, loss_fn, median_loss, lambda_mse=1.0):
    model.eval()
    total_loss = 0.0
    # Data collection arrays for logging
    all_centiles = []
    all_sexes = []
    all_ages = []
    all_datasets = []
    all_alphas = []
    all_cs = []
    with torch.no_grad():
        for X_batch, dataset_labels in dataloader:
            X_batch = X_batch.to(DEVICE)
            preds, alphas, cs = model(X_batch, return_alpha_c=True)
            loss = loss_fn(preds, X_batch[:, 0], X_batch[:, 1], dataset_labels)
            med_centile, gt_T = model.get_median_centile(X_batch, return_gt=True)
            med_loss = median_loss(med_centile, gt_T)
            loss = loss + lambda_mse*med_loss
            total_loss += loss.item() * X_batch.size(0)
            all_centiles.append(preds.detach().cpu())
            all_ages.append(X_batch[:, 0].detach().cpu())
            all_sexes.append(X_batch[:, 1].detach().cpu())
            all_datasets.append(np.array(dataset_labels))
            all_alphas.append(alphas.detach().cpu())
            all_cs.append(cs.detach().cpu())
    
    centile_data_dict = {
        'centiles': torch.cat(all_centiles).flatten(),
        'ages': torch.cat(all_ages).flatten(),
        'sexes': torch.cat(all_sexes).flatten(),
        'datasets': np.array(all_datasets).flatten(),
        'alpha': torch.cat(all_alphas).flatten(),
        'c': torch.cat(all_cs).flatten()
    }
    return centile_data_dict, total_loss #/ len(dataloader.dataset)

def pdf_one_epoch(model, dataloader, optimizer, loss_fn, median_loss, args, eval=False):
    if eval:
        model.eval()
    else:
        model.train()
    final_loss = 0.0
    # Data collection arrays for logging
    (all_centiles, all_sexes, all_ages, all_datasets,
        all_alphas, all_cs, all_site_alphas, all_site_cs) = ([] for _ in range(8))
    context_manager = torch.no_grad() if eval else torch.autograd.set_detect_anomaly(True)
    with context_manager:
        for X_batch, dataset_labels in dataloader:
            X_batch = X_batch.to(DEVICE)
            if not eval:
                optimizer.zero_grad()
            if not args.decoupled:
                preds, alphas, cs = model(X_batch, return_alpha_c=True)
            else:
                preds, alphas, cs, c_unique_offset, alpha_unique_offset = model(X_batch, return_alpha_c=True, return_site_offsets=True, training=not eval)
                all_site_cs.append(c_unique_offset.detach().cpu())
                all_site_alphas.append(alpha_unique_offset.detach().cpu())
            #loss = loss_fn(alphas, cs, X_batch[:, -1])
            T_input = X_batch[:, -1] / model.normalizer_scalar
            loss = loss_fn(alphas, preds, cs, T_input)

            #median centile loss
            #med_centile, gt_T = model.get_median_centile(X_batch, return_gt=True)
            #med_loss = median_loss(med_centile, gt_T)
            ##med_centile = model._calc_centile_tract_values(torch.tensor([0.5]).to(DEVICE), alphas, cs)
            ##med_loss = median_loss(med_centile, T_input)
            med_loss = median_loss(cs, T_input)

            if args.verbose:
                print(f"PDF loss: {loss.item():.4f}, Median loss: {args.lambda_mse*med_loss.item():.4f}")

            total_loss = args.lambda_pdf*loss + args.lambda_mse*med_loss

            if not eval:
                total_loss.backward()
                optimizer.step()
            else:
                pass
            
            final_loss += total_loss.item() * X_batch.size(0)
            
            all_centiles.append(preds.detach().cpu())
            ages = X_batch[:, 0].detach().cpu()
            if args.age_transform:
                ages = architecture.apply_age_transformation(ages, reverse=True, norm=not args.no_age_norm)
            elif not args.no_age_norm:
                ages = architecture.apply_age_norm(ages, reverse=True)
            all_ages.append(ages)
            all_sexes.append(X_batch[:, 1].detach().cpu())
            all_datasets.append(np.array(dataset_labels))
            all_alphas.append(alphas.detach().cpu())
            all_cs.append(cs.detach().cpu())
            all_site_cs.append(c_unique_offset.detach().cpu())
            all_site_alphas.append(alpha_unique_offset.detach().cpu())
    
    total_loss = total_loss #/ len(dataloader.dataset)

    centile_data_dict = {
        'centiles': torch.cat(all_centiles).flatten(),
        'ages': torch.cat(all_ages).flatten(),
        'sexes': torch.cat(all_sexes).flatten(),
        'datasets': np.hstack(all_datasets),
        'alpha': torch.cat(all_alphas).flatten(),
        'c': torch.cat(all_cs).flatten(),
        'site_c': torch.cat(all_site_cs).flatten(),
        'site_alpha': torch.cat(all_site_alphas).flatten()
    }
    return centile_data_dict, total_loss

def pdf_one_epoch_dataset_batch(model, dataloader, optimizer, loss_fn, median_loss, args, eval=False):
    """
    PDF epoch train/eval function (using datasets as batches)
    """
    if eval:
        model.eval()
    else:
        model.train()
    total_loss = 0.0
    # Data collection arrays for logging
    (all_centiles, all_sexes, all_ages, all_datasets,
        all_alphas, all_cs, all_site_alphas, all_site_cs) = ([] for _ in range(8))
    for X_batch, dataset_labels in dataloader:
        X_batch = X_batch.to(DEVICE)
        if not eval:
            optimizer.zero_grad()
        batch_sample_count = 0
        for dataset_idx in range(X_batch.shape[1]-3): ## minus 3 for the age, sex, and Tract metric
            subbatch_idxs = torch.where(X_batch[:, dataset_idx+2] == 1)[0]
            X_subbatch = X_batch[subbatch_idxs, :]
            dataset_labels_subbatch = [dataset_labels[i] for i in subbatch_idxs.cpu().numpy()]
            context_manager = torch.no_grad() if eval else torch.autograd.set_detect_anomaly(True)
            with context_manager:
                if not args.decoupled:
                    preds, alphas, cs = model(X_subbatch, return_alpha_c=True, training=not eval)
                else:
                    preds, alphas, cs, c_unique_offset, alpha_unique_offset = model(X_subbatch, return_alpha_c=True, return_site_offsets=True, training=not eval)
            #metric_predictions = model.get_centile_curves(X_batch[:, 0], X_batch[:, 1], centile=preds)
            #get the ages, sexes, and dataset information
            #loss = loss_fn(alphas, cs, X_batch[:, -1])
                loss = loss_fn(alphas, preds, cs, X_subbatch[:, -1])

                #median centile loss
                med_centile, gt_T = model.get_median_centile(X_subbatch, return_gt=True)
                med_loss = median_loss(med_centile, gt_T)

                if args.verbose:
                    print(f"PDF loss: {loss.item():.4f}, Median loss: {args.lambda_mse*med_loss.item():.4f}")
                #loss = loss + args.lambda_mse*med_loss
                final_loss = loss + args.lambda_mse*med_loss
        
            if not eval:
                scaled_loss = final_loss / X_subbatch.size(0)
                scaled_loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #to stabilize the training
                #optimizer.step()
            total_loss += final_loss.item()
            batch_sample_count += X_subbatch.size(0)
            #total_loss += loss.item() * X_subbatch.size(0)

            all_centiles.append(preds.detach().cpu())
            all_sexes.append(X_subbatch[:, 1].detach().cpu())
            all_ages.append(X_subbatch[:, 0].detach().cpu())
            all_datasets.append(np.array(dataset_labels_subbatch))
            all_alphas.append(alphas.detach().cpu())
            all_cs.append(cs.detach().cpu())
            if args.decoupled:
                all_site_cs.append(c_unique_offset.detach().cpu())
                all_site_alphas.append(alpha_unique_offset.detach().cpu())
        
        if not eval and batch_sample_count > 0:
            nan_grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None and p.grad.isnan().any()]
            if len(nan_grads) > 0:
                 print(f"!!! {len(nan_grads)} LAYERS HAVE NaN GRADIENTS. Check backward pass. !!!")
                 # Optional: Raise an exception to stop
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            optimizer.step()

    total_loss = total_loss / batch_sample_count

    centile_data_dict = {
        'centiles': torch.cat(all_centiles).flatten(),
        'sexes': torch.cat(all_sexes).flatten(),
        'ages': torch.cat(all_ages).flatten(),
        'datasets': np.hstack(all_datasets),
        'alpha': torch.cat(all_alphas).flatten(),
        'c': torch.cat(all_cs).flatten(),
    }
    if args.decoupled:
        centile_data_dict['site_c'] = torch.cat(all_site_cs).flatten(),
        centile_data_dict['site_alpha'] = torch.cat(all_site_alphas).flatten()

    return centile_data_dict, total_loss

def start_model_training(datacsv, outdir, args):
    #assumes that 'fold' is in the csv
    df_orig = pd.read_csv(datacsv)

    if args.pretrain_young:
        #filter to only ages <= 10
        df_orig = df_orig[df_orig['age'] <= 10].reset_index(drop=True)

    metric_col = [x for x in df_orig.columns if x not in ['age', 'sex', 'dataset', 'fold', 'diagnosis', 'subject']]
    
    dataset_names = sorted(df_orig['dataset'].unique().tolist()) 

    #train each fold individually
    for fold in sorted(df_orig['fold'].unique()):

        #see if the output already exists
        pref = f"magce_model_fold{fold}"
        if any([pref in x.name for x in outdir.iterdir()]):
            existing_file = [x.name for x in outdir.iterdir() if pref in x.name][0]
            print(f"Fold {fold} already trained: {existing_file} Skipping.")
            continue

        #split the dataframe into train and val
        df_train = df_orig[df_orig['fold'] != fold]
        df_val = df_orig[df_orig['fold'] == fold]
        if args.specific_dataset != '':
            df_train = df_train[df_train['dataset'] == args.specific_dataset].reset_index(drop=True)
            df_val = df_val[df_val['dataset'] == args.specific_dataset].reset_index(drop=True)
            print(df_train.shape)
            print(df_val.shape)
            assert len(df_train) > 0, f"No training data for fold {fold} with specific_dataset={args.specific_dataset}"
            assert len(df_val) > 0, f"No validation data for fold {fold} with specific_dataset={args.specific_dataset}"
        train_dataset = architecture.MAGCEDataset(df_train, age_transform=args.age_transform, age_norm=not args.no_age_norm)
        val_dataset = architecture.MAGCEDataset(df_val, age_transform=args.age_transform, age_norm=not args.no_age_norm)
        #get the batch size
        if args.equal_3_batch:
            train_batch_size = math.ceil(train_dataset.X.shape[0] / 3)
        elif args.batch_size == -1:
            train_batch_size = train_dataset.X.shape[0]
        else:
            train_batch_size = args.batch_size
        
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.X.shape[0], shuffle=False)

        #define model and other stuff
        if args.decoupled:
            model = architecture.DecoupledMAGCE(input_dim=train_dataset.X.shape[1]-1, site_dropout_p=args.site_dropout_prob).to(DEVICE) #one less than input size because last column is T
        else:
            model = architecture.MAGCE(input_dim=train_dataset.X.shape[1]-1).to(DEVICE) #one less than input size because last column is T
        model.set_normalizing_scalar(df_orig[metric_col].values)
        optimizer = optim.Adam(model.parameters(), lr=LEARNINGRATE)
        if not args.pdf_loss:
            loss_fn = architecture.CentileLoss(ref_min=args.reference_age_min, ref_max=args.reference_age_max,
                                            ref_step=args.reference_age_step_size, log_ages=args.logspace_reference,
                                            age_transform=args.age_transform, wass_power=args.wass_power,
                                            dataset_weighting=args.dataset_weighting, age_varying_kernel=args.age_varying_kernel
                                            ).to(DEVICE)
        else:
            loss_fn = architecture.PDFLoss().to(DEVICE)
        median_loss = architecture.MedianLoss().to(DEVICE)

        #wandb init
        if not args.no_wandb:
            wandb.init(
                project="MAGCE_Deep_Learning",
                name=f"{args.wandb_name}_fold_{fold}",
                config={
                    "learning_rate": LEARNINGRATE,
                    "fold": fold,
                    "device": str(DEVICE),
                    "wass_power": args.wass_power,
                    "reference_age_min": args.reference_age_min,
                    "reference_age_max": args.reference_age_max,
                    "reference_age_step_size": args.reference_age_step_size,
                    "logspace_reference": args.logspace_reference,
                    "age_transform": args.age_transform,
                    "dataset_weighting": args.dataset_weighting,
                    "age_varying_kernel": args.age_varying_kernel,
                    "lambda_mse": args.lambda_mse,
                    "pdf_loss": args.pdf_loss,
                    "specific_dataset": args.specific_dataset,
                    "dataset_batch": args.dataset_batch,
                    "age_norm": not args.no_age_norm,
                    "site_dropout_prob": args.site_dropout_prob,
                    "batch_size": train_batch_size
                }
            )


        val_min = float('inf')
        val_low_flag = 0
        epoch = 0
        #prev_outname = 'none.pt'
        while val_low_flag < 20:
            epoch += 1
            #train_centiles_data, train_loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn)
            if not args.pdf_loss:
                train_centiles_data, train_loss = old_train_one_epoch(model, train_dataloader, optimizer, loss_fn, median_loss, lambda_mse=args.lambda_mse)
                val_centiles_data, val_loss = eval_one_epoch(model, val_dataloader, loss_fn, median_loss, lambda_mse=args.lambda_mse)
            else:
                if args.dataset_batch:
                    train_centiles_data, train_loss = pdf_one_epoch_dataset_batch(model, train_dataloader, optimizer, loss_fn, median_loss, args)
                    val_centiles_data, val_loss = pdf_one_epoch_dataset_batch(model, val_dataloader, optimizer, loss_fn, median_loss, args, eval=True)
                else:
                    train_centiles_data, train_loss = pdf_one_epoch(model, train_dataloader, optimizer, loss_fn, median_loss, args)
                    val_centiles_data, val_loss = pdf_one_epoch(model, val_dataloader, optimizer, loss_fn, median_loss, args, eval=True)
            print(f"Fold {fold} Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if not args.no_wandb:
                wandb.log({
                    "Train/Avg_Loss": train_loss,
                    "Val/Avg_Loss": val_loss,
                    "epoch": epoch
                }, step=epoch)
            
            # --- W&B HISTOGRAM LOGGING ---
            # NOTE: Assumes dataset_labels (tensors) are numerical IDs 0, 1, 2...
            # corresponding to the indices of dataset_names list.
            if not args.no_wandb and epoch % 20 == 0:
                log_centile_histograms(model, epoch, "Train", train_centiles_data, dataset_names, fold, df_train, args)
                log_centile_histograms(model, epoch, "Val", val_centiles_data, dataset_names, fold, df_val, args)

            if val_loss < val_min:
                val_min = val_loss
                val_low_flag = 0
                outname = f"magce_model_fold{fold}_epoch{epoch}.pt"
                #torch.save(model.state_dict(), outdir/outname)
                best_model_state = model.state_dict().copy()
            else:
                val_low_flag += 1

        #save the model and log it
        torch.save(best_model_state, outdir/outname)
        if not args.no_wandb:
            wandb.finish()
        

def pa():
    p = argparse.ArgumentParser(description="Train MAGCE Deep Learning Model")
    p.add_argument("--datacsv", type=str, required=True, help="Path to CSV file containing data with fold assignments")
    p.add_argument("--outdir", type=str, required=True, help="Output directory to save trained models")
    p.add_argument("--wandb_name", type=str, default="", help="Name for the wandb run")
    p.add_argument("--reference_age_min", type=float, default=0, help="Minimum reference age for age wasserstein-weightings")
    p.add_argument("--reference_age_max", type=float, default=100, help="Maximum reference age for age wasserstein-weightings")
    p.add_argument("--reference_age_step_size", type=float, default=2, help="Step-size of reference ages for age wasserstein-weightings")
    p.add_argument("--logspace_reference", action="store_true", help="Have the reference ages be log-spaced to better cover the beginning of lifespan")
    p.add_argument("--wass_power", type=float, default=1.0, help="Power to raise the wasserstein distances to when computing the loss")
    p.add_argument("--age_transform", action="store_true", help="Apply age transformation for log-spaced ages post conception")
    p.add_argument("--dataset_weighting", action="store_true", help="Use dataset weighting (size-based) in the loss function")
    p.add_argument("--age_varying_kernel", action="store_true", help="Use age-varying kernel in the wasserstein loss function - max value of k=1 (not implemented)")
    p.add_argument("--lambda_mse", type=float, default=1, help="How much to weight the MSE loss for the median centile")
    p.add_argument("--lambda_pdf", type=float, default=1, help="How much to weight the PDF loss (if used)")
    p.add_argument("--pretrain_young", action="store_true", help="Pretrain on young subjects first (not implemented; currently just drops other ages)")
    p.add_argument("--specific_dataset", type=str, default='', help="If specified, only use this dataset for training")
    p.add_argument("--dataset_batch", action="store_true", help="Use dataset-wise batching for PDF loss)")
    p.add_argument("--batch_size", type=int, default=-1, help="Batch size for training (default: full batch)")
    p.add_argument("--pdf_loss", action="store_true", help="Use PDF-based loss instead of CDF-based loss")
    p.add_argument("--decoupled", action="store_true", help="Use decoupled network")
    p.add_argument("--site_dropout_prob", type=float, default=0.0, help="Dropout probability for site-specific parameters")
    p.add_argument("--no_age_norm", action="store_true", help="Disable age normalization in the model")
    p.add_argument("--equal_3_batch", action="store_true", help="Make the batch sizes consistent for three steps every epoch")
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--verbose", action="store_true", help="Enable verbose output during training")
    return p.parse_args()

if __name__ == "__main__":
    args = pa()
    outdir = Path(args.outdir)
    assert outdir.exists(), f"Output directory {outdir} does not exist"
    assert Path(args.datacsv).exists(), f"Data CSV file {args.datacsv} does not exist"
    #make sure outdir is empty
    #assert len(list(outdir.iterdir())) == 0, f"Output directory {outdir} is not empty"
    start_model_training(args.datacsv, Path(args.outdir), args)


# def pdf_one_epoch_dataset_batch(model, dataloader, optimizer, loss_fn, median_loss, args, eval=False):
#     """
#     PDF epoch train/eval function (using datasets as batches)
#     """
#     if eval:
#         model.eval()
#     else:
#         model.train()
#     total_loss = 0.0
#     # Data collection arrays for logging
#     all_centiles = []
#     all_sexes = []
#     all_ages = []
#     all_datasets = []
#     all_alphas = []
#     all_cs = []
#     all_site_cs = []
#     all_site_alphas = []
#     all_preds = []
#     for X_batch, dataset_labels in dataloader:
#         X_batch = X_batch.to(DEVICE)
#         if not eval:
#             optimizer.zero_grad()
#         batch_sample_count = 0
#         for dataset_idx in range(X_batch.shape[1]-3): ## minus 3 for the age, sex, and Tract metric
#             subbatch_idxs = torch.where(X_batch[:, dataset_idx+2] == 1)[0]
#             X_subbatch = X_batch[subbatch_idxs, :]
#             dataset_labels_subbatch = [dataset_labels[i] for i in subbatch_idxs.cpu().numpy()]
#             context_manager = torch.no_grad() if eval else torch.autograd.set_detect_anomaly(True)
#             with context_manager:
#                 if not args.decoupled:
#                     preds, alphas, cs = model(X_subbatch, return_alpha_c=True, training=not eval)
#                 else:
#                     preds, alphas, cs, c_unique_offset, alpha_unique_offset = model(X_subbatch, return_alpha_c=True, return_site_offsets=True, training=not eval)
#             #metric_predictions = model.get_centile_curves(X_batch[:, 0], X_batch[:, 1], centile=preds)
#             #get the ages, sexes, and dataset information
#             #loss = loss_fn(alphas, cs, X_batch[:, -1])
#                 loss = loss_fn(alphas, preds)

#                 #median centile loss
#                 med_centile, gt_T = model.get_median_centile(X_subbatch, return_gt=True)
#                 med_loss = median_loss(med_centile, gt_T)

#                 if args.verbose:
#                     print(f"PDF loss: {loss.item():.4f}, Median loss: {args.lambda_mse*med_loss.item():.4f}")
#                 #loss = loss + args.lambda_mse*med_loss
#                 final_loss = loss + args.lambda_mse*med_loss
        
#             if not eval:
#                 scaled_loss = final_loss / X_subbatch.size(0)
#                 scaled_loss.backward()
#                 #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #to stabilize the training
#                 #optimizer.step()
#             total_loss += final_loss.item()
#             batch_sample_count += X_subbatch.size(0)
#             #total_loss += loss.item() * X_subbatch.size(0)

#             all_centiles.append(preds.detach().cpu())
#             all_sexes.append(X_subbatch[:, 1].detach().cpu())
#             all_ages.append(X_subbatch[:, 0].detach().cpu())
#             all_datasets.append(np.array(dataset_labels_subbatch))
#             all_alphas.append(alphas.detach().cpu())
#             all_cs.append(cs.detach().cpu())
#             all_site_cs.append(c_unique_offset.detach().cpu())
#             all_site_alphas.append(alpha_unique_offset.detach().cpu())
        
#         if not eval and batch_sample_count > 0:
#             nan_grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None and p.grad.isnan().any()]
#             if len(nan_grads) > 0:
#                  print(f"!!! {len(nan_grads)} LAYERS HAVE NaN GRADIENTS. Check backward pass. !!!")
#                  # Optional: Raise an exception to stop
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
#             optimizer.step()

#     total_loss = total_loss / batch_sample_count