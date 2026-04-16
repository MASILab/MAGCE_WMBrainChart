#do the k-fold cross validation
#save the dataframe as a CSV with the folds as a new row
    #maybe just write another script to create the folds and save as a new CSV


from sklearn.model_selection import StratifiedKFold
import pandas as pd
import argparse
from pathlib import Path

def pa():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=str, help="Path to input CSV file")
    p.add_argument("output_csv", type=str, help="Path to output CSV file with folds")
    p.add_argument("--n_splits", type=int, default=5, help="Number of folds for StratifiedKFold")
    return p.parse_args()

def get_folds(df, n_splits=5, verbose=False):
    stratify_labels = df["dataset"].astype(str) + "_" + df["sex"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    df['fold'] = -1
    #set the fold numbers for each row
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, stratify_labels)):
        df.loc[val_idx, "fold"] = fold+1
    
    #for each fold, print out the number of 0 and 1 in the sex column for each dataset
    if verbose:
        for fold in df['fold'].unique():
            fdf = df[df['fold'] == fold]
            for dataset in df['dataset'].unique():
                ddf = fdf[fdf['dataset'] == dataset]
                sdct = ddf['sex'].value_counts().to_dict()
                print(f"{dataset}: {sdct}")
            print("****************")
    
    return df

args = pa()

df = pd.read_csv(args.input_csv)
assert not Path(args.output_csv).exists(), f"Output CSV {args.output_csv} already exists!"
df_folds = get_folds(df, n_splits=args.n_splits, verbose=False)

df_folds.to_csv(args.output_csv, index=False)

#t = pd.read_csv("/fs5/p_masi/kimm58/WMLifespan/data/LifespanExtensionModelFits/Revisions_CN_dataframes/AF_left_fa-mean_CN.csv")
#get_folds(t, n_splits=5)
