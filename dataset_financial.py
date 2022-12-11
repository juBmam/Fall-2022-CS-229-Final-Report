import pickle
from datetime import datetime
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def pricing_check(row, dataframe,):
    index = int(row.name)
    if (dataframe['Pricing'][index] == False):
        return np.nan
    return dataframe['Px'][index]


def parse_id(dataframe, missing_ratio=0.1, datasettype=None):

    observed_values = []


    if datasettype.lower() == 'test':
        observed_df = dataframe
        miss_indices = list(np.where(dataframe['Pricing'] == False)[0])
        
    if datasettype.lower() == 'train':
        observed_df = dataframe.loc[dataframe['Pricing'] == True]
        miss_indices = list(np.where(dataframe['Time'] >= '11:30:00')[0])
    
    observed_df = observed_df[[ 'Weight', 'beta', 'EURUSD', 'EWQ', 'Px']]
    for index, row in observed_df.iterrows():
        observed_values.append([row['Px'], row['beta'],row['EWQ'], row['EURUSD'], row['Weight']])
        #observed_values.append([row['Px'], row['EWQ']])
        
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()

        
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    return observed_values, observed_masks, gt_masks


tickers = ['SAN']

df = pd.read_csv('/content/drive/My Drive/CSDI-main/BetaValPricing.csv')
df = df[df['Ticker_holding'].isin(tickers)]
df = df[['ts', 'Ticker_holding', 'Weight', 'beta', 'EURUSD', 'EWQ', 'Px', 'Pricing']]
df['Time'] = pd.to_datetime(df['ts']).dt.strftime('%H:%M:%S')
df = df.loc[df['Time'] >= '09:30:00'].reset_index()
df['Px'] = df.apply(pricing_check, args=(df,), axis=1)



class Financial_Dataset(Dataset):
    def __init__(self, eval_length=3192, use_index_list=None, missing_ratio=0.0, seed=0): 
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = ("/content/drive/My Drive/CSDI-main/data/financial_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk")
        
        if os.path.isfile(path) == False:  # if datasetfile is none, create
  
            observed_values, observed_masks, gt_masks = parse_id(df, missing_ratio, 'train')
            self.observed_values.append(observed_values)
            self.observed_masks.append(observed_masks)
            self.gt_masks.append(gt_masks)
            
            observed_values, observed_masks, gt_masks = parse_id(df, missing_ratio, 'test')
            self.observed_values.append(observed_values)
            self.observed_masks.append(observed_masks)
            self.gt_masks.append(gt_masks)
     
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)
     
                # calc mean and std and normalize values

            self.observed_values = (self.observed_values * self.observed_masks)
        else:
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(f)
    
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list


    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):

    train_index = [0]
    test_index = [1]

    print('debug checkpoint')
    dataset = Financial_Dataset(eval_length=1816, use_index_list=train_index, missing_ratio=missing_ratio, seed=seed)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    print('debug checkpoint 2')
    test_dataset = Financial_Dataset(eval_length=1376 + 1816, use_index_list=test_index, missing_ratio=missing_ratio, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, test_loader