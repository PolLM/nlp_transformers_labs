#%%
# Load the emotion dataset
# !!! The dataset contains custom code, possible security risk for unknown datasets
from datasets import load_dataset
emotions = load_dataset("emotion", trust_remote_code=True)
emotions
# %%
# Select the train split and exploring the methods and attributes
train_ds = emotions['train']
print(train_ds)
print('\nDataset info:')
print(train_ds.info)
print('\nDataset length:')
print(len(train_ds))
print('\nDataset first item:')
print(train_ds[0])
print('\nDataset first item keys:')
print(train_ds[0].keys())
print('\nDataset features:')
print(train_ds.features)
print('\nDataset slice :5 :')
print(train_ds[:5])
print('\nDataset slice of text :5 :')
print(train_ds['text'][:5])
# %%
# From Datasets to DataFrames to plot and explore the data
import pandas as pd
emotions.set_format(type='pandas')
df = emotions['train'][:]
df.head()
# %%
