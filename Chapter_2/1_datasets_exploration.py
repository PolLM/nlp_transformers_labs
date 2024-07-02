#%%
### List datasets ###
#!pip install datasets

from datasets import list_datasets
all_datasets = list_datasets()
print(f"Total datasets: {len(all_datasets)}")
# %%
### List datasets and models using huggingface hub (new approach) ###
# FutureWarning: list_datasets is deprecated and will be removed in the next major version of datasets. Use 'huggingface_hub.list_datasets' instead 
# -> https://huggingface.co/docs/huggingface_hub/main/en/guides/search

from huggingface_hub import HfApi
hf_api = HfApi()

# list_datasets and list_models return a generator object
# one can apply filters to the generator object: 
# -> https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_datasets

datasets = hf_api.list_datasets(
    filter="task_categories:image-classification",
    sort="downloads", 
    direction=-1, 
    limit=5
)
print("Top 5 downloaded datasets for image classification:")
for dataset in datasets:
    print(dataset)
print('\n')

models = hf_api.list_models(
	task="image-classification",
	library="pytorch",
	trained_dataset="imagenet",
)
print(f"Total pytorch models for image classification with imagenet: {len(list(models))}")

# %%
