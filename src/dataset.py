import pandas as pd
import numpy as np
from utils import download_file, extract_zip
import os
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# downloading loading the dataset and store in ./dataset folder
url = "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL.zip"
download_path = "FLAT_RCL.zip"
extract_path = "../dataset"

download_file(url, download_path)
extract_zip(download_path, extract_path)
os.remove(download_path)
print(f"Cleaned up temporary file: {download_path}")


# ---------- preprocessing the dataset ----------
input_file = "../dataset/FLAT_RCL.txt"

# defining temporary column names based on .ipynb file
columns = [f"Column_{i}" for i in range(27)]
df = pd.read_csv(input_file, delimiter="\t", header=None, names=columns, on_bad_lines='skip', engine='python')

# selecting columns required and dropping unnecessary ones
df = df[['Column_2', 'Column_3', 'Column_4', 'Column_19', 'Column_20', 'Column_21']]

# renaming the columns for better accessibility
df.rename(columns={'Column_2': 'make',
                   'Column_3': 'model',
                   'Column_4': 'year',
                   'Column_19': 'defect',
                   'Column_20': 'consequence',
                   'Column_21': 'corrective'}, inplace=True)

# combining columns defect, consequences and corrective into one column for creating embeddings
df['combined_text'] = df['defect'] + ' ' + df['consequence'] + ' ' + df['corrective']
df.drop(['defect', 'consequence', 'corrective'], axis=1, inplace=True)

# selecting the subset of dataset that contains only Ford or Toyota examples
df = df[(df['make'] == 'FORD') | (df['make'] == 'TOYOTA')]

# removing the null values and saving the dataset as checkpoint
df = df.dropna()
df.to_csv('../dataset/filtered_df.csv', index=False)

# creating a column for embedding
df['embedding']=None
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device)
embedding_model.to(device)

for i in tqdm(range(len(df)), desc="Encoding Embeddings"):
    df.loc[i, 'embedding'] = embedding_model.encode(df['combined_text'].iloc[i])

print("Embeddings generated successfully.")

# saving the embedded dataset as checkpoint
df.to_csv('../dataset/final_embedded_dataset.csv')

print("Dataset preprocessing completed!")