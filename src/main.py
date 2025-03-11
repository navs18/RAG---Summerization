import torch
from sentence_transformers import util, SentenceTransformer
import pandas as pd
import numpy as np
import json


# loading the final dataset for searching and query by user
df = pd.read_csv('../dataset/final_embedded_dataset.csv')
df["embedding"] = df["embedding"].apply(lambda x: torch.tensor(eval(x), dtype=torch.float32))

with open('../input_output/query.json', "r") as json_file:
    query_input = json.load(json_file)

# defining the embedding model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                    device = device)


# ----------------------- Searching -----------------------

# Filter dataset based on 'make', 'model', and 'year'
filtered_df = df[
    (df["make"].str.lower() == query_input["make"].lower()) &
    (df["model"].str.lower() == query_input["model"].lower()) &
    (df["year"] == int(query_input["year"]))
]

if filtered_df.empty:
    print("No matching records found for the given make, model, and year.")
else:
    # Extract embeddings for the filtered dataset
    embeddings = torch.stack(filtered_df["embedding"].tolist()).to(device)

    # Encode the query issue
    query = query_input["issue"]
    print(f"Query: {query}")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)

    # Perform similarity search
    from time import perf_counter as timer
    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
    end_time = timer()

    # Get top k results
    top_results = torch.topk(dot_scores, k=min(1, len(embeddings)))

    # Display results
    print(f"Query: {query_input}")
    print("Results:")
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"Score: {score:.4f}")
        print("Matched Record:")
        print(filtered_df.iloc[int(idx)])
        print()
    
    # saving the searched item into text for summarization over kaggle notebook
    with open("../input_output/matched_record.txt", "w") as file:
        for score, idx in zip(top_results[0], top_results[1]):
            file.write(f"Searched Score: {score} \n")
            file.write(f"Make: {filtered_df['make'].iloc[int(idx)]} \n")
            file.write(f"Model: {filtered_df['model'].iloc[int(idx)]} \n")
            file.write(f"Year: {filtered_df['year'].iloc[int(idx)]} \n")
            file.write(f"Summary: {filtered_df['combined_text'].iloc[int(idx)]}")
            file.write("\n\n")

print("Seraching completed, file saved at ../input_output/matched_records.txt")
print("Please paste the `matched_record.txt` to kaggle notebook link provided.")