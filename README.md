## Introduction
This repo perform similarity search using RAG model and return the summary of all the documents matched. The input are in the form of .json and stored in `input/ouput`.

## There are two ways of running this repository.
1. Using Jupyter Notebook (preferably on kaggle)
2. Using python script (need to run on kaggle for summarization only)

### Running Jupyter Notebook
Download main.ipynb file from src folder and run on any Jupyter Notebook.<br/>
Or can be run directly on kaggle by clicking [here](https://www.kaggle.com/code/mltensor/testing-assignment/edit).
> **_NOTE:_**  This file need to import `Llama-3.1-8b-V2` version of Llama LLM.

### Running Python Script
Clone the repository.
#### Step 1
```
pip install -r requirements.txt
```
#### Step 2
Downloading, cleaning and vectorising the dataset (needed to be done once).
```
cd src
```
```
python dataset.py
```
#### Step 3
Retrieving the data from the document. The input is taken from input_output folder -> `query.json` in .json format.
```
python main.py
```
#### Step 4
The above code returns the output in the text format, which is stored in `input_output/matched_record.txt`.<br/>
The text needed to be copied and paste as input in [Jupyter Notebook](https://www.kaggle.com/code/mltensor/assignment/edit/run/214007757).<br/>
>**_NOTE:_** Above step needed `Llama-3.1-8b-V2` version of Llama LLM.

## Explanation
The process begins with downloading and unzipping the dataset. The data is then cleaned, which involves selecting relevant columns, removing unnecessary ones based on analysis, handling null values, and filtering the dataset to include only Ford and Toyota entries.<br/>
Next, embeddings are generated for the processed data, added as a new column, and saved as the final dataset.<br/>
This dataset is then used for searching based on user input, which includes value-based searching (e.g., make, model, and year) followed by similarity-based search for precise results.<br/>
Finally, the search results are passed to an LLM for summarization.<br/>
