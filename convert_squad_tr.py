from __future__ import annotations

import os
import uuid

import datasets
from datasets import load_dataset
from huggingface_hub import create_repo, upload_file


def preprocess_data(example: dict) -> dict:
    """Preprocessed the data in a format easier
    to handle for the loading of queries and corpus
    ------
    PARAMS
    example : element in med-qa dataset
    """
    return {
        "query-id": str(uuid.uuid4()),
        "query_text": example["question"],
        "corpus-id": str(uuid.uuid4()),
        "answer_text": example["context"],
    }

def create_retrieval_datasets(dataset: datasets.Dataset):
    dataset = dataset["train"]["data"]
    corpus = []
    queries = []
    default = []
    corpus_id = 0
    for row in dataset:
        title = row["title"]
        for paragraph in row["paragraphs"]:
            context = paragraph["context"]
            corpus.append({"_id": corpus_id, "title": title, "text": context})
            for qas in paragraph["qas"]:
                question = qas["question"]
                queries.append({"_id": qas["id"], "text": question})
                default.append({"query-id": qas["id"], "corpus-id": corpus_id, "score": 1})
            corpus_id += 1
    return corpus, queries, default



repo_name = "selmanbaysan/squad-tr-v2"
create_repo(repo_name, repo_type="dataset")


#raw_dset = load_dataset("boun-tabi/squad_tr")
#dset = raw_dset["validation"]
dset = load_dataset("json", data_files="squad-tr-dev-v1.0.0.json")

corpus, queries, default = create_retrieval_datasets(dset)
print(len(corpus), len(queries), len(default))
corpus_ds = datasets.Dataset.from_list(corpus)
default_ds = datasets.Dataset.from_list(default)
queries_ds = datasets.Dataset.from_list(queries)

data = {"corpus": corpus_ds, "default": default_ds, "queries": queries_ds}

for splits in ["default", "queries", "corpus"]:
    save_path = f"{splits}.jsonl"
    data[splits].to_json(save_path)
    upload_file(
        path_or_fileobj=save_path,
        path_in_repo=save_path,
        repo_id=repo_name,
        repo_type="dataset",
    )
    os.system(f"rm {save_path}")
