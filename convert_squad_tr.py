from __future__ import annotations

import os
import uuid

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


repo_name = "selmanbaysan/squad_tr_v2"
create_repo(repo_name, repo_type="dataset")


raw_dset = load_dataset("boun-tabi/squad_tr")
dset = raw_dset["validation"]
trimmed_dataset = dset.select(range(2048))
updated_dataset = trimmed_dataset.map(
    preprocess_data, remove_columns=["question", "context", "answers"]
)
corpus_ds = updated_dataset.map(
    lambda example: {"_id": example["corpus-id"], "text": example["answer_text"], "title": example["title"]},
    remove_columns=["id", "query-id", "query_text", "corpus-id", "answer_text"],
)
default_ds = updated_dataset.map(
    lambda example: example, remove_columns=["id", "title", "answer_text", "query_text"]
)
default_ds = default_ds.add_column("score", len(corpus_ds) * [1])
queries_ds = updated_dataset.map(
    lambda example: {"_id": example["query-id"], "text": example["query_text"]},
    remove_columns=["id", "corpus-id", "answer_text", "query-id", "query_text", "title"],
)
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