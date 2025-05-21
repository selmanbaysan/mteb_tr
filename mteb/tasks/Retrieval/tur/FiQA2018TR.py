from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

import datasets


class FiQA2018TR(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="FiQA2018TR",
        description="Turkish machine translated version of the FiQA 2018. Financial Opinion Mining and Question Answering",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "trmteb/fiqa-tr",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{
thakur2021beir,
title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
year={2021},
url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}""",
        prompt={
            "query": "Bir finansal soru verildiğinde, soruyu en iyi yanıtlayan kullanıcı yanıtlarını alın"
        },
    )

    def load_data(self, **kwargs) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = dict[doc_id, dict[str, str]] #id => dict with document datas like title and text
        self.queries = dict[query_id, str] #id => query
        self.relevant_docs = dict[query_id, dict[[doc_id, score]]
        """
        if self.data_loaded:
            return
        

        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        
        corpus_ds = datasets.load_dataset(self.metadata_dict["dataset"]["path"], "corpus")
        queries_ds = datasets.load_dataset(self.metadata_dict["dataset"]["path"], "queries")

        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}

        for split in self.metadata_dict["eval_splits"]:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
            ds = ds.shuffle(seed=42)
            max_samples = min(2048, len(ds))
            ds = ds.select(
                range(max_samples)
            )  # limit the dataset size to make sure the task does not take too long to run
            
            # filter the corpus_ds and queries_ds to only include the ids in the current split
            corpus_ids = set(ds["corpus-id"])
            queries_ids = set(ds["query-id"])

            corpus_ds = corpus_ds.filter(lambda x: x["_id"] in corpus_ids)
            queries_ds = queries_ds.filter(lambda x: x["_id"] in queries_ids)

            corpus_ds = corpus_ds["corpus"]
            queries_ds = queries_ds["queries"]

            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}
            
            for doc in corpus_ds:
                self.corpus[split][doc["_id"]] = {"title": "", "text": doc["text"]}
                
            for query in queries_ds:
                self.queries[split][query["_id"]] = query["text"]
            
            for rel in ds:
                query_id = rel["query-id"]
                doc_id = rel["corpus-id"]
                self.relevant_docs[split][query_id] = {}
                self.relevant_docs[split][query_id][doc_id] = 1
            
        self.data_loaded = True

