from __future__ import annotations
import datasets
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackGamingRetrievalTR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackGamingRetrievalTR",
        description="Turkish machine translated version of the CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        dataset={
            "path": "trmteb/cqadupstack-gaming-tr",
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
        bibtex_citation="""@inproceedings{hoogeveen2015,
author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
series = {ADCS '15},
year = {2015},
isbn = {978-1-4503-4040-3},
location = {Parramatta, NSW, Australia},
pages = {3:1--3:8},
articleno = {3},
numpages = {8},
url = {http://doi.acm.org/10.1145/2838931.2838934},
doi = {10.1145/2838931.2838934},
acmid = {2838934},
publisher = {ACM},
address = {New York, NY, USA},
}""",
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
