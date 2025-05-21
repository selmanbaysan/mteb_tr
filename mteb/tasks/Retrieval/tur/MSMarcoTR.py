from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class MSMarcoTRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSMarcoTRRetrieval",
        dataset={
            "path": "trmteb/msmarco-tr",
            "revision": "main",
        },
        description="Turkish version of MSMARCO Passage Ranking dataset. This dataset is a machine translated version of the original MSMARCO dataset into Turkish.",
        reference="https://huggingface.co/datasets/parsak/msmarco-tr",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=None,
        task_subtypes=None,
        domains=None,
        license=None,
        annotations_creators=None,
        dialect=[],
        sample_creation=None,
        bibtex_citation="""
        @article{DBLP:journals/corr/NguyenRSGTMD16,
        author       = {Tri Nguyen and
                        Mir Rosenberg and
                        Xia Song and
                        Jianfeng Gao and
                        Saurabh Tiwary and
                        Rangan Majumder and
                        Li Deng},
        title        = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
        journal      = {CoRR},
        volume       = {abs/1611.09268},
        year         = {2016},
        url          = {http://arxiv.org/abs/1611.09268},
        eprinttype    = {arXiv},
        eprint       = {1611.09268},
        timestamp    = {Thu, 11 Apr 2024 13:33:57 +0200},
        biburl       = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
        bibsource    = {dblp computer science bibliography, https://dblp.org}
        """,
        prompt={
            "query": "Bir web arama sorgusu verildiğinde, sorguyu yanıtlayan ilgili metinleri getir"
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
        
        corpus_ds = datasets.load_dataset(self.metadata_dict["dataset"]["path"], "corpus", split="corpus")
        queries_ds = datasets.load_dataset(self.metadata_dict["dataset"]["path"], "queries", split="queries")

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

            corpus_ds = corpus_ds.filter(lambda doc: str(doc["_id"]) in corpus_ids)
            queries_ds = queries_ds.filter(lambda query: str(query["_id"]) in queries_ids)
            
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            for doc in corpus_ds:
                self.corpus[split][str(doc["_id"])] = {"title": "", "text": doc["text"]}
                
            for query in queries_ds:
                self.queries[split][str(query["_id"])] = query["text"]
            
            for rel in ds:
                query_id = rel["query-id"]
                doc_id = rel["corpus-id"]
                self.relevant_docs[split][query_id] = {}
                self.relevant_docs[split][query_id][doc_id] = 1
        
        self.data_loaded = True
            