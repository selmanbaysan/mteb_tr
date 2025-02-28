from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FiQA2018TR(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="FiQA2018TR",
        description="Turkish machine translated version of the FiQA 2018. Financial Opinion Mining and Question Answering",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "selmanbaysan/fiqa-tr",
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
