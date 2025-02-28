from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class QuoraRetrievalTR(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="QuoraRetrievalTR",
        dataset={
            "path": "selmanbaysan/quora-tr",
            "revision": "main",
        },
        description=(
            "Turkish machine translated version of the QuoraRetrieval. QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
            + " question, find other (duplicate) questions."
        ),
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        type="Retrieval",
        category="s2s",
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
        bibtex_citation="""@misc{quora-question-pairs,
    author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
    title = {Quora Question Pairs},
    publisher = {Kaggle},
    year = {2017},
    url = {https://kaggle.com/competitions/quora-question-pairs}
}""",
        prompt={
            "query": "Bir soru verildiğinde, verilen soruya anlamsal olarak eşdeğer olan soruları geri getirin"
        },
    )
