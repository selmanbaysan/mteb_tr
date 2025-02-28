from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SciFactTR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFactTR",
        dataset={
            "path": "selmanbaysan/scifact-tr",
            "revision": "main",
        },
        description="Turkish machine translated version of the SciFact. SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://github.com/allenai/scifact",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Academic", "Medical", "Written"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{specter2020cohan,
  title={SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle={ACL},
  year={2020}
}""",
        prompt={
            "query": "Bilimsel bir iddia verildiğinde, iddiayı destekleyen veya çürüten belgeleri getir"
        },
    )
