from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SCIDOCSTR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCSTR",
        dataset={
            "path": "selmanbaysan/scidocs-tr",
            "revision": "main",
        },
        description=(
            "Turkish machine translated version of the SciDocs. SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
            + " prediction, to document classification and recommendation."
        ),
        reference="https://allenai.org/data/scidocs",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Academic", "Written", "Non-fiction"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators=None,
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{specter2020cohan,
  title={SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle={ACL},
  year={2020}
}""",
        prompt={
            "query": "Bir bilimsel makale başlığı verildiğinde, söz konusu makale tarafından atıfta bulunulan makale özetlerini alın"
        },
    )
