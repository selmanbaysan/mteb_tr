from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class TQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TQuadRetrieval",
        dataset={
            "path": "trmteb/tquad",
            "revision": "main",
        },
        description="This dataset is the Turkish Question & Answer dataset on Turkish & Islamic Science History within"
                    " the scope of Teknofest 2018 Artificial Intelligence competition.",
        reference="https://github.com/TQuad/turkish-nlp-qa-dataset",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        task_subtypes=["Question answering"],
        license="cc-by-nc-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@INPROCEEDINGS{Soygazi2021-ry,
          title           = "{THQuAD}: Turkish historic question answering dataset for
                             reading comprehension",
          booktitle       = "2021 6th International Conference on Computer Science and
                             Engineering ({UBMK})",
          author          = "Soygazi, Fatih and Ciftci, Okan and Kok, Ugurcan and
                             Cengiz, Soner",
          publisher       = "IEEE",
          month           =  sep,
          year            =  2021,
          conference      = "2021 6th International Conference on Computer Science and
                             Engineering (UBMK)",
          location        = "Ankara, Turkey"
        }
        """,
    )