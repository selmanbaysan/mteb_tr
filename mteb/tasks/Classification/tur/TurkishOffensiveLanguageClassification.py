from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TurkishOffensiveLanguageClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkishOffensiveLanguageClassification",
        description="Turkish split of Twitter dataset for Offensive Speech Identification in Social Media. "
                    "This dataset was collected from Twitter, where the tweets are annotated for offensive "
                    "speech with offensive or non-offensive labels",
        reference="https://sites.google.com/site/offensevalsharedtask/offenseval-2020",
        dataset={
            "path": "trmteb/offenseval",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="accuracy",
        task_subtypes=["Sentiment/Hate speech"],
        bibtex_citation="""
        @inproceedings{zampieri-etal-2020-semeval,
        title = {{SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media (OffensEval 2020)}},
        author = {Zampieri, Marcos and Nakov, Preslav and Rosenthal, Sara and Atanasova, Pepa and Karadzhov, 
        Georgi and Mubarak, Hamdy and Derczynski, Leon and Pitenis, Zeses and Çöltekin, Çağrı},
        booktitle = {Proceedings of SemEval},
        year = {2020}
        }
        """,
    )

    """def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )"""
