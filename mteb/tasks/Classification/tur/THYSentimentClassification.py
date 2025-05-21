from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class THYSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="THYSentimentClassification",
        description="This data set, provided by THY, is composed of tweets gathered form THY social media accounts. "
                    "The data set is composed of 23k (23300) manually labelled tweets. Each tweet is annotated with "
                    "polarity and tags. The tags are based on the information presented in the tweet.",
        reference="https://hackmd.io/@data-tdd/B1SyZQbyq",
        dataset={
            "path": "trmteb/thy_sa",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="accuracy",
        date=("2013-01-01", "2013-08-11"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
    )

    def dataset_transform(self):
        self.dataset = self.dataset["test"]
        self.dataset = self.dataset.class_encode_column("label")
        self.dataset = self.dataset.train_test_split(test_size=0.2, seed=self.seed, stratify_by_column="label")

