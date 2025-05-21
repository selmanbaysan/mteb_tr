from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TurkishIronyClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkishIronyClassification",
        description="Extended Turkish Social Media Dataset for Irony Detection, extended over Turkish Irony Dataset",
        reference="https://github.com/teghub/IronyTR",
        dataset={
            "path": "trmteb/irony-tr",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="accuracy",
        domains=["Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @article{ozturk2021ironytr,
          title={IronyTR: Irony Detection in Turkish Informal Texts},
          author={Ozturk, Asli Umay and Cemek, Yesim and Karagoz, Pinar},
          journal={International Journal of Intelligent Information Technologies (IJIIT)},
          volume={17},
          number={4},
          pages={1--18},
          year={2021},
          publisher={IGI Global}
        }
        """,
    )

    def dataset_transform(self):
        self.dataset = self.dataset["test"]
        self.dataset = self.dataset.class_encode_column("label")
        self.dataset = self.dataset.train_test_split(test_size=0.2, seed=self.seed, stratify_by_column="label")
