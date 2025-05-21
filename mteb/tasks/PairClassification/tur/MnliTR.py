from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class MnliTr(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="MnliTr",
        dataset={
            "path": "trmteb/multinli_tr",
            "revision": "main",
        },
        description="Textual Entailment Recognition for Turkish. This task requires to recognize, given two text fragments, "
                    + "whether the meaning of one text is entailed (can be inferred) from the other text.",
        reference="https://arxiv.org/pdf/2004.14963",
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test_matched", "test_mismatched"],
        eval_langs=["tur-Latn"],
        main_score="max_ap",
        date=("2000-01-01", "2018-01-01"),
        domains=["News", "Web", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{budur-etal-2020-data,
            title = "Data and Representation for Turkish Natural Language Inference",
            author = "Budur, Emrah and
              Özçelik, Rıza and
              Güngör, Tunga",
            booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
            month = nov,
            year = "2020",
            address = "Online",
            publisher = "Association for Computational Linguistics"
        }""",
        prompt="Given a premise, retrieve a hypothesis that is entailed by the premise",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
