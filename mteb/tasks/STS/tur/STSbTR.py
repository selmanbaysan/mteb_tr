from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STSbTR(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSbTR",
        dataset={
            "path": "trmteb/stsb-tr",
            "revision": "main",
        },
        description="Semantic Textual Similarity (STS) benchmark Turkish (STSb-TR) dataset is the machine translated "
                    "version of English STS benchmark dataset using Google Cloud Translation API. No human corrections"
                    " have been made to the translations.",
        reference="https://github.com/verimsu/STSb-TR",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{beken-fikri-etal-2021-semantic,
    title = "Semantic Similarity Based Evaluation for Abstractive News Summarization",
    author = "Beken Fikri, Figen  and Oflazer, Kemal and Yanikoglu, Berrin",
    booktitle = "Proceedings of the 1st Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.gem-1.3",
    doi = "10.18653/v1/2021.gem-1.3",
    pages = "24--33",
    abstract = "ROUGE is a widely used evaluation metric in text summarization. However, it is not suitable for the evaluation of abstractive summarization systems as it relies on lexical overlap between the gold standard and the generated summaries. This limitation becomes more apparent for agglutinative languages with very large vocabularies and high type/token ratios. In this paper, we present semantic similarity models for Turkish and apply them as evaluation metrics for an abstractive summarization task. To achieve this, we translated the English STSb dataset into Turkish and presented the first semantic textual similarity dataset for Turkish as well. We showed that our best similarity models have better alignment with average human judgments compared to ROUGE in both Pearson and Spearman correlations.",
}""",
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
