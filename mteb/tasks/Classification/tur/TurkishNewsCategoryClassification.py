from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TurkishNewsCategoryClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkishNewsCategoryClassification",
        description="1150 news articles are labeled with one of the following five categories:"
                    " health, sports, economy, politics, magazines. This dataset is annotated and collected"
                    " from Turkish news resources",
        reference="http://www.kemik.yildiz.edu.tr/veri_kumelerimiz.html",
        dataset={
            "path": "trmteb/news-cat",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="accuracy",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        bibtex_citation="""
        @article{amasyali2004otomatik,
          title={Otomatik haber metinleri s{\i}n{\i}fland{\i}rma},
          author={Amasyal{\i}, MF and Y{\i}ld{\i}r{\i}m, T},
          journal={SIU 2004},
          pages={224--226},
          year={2004}
        }
        """,
    )
