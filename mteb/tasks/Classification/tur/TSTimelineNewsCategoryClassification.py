from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TSTimelineNewsCategoryClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TSTimelineNewsCategoryClassification",
        description="TS TimeLine News Category Dataset is a collection of news harvested from online newspapers. "
                    "The dataset is composed of 551k news that covers a period of 19 years (1998-2016). "
                    "The news is segmented into sentences and labelled with 12 categories.",
        reference="https://hackmd.io/@data-tdd/BkXpTWxDK",
        dataset={
            "path": "trmteb/ts_timeline_news_category",
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
        citation="Sezer, B. ve Sezer, T. (2020). Büyük veride “kadın” sözcüğünün eşdizim örüntüsü. "
                 "E. Arslan (Ed.) İletişim Çalışmaları ve Kadın içinde (17-46). Konya: Literatürk Academia",
    )
