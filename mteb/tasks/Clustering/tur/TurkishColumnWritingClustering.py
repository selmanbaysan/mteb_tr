from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import clustering_downsample
from mteb.abstasks.TaskMetadata import TaskMetadata


class TurkishColumnWritingClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="TurkishColumnWritingClustering",
        description="Column Writings dataset contains 630 Turkish column writings from 18 different authors,"
                    " each has 35 coloumn writings. Dataset is genereated by Kemik Natural Language Processing Group.",
        reference="https://hackmd.io/@data-tdd/HJPOie2Tc",
        dataset={
            "path": "trmteb/630koseyazisi_p2p",
            "revision": "main",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="v_measure",
        domains=["News", "Written"],
        task_subtypes=[],
        dialect=[],
        bibtex_citation="""@article{amasyali2006automatic,
          author   = {Amasyali, MF and Diri, B},
          title    = {Automatic Turkish Text Categorization in Terms of Author, Genre and Gender},
          journal  = {11th International Conference on Applications of Natural Language to Information Systems-NLDB},
          year     = {2006},
          volume   = {LNCS Volume 3999}
        }""",
    )
    
    """def dataset_transform(self):
        ds = clustering_downsample(self.dataset, self.seed, max_samples_in_cluster=len(self.dataset["test"]["sentences"][0]) // 10)
        self.dataset = ds"""
