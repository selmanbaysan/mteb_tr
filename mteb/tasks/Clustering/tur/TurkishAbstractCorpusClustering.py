from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import clustering_downsample
from mteb.abstasks.TaskMetadata import TaskMetadata


class TurkishAbstractCorpusClustering(AbsTaskClustering):
  metadata = TaskMetadata(
        name="TurkishAbstractCorpusClustering",
        description="Abstracts of 6234 papers, categorized by scientific discipline. Test dataset is a collection of "
                    "categorized and annotated abstracts from 6234 papers from various diciplines Abstract Corpus. "
                    "The main goal for this dataset is to investigate the patterns used in scientific writing,"
                    " the lexicon variety by different diciplines.",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "trmteb/ts_abstract_corpus_p2p",
            "revision": "main",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="v_measure",
        domains=["Academic", "Written"],
        task_subtypes=[],
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@INCOLLECTION{Ozturk2014-au,
          title     = "Turkish labeled text corpus",
          booktitle = "2014 22nd Signal Processing and Communications Applications
                       Conference ({SIU})",
          author    = "Özturk, S and Sankur, B and Güngör, T and Yılmaz, M B
                       and Köroǧlu, B and Aǧın, O and Ahat, M",
          publisher = "IEEE",
          pages     = "1395--1398",
          year      =  2014
        }""",
    )

  """def dataset_transform(self):
      ds = clustering_downsample(self.dataset, self.seed, max_samples_in_cluster=len(self.dataset["test"]["sentences"][0]) // 10)
      self.dataset = ds"""

