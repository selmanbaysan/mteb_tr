from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NFCorpusTR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NFCorpusTR",
        dataset={
            "path": "selmanbaysan/nfcorpus-tr",
            "revision": "main",
        },
        description="Turkish machine translated version of the NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Academic", "Written"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{boteva2016,
  author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
  title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
  journal = {Proceedings of the 38th European Conference on Information Retrieval},
  journal-abbrev = {ECIR},
  year = {2016},
  city = {Padova},
  country = {Italy},
  url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf}
}""",
        prompt={
            "query": "Verilen soruya en iyi cevapları veren ilgili belgeleri alın"
        },
    )
