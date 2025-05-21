from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ArguAnaTR(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="ArguAnaTR",
        description="Turkish machine translated version of the ArguAna.",
        reference="http://argumentation.bplaced.net/arguana/data",
        dataset={
            "path": "trmteb/arguana-tr",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Written"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        annotations_creators=None,
        dialect=[],
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
        prompt={"query": "Bir iddia verildiğinde, iddiayı çürüten belgeleri bulun"},
    )
