from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SquadTRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SquadTRRetrieval",
        dataset={
            "path": "trmteb/squad-tr",
            "revision": "main",
        },
        description="SQuAD-TR is a machine translated version of the original SQuAD2.0 dataset into Turkish, using Amazon Translate.",
        reference="https://github.com/boun-tabi/SQuAD-TR",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        task_subtypes=["Question answering"],
        license="cc-by-nc-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{budur-etal-2024-squad-tr,
          title={Building Efficient and Effective OpenQA Systems for Low-Resource Languages}, 
          author={Emrah Budur and Rıza Özçelik and Dilara Soylu and Omar Khattab and Tunga Güngör and Christopher Potts},
          year={2024},
          eprint={TBD},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
        }
        """,
    )