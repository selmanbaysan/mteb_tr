from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SquadTRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SquadTRRetrieval",
        dataset={
            "path": "boun-tabi/squad_tr",
            "revision": "main",
        },
        description="SQuAD-TR is a machine translated version of the original SQuAD2.0 dataset into Turkish, using Amazon Translate.",
        reference="https://github.com/boun-tabi/SQuAD-TR",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["validation"],
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

    def load_data(self, **kwargs) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = dict[doc_id, dict[str, str]] #id => dict with document datas like title and text
        self.queries = dict[query_id, str] #id => query
        self.relevant_docs = dict[query_id, dict[[doc_id, score]]
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])

        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.metadata_dict["eval_splits"]:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
            ds = ds.shuffle(seed=42)
            max_samples = min(1024, len(ds))
            ds = ds.select(
                range(max_samples)
            )  # limit the dataset size to make sure the task does not take too long to run
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            question = ds["question"]
            context = ds["context"]
            answer = [a["text"] for a in ds["answers"]]

            n = 0
            for q, cont, ans in zip(question, context, answer):
                self.queries[split][str(n)] = q
                q_n = n
                n += 1
                if cont not in text2id:
                    text2id[cont] = n
                    self.corpus[split][str(n)] = {"title": "", "text": cont}
                    n += 1
                if ans not in text2id:
                    text2id[ans] = n
                    self.corpus[split][str(n)] = {"title": "", "text": ans}
                    n += 1

                self.relevant_docs[split][str(q_n)] = {
                    str(text2id[ans]): 1,
                    str(text2id[cont]): 1,
                }  # only two correct matches
            self.data_loaded = True
