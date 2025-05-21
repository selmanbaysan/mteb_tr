from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata


class WMT16BitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="WMT16BitextMining",
        dataset={
            "path": "trmteb/wmt16_en_tr",
            "revision": "main",
            "trust_remote_code": True,
        },
        description="This dataset contains parallel sentences in English and Turkish. "
        + "It is a subset of the WMT16 dataset.",
        reference="http://www.aclweb.org/anthology/W/W16/W16-2301",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="f1",
        bibtex_citation="""
@InProceedings{bojar-EtAl:2016:WMT1,
  author    = {Bojar, Ond
{r}ej  and  Chatterjee, Rajen  and  Federmann, Christian  and  Graham, Yvette  and  Haddow, Barry  and  Huck, Matthias  and  Jimeno Yepes, Antonio  and  Koehn, Philipp  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, Matteo  and  Neveol, Aurelie  and  Neves, Mariana  and  Popel, Martin  and  Post, Matt  and  Rubino, Raphael  and  Scarton, Carolina  and  Specia, Lucia  and  Turchi, Marco  and  Verspoor, Karin  and  Zampieri, Marcos},
  title     = {Findings of the 2016 Conference on Machine Translation},
  booktitle = {Proceedings of the First Conference on Machine Translation},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics},
  pages     = {131--198},
  url       = {http://www.aclweb.org/anthology/W/W16/W16-2301}
}
""",
        prompt="Retrieve parallel sentences.",
    )