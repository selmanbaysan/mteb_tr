from __future__ import annotations

import argparse
import logging
import math
import queue
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm.autonotebook import trange
from transformers import AutoModel, AutoTokenizer

from mteb import MTEB

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

MTEB_TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)


CMTEB_TASK_LIST = [
    "TNews",
    "IFlyTek",
    "MultilingualSentiment",
    "JDReview",
    "OnlineShopping",
    "Waimai",
    "AmazonReviewsClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MultilingualSentiment",
    "CLSClusteringS2S",
    "CLSClusteringP2P",
    "ThuNewsClusteringS2S",
    "ThuNewsClusteringP2P",
    "Ocnli",
    "Cmnli",
    "T2Reranking",
    "MmarcoReranking",
    "CMedQAv1",
    "CMedQAv2",
    "T2Retrieval",
    "MMarcoRetrieval",
    "DuRetrieval",
    "CovidRetrieval",
    "CmedqaRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "VideoRetrieval",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "STSB",
    "AFQMC",
    "QBQTC",
    "STS22",
]

MTEB_PL = [
    "CBD",
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
    "AllegroReviews",
    "PAC",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "SICK-E-PL",
    "PPC",
    "CDSC-E",
    "PSC",
    "8TagsClustering",
    "SICK-R-PL",
    "CDSC-R",
    "STS22",
    "ArguAna-PL",
    "DBPedia-PL",
    "FiQA-PL",
    "HotpotQA-PL",
    "MSMARCO-PL",
    "NFCorpus-PL",
    "NQ-PL",
    "Quora-PL",
    "SCIDOCS-PL",
    "SciFact-PL",
    "TRECCOVID-PL",
]

MTEB_FR = [
    "AmazonReviewsClassification",
    "MasakhaNEWSClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "OpusparcusPC",
    "PawsX",
    "AlloProfClusteringP2P",
    "AlloProfClusteringS2S",
    "HALClusteringS2S",
    "MasakhaNEWSClusteringP2P",
    "MasakhaNEWSClusteringS2S",
    "MLSUMClusteringP2P",
    "MLSUMClusteringS2S",
    "SyntecReranking",
    "AlloprofReranking",
    "AlloprofRetrieval",
    "BSARDRetrieval",
    "SyntecRetrieval",
    "XPQARetrieval",
    "MintakaRetrieval",
    "SummEvalFr",
    "STSBenchmarkMultilingualSTS",
    "STS22",
    "SICKFr",
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s : %(message)s"
)

logger = logging.getLogger("eval_mteb_qwen.py")


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ""

    return "Instruct: {}\nQuery: ".format(task_description)


def get_task_def_by_task_name_and_type(
    task_name: str,
    task_type: str,
    default_instruct="Given a web search query, retrieve relevant passages that answer the query",
) -> str:
    if task_type in ["STS"]:
        return "Retrieve semantically similar text"

    if task_type in ["Summarization"]:
        return "Given a news summary, retrieve other semantically similar summaries"

    if task_type in ["BitextMining"]:
        return "Retrieve parallel sentences"

    if task_type in ["Classification"]:
        task_name_to_instruct: Dict[str, str] = {
            "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
            "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment",
            "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category",
            "Banking77Classification": "Given a online banking query, find the corresponding intents",
            "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
            "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
            "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
            "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
            "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation",
            "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation",
            "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic",
            "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
            # C-MTEB eval instructions
            "TNews": "Classify the fine-grained category of the given news title",
            "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
            "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative",
            "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative",
            "OnlineShopping": "Classify the customer review for online shopping into positive or negative",
            "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
            # MTEB-pl eval instructions
            "CBD": "Classify the sentiment of polish tweet reviews",
            "PolEmo2.0-IN": "Classify the sentiment of in-domain (medicine and hotels) online reviews",
            "PolEmo2.0-OUT": "Classify the sentiment of out-of-domain (products and school) online reviews",
            "AllegroReviews": "Classify the sentiment of reviews from e-commerce marketplace Allegro",
            "PAC": 'Classify the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA"',
        }
        return task_name_to_instruct[task_name]

    if task_type in ["Clustering"]:
        task_name_to_instruct: Dict[str, str] = {
            "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
            "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
            "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts",
            "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles",
            "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts",
            "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles",
            "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles",
            "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
            "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles",
            "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
            "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles",
            # C-MTEB eval instructions
            "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles",
            "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
            "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
            "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
            # MTEB-fr eval instructions
            "AlloProfClusteringP2P": "Identify the main category of Allo Prof document based on the titles and descriptions",
            "AlloProfClusteringS2S": "Identify the main category of Allo Prof document based on the titles",
            "HALClusteringS2S": "Identify the main category of academic passage based on the titles and contents",
            "MasakhaNEWSClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
            "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
            "MLSUMClusteringP2P": "Identify the topic or theme of the given articles based on the titles and contents",
            "MLSUMClusteringS2S": "Identify the topic or theme of the given articles based on the titles",
            # MTEB-pl eval instructions
            "8TagsClustering": "Identify of headlines from social media posts in Polish  into 8 categories: film, history, food, medicine, motorization, work, sport and technology",
        }
        return task_name_to_instruct[task_name]

    if task_type in ["Reranking", "PairClassification"]:
        task_name_to_instruct: Dict[str, str] = {
            "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum",
            "MindSmallReranking": "Retrieve relevant news articles based on user browsing history",
            "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
            "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum",
            "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
            "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
            "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
            # C-MTEB eval instructions
            "T2Reranking": "Given a Chinese search query, retrieve web passages that answer the question",
            "MmarcoReranking": "Given a Chinese search query, retrieve web passages that answer the question",
            "CMedQAv1": "Given a Chinese community medical question, retrieve replies that best answer the question",
            "CMedQAv2": "Given a Chinese community medical question, retrieve replies that best answer the question",
            "Ocnli": "Retrieve semantically similar text.",
            "Cmnli": "Retrieve semantically similar text.",
            # MTEB-fr eval instructions
            "AlloprofReranking": "Given a question, retrieve passages that answer the question",
            "OpusparcusPC": "Retrieve semantically similar text",
            "PawsX": "Retrieve semantically similar text",
            "SyntecReranking": "Given a question, retrieve passages that answer the question",
            # MTEB-pl eval instructions
            "SICK-E-PL": "Retrieve semantically similar text",
            "PPC": "Retrieve semantically similar text",
            "CDSC-E": "Retrieve semantically similar text",
            "PSC": "Retrieve semantically similar text",
        }
        return task_name_to_instruct[task_name]

    if task_type in ["Retrieval"]:
        if task_name.lower().startswith("cqadupstack"):
            return "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question"

        task_name_to_instruct: Dict[str, str] = {
            "ArguAna": "Given a claim, find documents that refute the claim",
            "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
            "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
            "FEVER": "Given a claim, retrieve documents that support or refute the claim",
            "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
            "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
            "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
            "NFCorpus": "Given a question, retrieve relevant documents that best answer the question",
            "NQ": "Given a question, retrieve Wikipedia passages that answer the question",
            "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
            "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
            "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
            "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
            "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
            # C-MTEB eval instructions
            "T2Retrieval": "Given a Chinese search query, retrieve web passages that answer the question",
            "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
            "DuRetrieval": "Given a Chinese search query, retrieve web passages that answer the question",
            "CovidRetrieval": "Given a question on COVID-19, retrieve news articles that answer the question",
            "CmedqaRetrieval": "Given a Chinese community medical question, retrieve replies that best answer the question",
            "EcomRetrieval": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
            "MedicalRetrieval": "Given a medical question, retrieve user replies that best answer the question",
            "VideoRetrieval": "Given a video search query, retrieve the titles of relevant videos",
            # MTEB-fr eval instructions
            "AlloprofRetrieval": "Given a question, retrieve passages that answer the question",
            "BSARDRetrieval": "Given a question, retrieve passages that answer the question",
            "SyntecRetrieval": "Given a question, retrieve passages that answer the question",
            "XPQARetrieval": "Given a question, retrieve passages that answer the question",
            "MintakaRetrieval": "Given a question, retrieve passages that answer the question",
            # MTEB-pl eval instructions
            "ArguAna-PL": "Given a claim, find documents that refute the claim",
            "DBPedia-PL": "Given a query, retrieve relevant entity descriptions from DBPedia",
            "FiQA-PL": "Given a financial question, retrieve user replies that best answer the question",
            "HotpotQA-PL": "Given a multi-hop question, retrieve documents that can help answer the question",
            "MSMARCO-PL": "Given a web search query, retrieve relevant passages that answer the query",
            "NFCorpus-PL": "Given a question, retrieve relevant documents that best answer the question",
            "NQ-PL": "Given a question, retrieve Wikipedia passages that answer the question",
            "Quora-PL": "Given a question, retrieve questions that are semantically equivalent to the given question",
            "SCIDOCS-PL": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
            "SciFact-PL": "Given a scientific claim, retrieve documents that support or refute the claim",
            "TRECCOVID-PL": "Given a query on COVID-19, retrieve documents that answer the query",
        }

        # add lower case keys to match some beir names
        task_name_to_instruct.update({k.lower(): v for k, v in task_name_to_instruct.items()})
        # other cases where lower case match still doesn't work
        task_name_to_instruct["trec-covid"] = task_name_to_instruct["TRECCOVID"]
        task_name_to_instruct["climate-fever"] = task_name_to_instruct["ClimateFEVER"]
        task_name_to_instruct["dbpedia-entity"] = task_name_to_instruct["DBPedia"]
        task_name_to_instruct["webis-touche2020"] = task_name_to_instruct["Touche2020"]
        task_name_to_instruct["fiqa"] = task_name_to_instruct["FiQA2018"]
        task_name_to_instruct["quora"] = task_name_to_instruct["QuoraRetrieval"]

        # for miracl evaluation
        task_name_to_instruct["miracl"] = (
            "Given a question, retrieve Wikipedia passages that answer the question"
        )

        return task_name_to_instruct[task_name]
    logging.warning(
        f"No instruction config for task {task_name} with type {task_type}, use default instruction."
    )
    return default_instruct


class Encoder(torch.nn.Module):
    def __init__(self, name_or_path: str, pooling: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True)
        self.model = self.model.half()
        self.model.eval()
        self.pooling = pooling

    def forward(self, **features) -> torch.Tensor:
        output = self.model(**features, output_hidden_states=True, return_dict=True)
        hidden_state = output.hidden_states[-1]
        embeddings = self.pooler(hidden_state, **features)
        return embeddings

    def pooler(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if attention_mask.ndim == 2:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size())
        elif attention_mask.ndim == 3:
            mask_expanded = attention_mask
        else:
            raise RuntimeError(f"Unexpected {attention_mask.ndim=}")

        hidden_state = hidden_state * mask_expanded

        if self.pooling == "first":
            pooled_output = hidden_state[:, 0]

        elif self.pooling == "last":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_state.shape[0]
            return hidden_state[
                torch.arange(batch_size, device=hidden_state.device), sequence_lengths
            ]
        elif self.pooling == "mean":
            # TODO: weight
            lengths = mask_expanded.sum(1).clamp(min=1e-9)
            pooled_output = hidden_state.sum(dim=1) / lengths

        elif self.pooling == "weightedmean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            # hidden_state shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=hidden_state.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(hidden_state.size())
                .float()
                .to(hidden_state.device)
            )
            assert weights.shape == hidden_state.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        else:
            raise ValueError(f"Wrong pooler mode : {self.pooling}")
        return pooled_output


class Wrapper:
    def __init__(
        self,
        tokenizer,
        encoder: Encoder,
        batch_size: int,
        max_seq_len: int = 512,
        normalize_embeddings: bool = False,
        default_query: bool = False,
        force_default: bool = False,
        sep: str = " ",
        mp_tensor_to_cuda: bool = False,
        instruction: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.model = encoder
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pool: Optional[dict] = None
        self.normalize_embeddings = normalize_embeddings
        self.mp_tensor_to_cuda = mp_tensor_to_cuda
        self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eod_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.instruction = instruction
        self.default_query = default_query
        self.sep = sep
        self.force_default = force_default
        if self.tokenizer.padding_side != "right":
            logger.warning(
                f"Change tokenizer.padding_side from {self.tokenizer.padding_side} to right"
            )
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            logger.warning(f"Set tokenizer.pad_token as eos_token {self.tokenizer.eos_token}")
            self.tokenizer.pad_token = "<|endoftext|>"

    def start(self, target_devices: Optional[List[str]] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process

        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ["cpu"] * 4

        logger.info(
            "Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices)))
        )
        print("multi instruction", self.instruction)
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(
                target=self._encode_multi_process_worker,
                args=(cuda_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        self.pool = {"input": input_queue, "output": output_queue, "processes": processes}

    def stop(self):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in self.pool["processes"]:
            p.terminate()

        for p in self.pool["processes"]:
            p.join()
            p.close()

        self.pool["input"].close()
        self.pool["output"].close()

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                id, sentences, kwargs = input_queue.get()
                kwargs.update(device=target_device, show_progress_bar=False, convert_to_numpy=True)
                embeddings = model._encode(sentences, **kwargs)
                results_queue.put([id, embeddings])
            except queue.Empty:
                break

    def encode_multi_process(self, sentences: List[str], **kwargs):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :param kwargs: other keyword arguments for model.encode() such as batch_size
        :return: Numpy matrix with all embeddings
        """
        part_size = math.ceil(len(sentences) / len(self.pool["processes"]))
        chunk_size = part_size if part_size < 3200 else 3200  # for retrieval chunk 50000

        logger.debug(
            f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}"
        )

        input_queue = self.pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, chunk, kwargs])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk, kwargs])
            last_chunk_id += 1

        output_queue = self.pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0]
        )
        embeddings = np.concatenate([result[1] for result in results_list])
        return embeddings

    @staticmethod
    def batch_to_device(batch, target_device):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(target_device)
        return batch

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def _tokenize(self, sentences: List[str], is_query: bool):
        batch_dict = self.tokenizer(
            sentences,
            max_length=self.max_seq_len - 1,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]
        ]
        batch_dict = self.tokenizer.pad(
            batch_dict, padding=True, return_attention_mask=True, return_tensors="pt"
        )
        batch_dict["is_causal"] = False
        return batch_dict

    def _encode(
        self,
        sentences: List[str],
        is_query: bool,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
        show_progress_bar: bool = True,
        **kwargs,
    ):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.model.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), self.batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + self.batch_size]
            features = self._tokenize(sentences_batch, is_query)
            features = self.batch_to_device(features, device)

            with torch.no_grad():
                embeddings = self.model(**features)

                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            # all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
            all_embeddings = np.asarray([emb.to(torch.float).numpy() for emb in all_embeddings])
        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def encode(
        self,
        sentences: List[str],
        is_query: Optional[bool] = None,
        convert_to_tensor: bool = False,
        **kwargs,
    ):
        is_query = self.default_query if is_query is None else is_query
        if is_query and self.instruction:
            sentences = [self.instruction + sent for sent in sentences]
        kwargs.update(is_query=is_query)
        if self.pool is not None:
            kwargs.update(show_progress_bar=False)
            embeddings = self.encode_multi_process(sentences, **kwargs)
            if convert_to_tensor:
                embeddings = torch.from_numpy(embeddings)
                if self.mp_tensor_to_cuda and torch.cuda.is_available():
                    embeddings = embeddings.to(torch.device("cuda"))  # default 0-th gpu
            return embeddings

        return self._encode(sentences, convert_to_tensor=convert_to_tensor, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs):
        is_query = self.default_query if self.force_default else True
        return self.encode(queries, is_query=is_query, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        # borrowed from mteb.abstasks.AbsTaskRetrieval.DRESModel
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        elif isinstance(corpus[0], dict):
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]
        else:
            sentences = corpus
        is_query = self.default_query if self.force_default else False
        return self.encode(sentences, is_query=is_query, **kwargs)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoder = Encoder(args.model, args.pooling)
    default_query = args.default_type == "query"
    model = Wrapper(
        tokenizer,
        encoder,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        normalize_embeddings=args.norm,
        default_query=default_query,
    )
    sym_retrievals = ["QuoraRetrieval", "ArguAna", "CQADupstack"]
    if args.task == "mteb":
        task_names = MTEB_TASK_LIST
        lang = ["en"]
    elif args.task == "cmteb":
        task_names = CMTEB_TASK_LIST
        lang = ["zh", "zh-CN"]
    elif args.task == "mteb-fr":
        task_names = MTEB_FR
        lang = ["fr"]
    elif args.task == "mteb-pl":
        task_names = MTEB_PL
        lang = ["pl"]
    else:
        task_names = [args.task]
        lang = ["en", "zh", "zh-CN", "pl", "fr"]
    for task in task_names:
        evaluation = MTEB(tasks=[task], task_langs=lang)
        task_cls = evaluation.tasks[0]
        task_name: str = task_cls.metadata_dict["name"]
        task_type: str = task_cls.metadata_dict["type"]
        instruction = get_task_def_by_task_name_and_type(task_name, task_type)
        model.instruction = get_detailed_instruct(instruction)
        if task == "MSMARCO":
            eval_splits = ["dev"]
        elif task in CMTEB_TASK_LIST:
            eval_splits = task_cls.metadata_dict["eval_splits"]
        else:
            eval_splits = ["test"]
        sym = False
        for name in sym_retrievals:
            if task.startswith(name):
                sym = True
                break
            else:
                sym = False
        if sym:
            logger.info(
                f"Switch to symmetric mode for {task}, all as {'query' if default_query else 'doc'}."
            )
            model.force_default = True
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits)

        if sym:
            logger.info(f"Switch back.")
            model.force_default = force_default_ori
        print("\n")


if __name__ == "__main__":
    _PARSER = argparse.ArgumentParser()
    _PARSER.add_argument("-m", "--model", type=str, default=None)
    _PARSER.add_argument("--pooling", type=str, default="last")
    _PARSER.add_argument("--output_dir", type=str, default=None)
    _PARSER.add_argument("--default_type", type=str, default="query")
    _PARSER.add_argument("--max_seq_len", type=int, default=512)
    _PARSER.add_argument("-b", "--batch_size", type=int, default=32)
    _PARSER.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,  # None for running default tasks
    )
    _PARSER.add_argument("--norm", action="store_true")
    _ARGS = _PARSER.parse_args()
    main(_ARGS)
