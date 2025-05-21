from mteb import MTEB
import mteb
from fasttext import load_model
import numpy as np
from typing import List, Dict
from mteb.encoder_interface import PromptType


class FastTextEvaluator:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        
    def encode(
        self,
        sentences: List[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs
    ) -> np.ndarray:
        """Encodes the given sentences using FastText.
        
        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.
            
        Returns:
            The encoded sentences.
        """
        embeddings = []
        for sentence in sentences:
            sentence = sentence.replace("\n", " ")
            vec = self.model.get_sentence_vector(sentence)
            embeddings.append(vec)
        return np.array(embeddings)

def main():
    # Initialize FastText model
    model_path = "/models/cc.tr.300.bin"  # Turkish FastText model
    model = FastTextEvaluator(model_path)
    
    # Initialize MTEB with specific tasks
    mteb_tr = mteb.get_benchmark("MTEB(Turkish)")
    
    evaluation = MTEB(tasks=mteb_tr)
    # Run evaluation
    results = evaluation.run(model, output_folder="results/fasttext")

main()
