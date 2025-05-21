from mteb import MTEB
import mteb
from openai import OpenAI
import numpy as np
from typing import List
from mteb.encoder_interface import PromptType

client = OpenAI()

class OpenAIEvaluator:
    def __init__(self, model_path: str):
        self.model = model_path
        
    def encode(
        self,
        sentences: List[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs
    ) -> np.ndarray:
        """Encodes the given sentences using OpenAI API.
        
        Args:
            sentences: The sentences to encode.
        Returns:
            The encoded sentences.
        """
        embeddings = []
        for sentence in sentences:
            sentence = sentence.replace("\n", " ")
            vec = client.embeddings.create(input = [sentence], model=self.model).data[0].embedding
            embeddings.append(vec)
        
        return np.array(embeddings)

def main():
    
    # Initialize MTEB with specific tasks
    mteb_tr = mteb.get_benchmark("MTEB(Turkish)")
    evaluation = MTEB(tasks=mteb_tr)

    # Initialize OpenAI model
    model_path = "text-embedding-3-small"
    model = OpenAIEvaluator(model_path)
    # Run evaluation
    results = evaluation.run(model, output_folder=f"results/{model_path}")

main()
