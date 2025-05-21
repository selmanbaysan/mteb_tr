from mteb import MTEB
import mteb
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec 

import numpy as np
from typing import List, Dict
from mteb.encoder_interface import PromptType


class GloveEvaluator:
    def __init__(self, glove_path: str):
        self.model = self.load_glove_model(glove_path)
        
    def load_glove_model(self, glove_path: str):
        model = KeyedVectors.load_word2vec_format(glove_path, encoding='utf8')
        return model
        
    def encode(
        self,
        sentences: List[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs
    ) -> np.ndarray:
        """Encodes the given sentences using GloVe.
        
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
            words = [self.model[word] for word in sentence.split() if word in self.model]
            if words:
                vec = np.mean(words, axis=0)
            else:
                vec = np.zeros(self.model.vector_size)

            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float64)

def main():
    
    word2vec_path = "models/word2vec.txt" # Turkish GloVe model
    model = GloveEvaluator(word2vec_path)
    
    # Initialize MTEB with specific tasks
    mteb_tr = mteb.get_benchmark("MTEB(Turkish)")
    
    evaluation = MTEB(tasks=mteb_tr)
    # Run evaluation
    results = evaluation.run(model, output_folder="results/glove")

main()
