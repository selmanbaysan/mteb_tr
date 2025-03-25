import mteb
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import os
import glob

mteb_tr = mteb.get_benchmark("MTEB(Turkish)")

# Initialize model
# model_name = "sentence-transformers/LaBSE" # or path to local model
def evaluate_model(model_name):
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Initialize MTEB evaluation
    evaluation = MTEB(tasks=mteb_tr)

    # Run evaluation
    results = evaluation.run(model, output_folder="results")


models = [
    'selmanbaysan/multilingual-e5-base_fine_tuned',
]

for model in models:
    print(f"Evaluating {model}")
    evaluate_model(model)
    print("Done")
