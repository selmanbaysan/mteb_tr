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
    'Alibaba-NLP/gte-multilingual-base',
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-large-instruct",
    "trmteb/bert-base-turkish-cased_contrastive_loss_training",
    "trmteb/bert-base-turkish-cased_large_scale_contrastive_learning",
    "trmteb/bert-base-turkish-cased-mean-nli-stsb-tr_contrastive_loss_training",
    "trmteb/berturk_base_contrastive_loss_training",
    "trmteb/multilingual-e5-base_contrastive_loss_training",
    "trmteb/multilingual-e5-base_fine_tuned",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "dbmdz/bert-base-turkish-uncased",
]

for model in models:
    print(f"Evaluating {model}")
    evaluate_model(model)
    print("Done")
