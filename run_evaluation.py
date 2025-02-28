import mteb
from mteb import MTEB
from sentence_transformers import SentenceTransformer

mteb_tr = mteb.get_benchmark("MTEB(Turkish)")

# Initialize model
model_name = "sentence-transformers/LaBSE" # or path to local model
model = SentenceTransformer(model_name)

# Initialize MTEB evaluation
evaluation = MTEB(tasks=mteb_tr)

# Run evaluation
results = evaluation.run(model, output_folder="results")

# Print results
print(results)
