from mteb import MTEB
from sentence_transformers import SentenceTransformer
from mteb.tasks.BitextMining.tur.WMT16BitextMining import WMT16BitextMining

# Initialize model
model_name = "sentence-transformers/all-MiniLM-L6-v2" # or path to local model
model = SentenceTransformer(model_name)

# Define evaluation tasks
tasks = [WMT16BitextMining()]  # Add your custom tasks here

# Initialize MTEB evaluation
evaluation = MTEB(tasks=tasks)

# Run evaluation
results = evaluation.run(model, output_folder="results")

# Print results
print(results)
