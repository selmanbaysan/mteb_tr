#!/usr/bin/env python3

import argparse
import mteb
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import os
import sys

def evaluate_model(model_name, output_folder):
    """
    Evaluate a model using MTEB-TR benchmark
    
    Args:
        model_name (str): Name or path of the model to evaluate
        output_folder (str): Path to save the evaluation results
    """
    try:
        # Get MTEB-TR benchmark
        mteb_tr = mteb.get_benchmark("MTEB(Turkish)")
        
        # Initialize model
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        
        # Initialize MTEB evaluation
        evaluation = MTEB(tasks=mteb_tr)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Run evaluation
        print(f"Starting evaluation. Results will be saved to: {output_folder}")
        results = evaluation.run(model, output_folder=output_folder)
        
        print("Evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run MTEB-TR benchmark evaluation for a given model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "model_name",
        help="Name or path of the model to evaluate (e.g., 'sentence-transformers/LaBSE' or path to local model)"
    )
    
    parser.add_argument(
        "--output-folder",
        "-o",
        default="results",
        help="Path to save the evaluation results"
    )
    
    args = parser.parse_args()
    
    success = evaluate_model(args.model_name, args.output_folder)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 