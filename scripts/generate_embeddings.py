import json
import numpy as np
import os
import sys

# Add the root directory (ats-llm-system) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Navigate to 'ats-llm-system' directory
sys.path.append(project_root)

from src.embeddings.embedder import generate_embeddings, build_candidate_text


INPUT_PATH = "data/processed/candidates.json"
OUTPUT_PATH = "data/embeddings/candidate_embeddings.npy"


def load_data(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    data = load_data(INPUT_PATH)

    texts = [build_candidate_text(c) for c in data]

    embeddings = generate_embeddings(texts, is_query=False)

    np.save(OUTPUT_PATH, embeddings)

    print(f"✅ Saved embeddings: {embeddings.shape}")
