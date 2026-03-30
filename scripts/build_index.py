import numpy as np
import faiss


EMBEDDINGS_PATH = "data/embeddings/candidate_embeddings.npy"
INDEX_PATH = "data/embeddings/faiss.index"


if __name__ == "__main__":
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    # 1. Normalize the embeddings to unit length (length = 1)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print("✅ FAISS index built and saved!")
