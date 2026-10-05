from functools import lru_cache
EMBEDDING_MODEL="intfloat/multilingual-e5-base"
@lru_cache(maxsize=1)
def get_embedding_model(model_name=EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    print(f"Loading multilingual embedding model: {model_name}",flush=True)
    model=SentenceTransformer(model_name)
    print("Multilingual embedding model loaded.",flush=True)
    return model
def encode_query(model,text):
    return model.encode(f"query: {str(text).strip()}",normalize_embeddings=True)
def encode_passages(model,texts):
    return model.encode([f"passage: {str(text).strip()}" for text in texts],normalize_embeddings=True,show_progress_bar=True)