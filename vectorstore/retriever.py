import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "tata_annual_report"

def get_embedding_model(model_name=EMBEDDING_MODEL):
    return SentenceTransformer(model_name)

def get_collection(chroma_path=CHROMA_PATH,collection_name=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection(name=collection_name)

def retrieve_documents(query,top_k=5,model=None,collection=None,document_id=None):
    if not query or not query.strip():
        return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}
    if model is None:
        model = get_embedding_model()
    if collection is None:
        collection = get_collection()
    query_embedding = model.encode(query).tolist()
    query_kwargs = {
        "query_embeddings":[query_embedding],
        "n_results":top_k,
    }
    if document_id:
        query_kwargs["where"] = {"document_id":document_id}
    return collection.query(**query_kwargs)

def format_results(results):
    if not results or not results.get("documents"):
        return []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances",[[]])[0]
    ids = results.get("ids",[[]])[0]
    formatted = []
    for index,document in enumerate(documents):
        formatted.append({
            "chunk_id":ids[index] if index < len(ids) else "",
            "document":document,
            "metadata":metadatas[index] if index < len(metadatas) else {},
            "distance":distances[index] if index < len(distances) else None,
        })
    return formatted

if __name__ == "__main__":
    model = get_embedding_model()
    collection = get_collection()
    print("Embedding model loaded successfully.")
    print("Connected to ChromaDB.")
    print(f"Collection loaded. Total documents: {collection.count()}")
    query = input("\nEnter your question: ").strip()
    if not query:
        print("Empty query provided. Exiting.")
        raise SystemExit(0)
    results = retrieve_documents(
        query,
        top_k=5,
        model=model,
        collection=collection,
    )
    formatted_results = format_results(results)
    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)
    for index,result in enumerate(formatted_results,start=1):
        print(f"\nResult {index}")
        print("-" * 50)
        print(f"Chunk ID:\n{result['chunk_id']}")
        print(f"\nDocument:\n{result['document']}")
        print(f"\nMetadata:\n{result['metadata']}")
        print(f"\nDistance:\n{result['distance']}")