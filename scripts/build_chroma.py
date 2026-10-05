from vectorstore.chroma_store import rebuild_collection
if __name__=="__main__":
    collection=rebuild_collection()
    print(f"SOURVETA multilingual ChromaDB ready: {collection.count()} records")