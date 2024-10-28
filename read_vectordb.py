import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings



def read_vectordb_as_df(db_path:str):
    result = []
    client = chromadb.PersistentClient(path=db_path)
    for collection in client.list_collections():
        data = collection.get(include=['documents', 'metadatas'])
        result.append(data)
        df = pd.DataFrame({"ids":data["ids"], 
                        #    "embeddings":data["embeddings"], 
                           "metadatas":data["metadatas"], 
                           "documents":data["documents"]})
        df["first_div"] = df["metadatas"].apply(lambda x: x["First Division"])
        df["second_div"] = df["metadatas"].apply(lambda x: x["Second Division"])
        df["filename"] = df["metadatas"].apply(lambda x: x["File Name"])
        df = df[["ids", "first_div", "second_div","filename","documents", "metadatas"]]
    return df

if __name__ == "__main__":
    


    db_path = "./chroma_rule_db"
    # db_path = "./chroma_langchain_db"
    vector_store = Chroma(collection_name="my_collection", persist_directory=db_path, embedding_function=OllamaEmbeddings(model="bge-m3:latest"))
    print(vector_store)


    df = read_vectordb_as_df(db_path=db_path)
    # print(df)
    print(df["second_div"].unique())
    print(df["filename"].unique())

    # print(df.iloc[-2:,:]["documents"].values)
    # print(df.iloc[-2:,:]["metadatas"].values)


    # results = vector_store.similarity_search(
    #     "in Win GD, explain the specification of X52DF-1.1",
    #     k=2,
    #     )
    # print(results)
