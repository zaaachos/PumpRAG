# os adn environment imports
import os
from dotenv import load_dotenv

# pretty prints
from icecream import ic

# VectorDB imports
import time
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec

# firstly load config keys
load_dotenv()


class RAGVectorDatabaseClient:

    def __init__(self, index_name: str) -> None:
        self.index_name = index_name
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name
        self.index = self.__build_index()

    def __build_index(self) -> Pinecone.Index:
        if self.index_name in self.pc.list_indexes().names():
            ic("[INFO] Index already exists!")
            return self.pc.Index(self.index_name)
        
        ic("[INFO] We did not find index. Thus, we are now creating it:")
        # we create a new index
        self.pc.create_index(
            self.index_name,
            dimension=1536,  # dimensionality of text-embedding-ada-002
            metric="cosine",            # setting the retrieval algorithm
            spec=PodSpec(environment="gcp-starter"),        # gcp-starter Pod environment for the Free Tier
        )
        time.sleep(1)
        return self.pc.Index(self.index_name)

    def display_index(self ) -> None:
        ic(self.index.describe_index_stats())

    def store_embeddings(self, metadata: zip) -> None:
        try:
            self.index.upsert(vectors=metadata)
        except:
            ic("[ERROR] Upserting current vector failed.")

