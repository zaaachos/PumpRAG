# os adn environment imports
import os
from typing import List, Tuple
from utils.config import Config
from dotenv import load_dotenv

# pretty prints
from icecream import ic

# VectorDB imports
import time
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec
from utils.data import WikiDataset, create_dataloader, GymDataset


from torch.utils.data import DataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from clients.openai_client import OpenAIClient
from tqdm import tqdm
import tiktoken


# firstly load config keys
load_dotenv()
cnfg = Config()


class RAGVectorDatabaseClient:

    def __init__(self, index_name: str, max_tokens: int = 8192) -> None:
        self.index_name = index_name
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name
        self.index = self.__build_index()
        try:
            self.encoder_tokenizer = tiktoken.encoding_for_model(
                "text-embedding-ada-002"
            )
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            self.encoder_tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens

    def __build_index(self) -> Pinecone.Index:
        if self.index_name in self.pc.list_indexes().names():
            ic("[INFO] Index already exists!")
            return self.pc.Index(self.index_name)

        ic("[INFO] We did not find index. Thus, we are now creating it:")
        # we create a new index
        self.pc.create_index(
            self.index_name,
            dimension=1536,  # dimensionality of text-embedding-ada-002
            metric="cosine",  # setting the retrieval algorithm
            spec=PodSpec(
                environment="gcp-starter"
            ),  # gcp-starter Pod environment for the Free Tier
        )
        time.sleep(1)
        return self.pc.Index(self.index_name)

    def display_index(self) -> None:
        ic(self.index.describe_index_stats())

    def store_embeddings(self, metadata: zip) -> None:
        try:
            self.index.upsert(vectors=metadata)
        except:
            ic("[ERROR] Upserting current vector failed.")

    def check_vector_fullness(self) -> bool:
        return self.index.describe_index_stats()["total_vector_count"] > 0

    def retrieve(self, text_embeddings: List[float], top_docs: int = 3) -> None:
        ic("[RAG] Beging Retrieval")
        retrieved = self.index.query(
            vector=[text_embeddings],
            top_k=top_docs,
            include_metadata=True,
            include_values=True,
        )
        ic("[RAG] Retrieval Finished")

        return retrieved

    def upload_full_data(
        self,
        openai_client: OpenAIClient,
        custom_dataloader: DataLoader,
        chunk_splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        ic("Beginning the upserting of full data")
        for batch, data in enumerate(tqdm(custom_dataloader)):
            # print(data)
            self.upload_batch_data(
                batch_data=data,
                openai_client=openai_client,
                chunk_splitter=chunk_splitter,
            )
        ic("Uploading completed!")

    def upload_batch_data(
        self,
        batch_data: dict,
        openai_client: OpenAIClient,
        chunk_splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        batch_limit = 100
        texts, metadatas, textEmbeddings, vectorIds = [], [], [], []

        for i in range(cnfg.BATCH_SIZE):
            # build current batch knowledge
            (
                metadata,
                batch_record_texts,
                batch_record_metadatas,
                batch_model_embeddings,
            ) = self.__create_vector_knowledge(
                openai_client=openai_client,
                batch_index=i,
                batch_data=batch_data,
                chunk_splitter=chunk_splitter,
            )
            # append these to current batches
            texts.extend(batch_record_texts)
            metadatas.extend(batch_record_metadatas)
            textEmbeddings.extend(batch_model_embeddings)
            vectorIds.extend(
                [f"{metadata['id']}-{idx}" for idx in range(len(batch_record_texts))]
            )
            # if we have reached the batch_limit we can add texts
            if len(texts) >= batch_limit:
                self.store_embeddings(
                    metadata=zip(vectorIds, textEmbeddings, metadatas)
                )
                texts, metadatas, textEmbeddings, vectorIds = [], [], [], []

        if len(texts) > 0:
            self.store_embeddings(metadata=zip(vectorIds, textEmbeddings, metadatas))

    def __create_vector_knowledge(
        self,
        openai_client: OpenAIClient,
        batch_index: int,
        batch_data: dict,
        chunk_splitter: RecursiveCharacterTextSplitter,
        gym_data: bool = True,
    ) -> Tuple[dict, List, List, List]:

        if gym_data:
            metadata = {
                "id": str(batch_data["sample_id"][batch_index]),
                "source": batch_data["source"][batch_index],
                "title": batch_data["title"][batch_index],
                "type": batch_data["type"][batch_index],
                "body": batch_data["body"][batch_index],
                "equipment": batch_data["equipment"][batch_index],
                "level": batch_data["level"][batch_index],
            }
        else:
            metadata = {
                "id": str(batch_data["sample_id"][batch_index]),
                "source": batch_data["source"][batch_index],
                "title": batch_data["title"][batch_index],
            }

        # now we create chunks from the record text
        batch_record_texts = chunk_splitter.split_text(batch_data["text"][batch_index])
        # create individual metadata dicts for each chunk
        batch_record_metadatas = [
            {"chunk": j, "text": text, **metadata}
            for j, text in enumerate(batch_record_texts)
        ]

        batch_tokenized_texts = [
            self.tokenize(text=text) for text in batch_record_texts
        ]

        batch_model_embeddings = [
            openai_client.embedding_model.embeddings.create(
                input=tokenized_text, model=openai_client.embedding_model_name
            )
            .data[0]
            .embedding
            for tokenized_text in batch_tokenized_texts
        ]
        return (
            metadata,
            batch_record_texts,
            batch_record_metadatas,
            batch_model_embeddings,
        )

    def tokenize(self, text: str) -> str:
        # firstly tokenize it
        tokenized = self.encoder_tokenizer.encode(text)

        # and then truncate the long texts before passing it to the model
        return self.encoder_tokenizer.decode(tokenized[: self.max_tokens])

    def custom_length_function(self, text: str):
        tokens = self.encoder_tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    def build_wiki_dataset_objects(self):
        custom_dataset = WikiDataset(
            dataset_name=cnfg.TEXT_DATASET_NAME,
            dataset_version=cnfg.TEXT_DATASET_VERSION,
            num_samples=cnfg.NUM_SAMPLES,
        )

        custom_dataloader = create_dataloader(
            dataset=custom_dataset,
            batch_size=cnfg.BATCH_SIZE,
            num_workers=cnfg.NUM_WORKERS,
        )
        chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=self.custom_length_function,
            separators=["\n\n", "\n", " ", ""],
        )
        ic("Sample wikipedia dataset objects initialized!")
        return custom_dataset, custom_dataloader, chunk_splitter

    def build_gym_dataset_objects(self):
        custom_dataset = GymDataset(dataset_path=cnfg.GYM_DATASET_PATH)

        custom_dataloader = create_dataloader(
            dataset=custom_dataset,
            batch_size=cnfg.BATCH_SIZE,
            num_workers=cnfg.NUM_WORKERS,
        )
        chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            length_function=self.custom_length_function,
            separators=["\n\n", "\n", " ", "----"],
        )

        ic("Sample gym dataset objects initialized!")
        return custom_dataset, custom_dataloader, chunk_splitter
