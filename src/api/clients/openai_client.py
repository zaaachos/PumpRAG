import os
from typing import List, Tuple

import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
# directory imports
import os
from pathlib import Path

from utils.data import WikiDataset, create_dataloader
from src.api.clients.rag_client import VectorStore
from utils.config import Config

from torch.utils.data import DataLoader, Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

from icecream import ic

# logging imports
import logging
from tqdm import tqdm
import warnings
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


# ignore warnings
warnings.filterwarnings("ignore")
# logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logging.getLogger().setLevel(logging.CRITICAL)


# logging.info(msg="[SYSTEM] Fetching configuration parameters")
cnfg = Config()


class OpenAIClient:

    def __init__(
        self,
        embedding_model_name: str,
        max_tokens: int = 8192,
    ):
        self.chatBot = self.init_chatbot()
        assert isinstance(self.chatBot, AzureOpenAI)
        self.vector_db = self.init_vectorDB_connection()
        assert isinstance(self.vector_db, VectorStore)
        self.embedding_model = self.chatBot
        self.embedding_model_name = embedding_model_name
        try:
            self.encoder_tokenizer = tiktoken.encoding_for_model(
                "text-embedding-ada-002"
            )
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            self.encoder_tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens

    def tokenize(self, text: str) -> str:
        # firstly tokenize it
        tokenized = self.encoder_tokenizer.encode(text)

        # and then truncate the long texts before passing it to the model
        return self.encoder_tokenizer.decode(tokenized[: self.max_tokens])

    def custom_length_function(self, text: str):
        tokens = self.encoder_tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    def check_vector_fullness(self) -> bool:
        return self.vector_db.index.describe_index_stats()["total_vector_count"] > 0

    def init_chatbot(self):
        try:
            chatbot = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            ic("ChatBot initialized successfully.")
            return chatbot
        except Exception as e:
            ic("Error initializing ChatBot with error:", e)
            return -1

    def init_vectorDB_connection(self):
        try:
            custom_vector_db = VectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"))
            return custom_vector_db
        except Exception as e:
            ic("Error initializing VectorDB with error:", e)
            return -1

    def build_dataset_objects(self):
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

    def upload_full_data(
        self,
        custom_dataloader: DataLoader,
        chunk_splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        ic("Beginning the upserting of full data")
        for batch, data in enumerate(tqdm(custom_dataloader)):
            self.upload_batch_data(batch_data=data, chunk_splitter=chunk_splitter)
        ic("Uploading completed!")

    def upload_batch_data(
        self, batch_data: dict, chunk_splitter: RecursiveCharacterTextSplitter
    ) -> None:
        batch_limit = 100
        texts, metadatas, textEmbeddings, vectorIds = [], [], [], []

        for i in range(cnfg.BATCH_SIZE):
            # build current batch knowledge
            (
                metadata,
                wiki_record_texts,
                wiki_record_metadatas,
                wiki_model_embeddings,
            ) = self.__create_vector_knowledge(
                batch_index=i, batch_data=batch_data, chunk_splitter=chunk_splitter
            )
            # append these to current batches
            texts.extend(wiki_record_texts)
            metadatas.extend(wiki_record_metadatas)
            textEmbeddings.extend(wiki_model_embeddings)
            vectorIds.extend(
                [
                    f"{metadata['wiki-id']}-{idx}"
                    for idx in range(len(wiki_record_texts))
                ]
            )
            # if we have reached the batch_limit we can add texts
            if len(texts) >= batch_limit:
                self.vector_db.store_embeddings(
                    metadata=zip(vectorIds, textEmbeddings, metadatas)
                )
                texts, metadatas, textEmbeddings, vectorIds = [], [], [], []

        if len(texts) > 0:
            self.vector_db.store_embeddings(
                metadata=zip(vectorIds, textEmbeddings, metadatas)
            )

    def __create_vector_knowledge(
        self,
        batch_index: int,
        batch_data: dict,
        chunk_splitter: RecursiveCharacterTextSplitter,
    ) -> Tuple[dict, List, List, List]:
        metadata = {
            "wiki-id": str(batch_data["sample_id"][batch_index]),
            "source": batch_data["source"][batch_index],
            "title": batch_data["title"][batch_index],
        }

        # now we create chunks from the record text
        wiki_record_texts = chunk_splitter.split_text(batch_data["text"][batch_index])
        # create individual metadata dicts for each chunk
        wiki_record_metadatas = [
            {"chunk": j, "text": text, **metadata}
            for j, text in enumerate(wiki_record_texts)
        ]

        wiki_tokenized_texts = [self.tokenize(text=text) for text in wiki_record_texts]

        wiki_model_embeddings = [
            self.embedding_model.embeddings.create(
                input=tokenized_text, model=self.embedding_model_name
            )
            .data[0]
            .embedding
            for tokenized_text in wiki_tokenized_texts
        ]
        return metadata, wiki_record_texts, wiki_record_metadatas, wiki_model_embeddings

    def __retrieve(self, text_embeddings: List[float], top_docs: int = 3) -> None:
        ic("[RAG] Beging Retrieval")
        retrieved = self.vector_db.index.query(
            vector=[text_embeddings], top_k=top_docs, include_metadata=True
        )
        ic("[RAG] Retrieval Finished")

        return retrieved

    def rag_query(self, query_text: str, top_k: int = 3):
        ic("[BOT] Begin text embedding")
        query_embeddings = (
            self.chatBot.embeddings.create(
                input=query_text, model=self.embedding_model_name
            )
            .data[0]
            .embedding
        )
        ic("[BOT] text embedding: Finished")
        retrieved_texts = self.__retrieve(query_embeddings, top_docs=top_k)
        return retrieved_texts

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def respond(self, messages: List[str]) -> str:
        ic("[BOT] Begin respond")
        response = self.chatBot.chat.completions.create(
            model="gpt-4-turbo",  # model = "deployment_name".
            temperature=0.1,
            max_tokens=200,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True,
            stop=None,
            messages=messages
        )
        ic("[BOT] Respond: Finished")
        
        return response
