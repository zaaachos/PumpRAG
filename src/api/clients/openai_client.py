import os
from typing import List, Tuple

import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
# directory imports
import os
from pathlib import Path

from utils.config import Config

from torch.utils.data import DataLoader, Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        embedding_model_name: str
    ):
        self.chatBot = self.init_chatbot()
        assert isinstance(self.chatBot, AzureOpenAI)
        self.embedding_model = self.chatBot
        self.embedding_model_name = embedding_model_name
        

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

    def generate_text_embeds(self, query_text: str):
        query_embeddings = (
            self.chatBot.embeddings.create(
                input=query_text, model=self.embedding_model_name
            )
            .data[0]
            .embedding
        )
        return query_embeddings

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
            messages=messages,
        )
        ic("[BOT] Respond: Finished")

        return response
