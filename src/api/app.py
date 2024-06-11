import asyncio
from httpx import AsyncClient
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from collections import deque
from dotenv import load_dotenv
import uvicorn

from clients.openai_client import OpenAIClient
from clients.rag_client import RAGVectorDatabaseClient
from utils.config import Config
import json

# Initialize FASTAPI app
app = FastAPI()


# Set the templates directory
templates_directory = os.path.join(os.path.dirname(__file__), "../frontend/templates")
if not os.path.isdir(templates_directory):
    raise RuntimeError(f"Templates directory '{templates_directory}' does not exist")

# Mount the static files directory
static_directory = os.path.join(os.path.dirname(__file__), "../frontend/static")
if not os.path.isdir(static_directory):
    raise RuntimeError(f"Static directory '{static_directory}' does not exist")

templates = Jinja2Templates(directory=templates_directory)
app.mount("/static", StaticFiles(directory=static_directory), name="static")

load_dotenv()
cnfg = Config()

# init VirtualAssistant memory
memory = deque(maxlen=cnfg.MAX_MEMORY_SIZE)

# init the clients
websocket_clients = []

# build the chat_client object
chat_client = OpenAIClient(embedding_model_name=os.getenv("EMBEDDINGS_MODEL_NAME"))
rag_client = RAGVectorDatabaseClient(index_name=os.getenv("PINECONE_INDEX_NAME"))


if not rag_client.check_vector_fullness():
    custom_dataset, custom_dataloader, chunk_splitter = (
        rag_client.build_gym_dataset_objects()
    )
    rag_client.upload_full_data(chat_client, custom_dataloader, chunk_splitter)


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    # Accept incoming websocket connection
    await websocket.accept()
    # Add the websocket to the list of clients
    websocket_clients.append(websocket)

    try:
        # initialise the role of the system
        messages = list()

        while True:
            # Receive message from the client
            websocket_data = await websocket.receive_text()
            data_json = json.loads(websocket_data)
            user_message = data_json["message"]
            is_rag_enabled = data_json["isRagEnabled"]

            if is_rag_enabled:
                messages.append(
                    {
                        "role": "system",
                        "content": """You are gym trainer that helps humans to train their muscles providing different exercises. Use the provided CONTEXT \
                            delimited by [] quotes to answer questions, with "[QUESTION]" format. If the answer cannot be found in the CONTEXT, \
                          write "Sorry, I don't know the answer to this question.". Be Specific and Descriptive. \
                            Order matters, so if for example user tells you 'summarize the following...', \
                            you have to summarize the query provided after the word 'following'.""",
                    }
                )
                # use RAG to retrieve relevant text-passages
                user_query_text_embeds = chat_client.generate_text_embeds(
                    query_text=user_message
                )
                retrieved = rag_client.retrieve(text_embeddings=user_query_text_embeds)

                top_k_matches = retrieved["matches"]
                matched_scores = [m for m in top_k_matches if m["score"] > 0.8]
                matched_contexts = [m["metadata"]["text"] for m in matched_scores]

                # create the context message to pass to LLM
                context_str = "\n\n---\n\n".join(matched_contexts) + "\n\n-----\n\n"
                # print(context_str)
                memory.append(context_str)

                # Combine RAG contexts with past interactions from memory
                combined_context = "\n\n".join(matched_contexts + list(memory))
                messages.append(
                    {"role": "user", "content": f"[CONTEXT]: {combined_context}"}
                )
                memory.append(context_str)


            # now pass the user message to model
            messages.append({"role": "user", "content": f"[QUESTION]: {user_message}"})

            # Send message to Azure OpenAI and get response
            chat_client_response = await get_openai_response(websocket, messages)
            memory.append(chat_client_response)

            # then append to history messages
            messages.append({"role": "assistant", "content": chat_client_response})

    finally:
        # Remove the websocket from the list of clients if connection is closed
        websocket_clients.remove(websocket)


async def get_openai_response(websocket, message: str) -> str:
    async with AsyncClient() as client:

        response = chat_client.respond(message)
        responses = []
        for chunk in response:
            if len(chunk.choices) > 0:
                msg = chunk.choices[0].delta.content
                msg = "" if msg is None else msg
                responses.append(msg)

                await websocket.send_text(msg)

        response = "".join(response)
        return response


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
