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

# from clients.openai_client import OpenAIClient
# from clients.rag_client import RAGVectorDatabaseClient

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


# from src.api.clients.openai_client import ChatBot
# from utils.config import Config


load_dotenv()

# cnfg = Config()


# init VirtualAssistant memory
# memory = deque(maxlen=cnfg.MAX_MEMORY_SIZE)

# init the clients
websocket_clients = []

# # build the chatBot object
# chatbot = ChatBot(embedding_model_name=os.getenv("EMBEDDINGS_MODEL_NAME"))
# if not chatbot.check_vector_fullness():
#     custom_dataset, custom_dataloader, chunk_splitter = chatbot.build_dataset_objects()
#     chatbot.upload_full_data(custom_dataloader, chunk_splitter)


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.websocket("/chat")
# async def chat_endpoint(websocket: WebSocket):
#     # Accept incoming websocket connection
#     await websocket.accept()
#     # Add the websocket to the list of clients
#     websocket_clients.append(websocket)

#     try:
#         # initialise the role of the system
#         messages = list()
#         messages.append(
#             {
#                 "role": "system",
#                 "content": """You are Q&A bot. A highly intelligent system that answers user questions based on the 'CONTEXT' provided by the user above
#                             each 'QUESTION'. Be Specific and Descriptive. Order matters, so if for example user tells you 'summarize the following...', you have to summarize the
#                             query provided after the word 'following'. If you don't know the answer, please think rationally answer your own knowledge base.""",
#             }
#         )

#         while True:
#             # Receive message from the client
#             user_message = await websocket.receive_text()
#             # use RAG to retrieve relevant text-passages
#             retrieved = chatbot.rag_query(query_text=user_message)

#             top_k_matches = retrieved["matches"]
#             contexts = [m["metadata"]["text"] for m in top_k_matches]

#             # create the context message to pass to LLM
#             context_str = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"
#             print(context_str)
#             memory.append(context_str)

#             # Combine RAG contexts with past interactions from memory
#             combined_context = "\n\n".join(contexts + list(memory))
#             messages.append(
#                 {"role": "user", "content": f"'CONTEXT': {combined_context}"}
#             )

#             # now pass the user message to model
#             messages.append({"role": "user", "content": f"'QUESTION': {user_message}"})
#             memory.append(context_str)
#             # Send message to Azure OpenAI and get response
#             chatbot_response = await get_openai_response(websocket, messages)
#             memory.append(chatbot_response)

#             # then append to history messages
#             messages.append({"role": "assistant", "content": chatbot_response})

#     finally:
#         # Remove the websocket from the list of clients if connection is closed
#         websocket_clients.remove(websocket)


# async def get_openai_response(websocket, message: str) -> str:
#     async with AsyncClient() as client:

#         response = chatbot.respond(message)
#         responses = []
#         for chunk in response:
#             if len(chunk.choices) > 0:
#                 msg = chunk.choices[0].delta.content
#                 msg = "" if msg is None else msg
#                 responses.append(msg)

#                 await websocket.send_text(msg)

#         response = "".join(response)
#         return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
