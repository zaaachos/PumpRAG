# PumpRAG

Welcome to PumpRAG, your cutting-edge virtual Agent powered by Generative AI (GenAI) models and the innovative Retrieval-Augmented Generation (RAG) approach!

## About

PumpRAG is designed to be your ultimate companion, providing precise and relevant answers to your queries using context received from VectorDB that contains Gym Exercises, through state-of-the-art AI technology. With the integration of the RAG model, PumpRAG ensures that you receive accurate information tailored to your needs.

## Features

- **Advanced AI Capabilities:** Leveraging **Generative AI (GenAI)** models for intelligent responses.
- **Retrieval-Augmented Generation (RAG):** Incorporating the RAG model for precise and relevant answers. Used **Pinecone** for **Vector Database**.
- **Web API:** Simple **FastAPI** interface for user-agent interaction.
## To be added features
- [x] Add static templates and use FastAPI framework to load them
- [ ] Add RAG control
- [x] Create a more friendly UI

## Getting Started

To start using PumpRAG, follow these simple steps:

### 1. Environment Setup

Follow these steps to set up your environment:
- Clone the Repository:

```bash
git clone https://github.com/zaaachos/PumpRAG.git
```

- Install Dependencies:
  
It is highly recommended, to use **conda** as your virtual environment:
```bash
conda create -n venv python=3.9
```
```bash
conda activate venv
```

### 2. Dependencies
Install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

You will also need to have an Azure subscription, and create an .env file having the following variables:
```
AZURE_OPENAI_API_KEY=<YOUR_OPENAI_KEY>
OPENAI_MODEL_NAME=<YOUR_OPENAI_MODEL>
OPENAI_MODEL_VERSION=<YOUR_VERSION>
OPENAI_MODEL_DEPLOYMENT_VERSION=<YOUR_OPENAI_DEPLOYMENT_MODEL>
AZURE_OPENAI_ENDPOINT=<YOUR_OPENAI_ENDPOINT>
OPENAI_API_TYPE=azure
OPENAI_API_VERSION=2023-07-01-preview
PINECONE_API_KEY=<YOUR_PINECONE_KEY>
PINECONE_INDEX_NAME=<YOUR_PINECONE_INDEX>
EMBEDDINGS_MODEL_NAME=<YOUR_OPENAI_EMBEDDING>
```

### 3. Application
Run the Application Locally. Once dependencies are installed, you can run the FastAPI application locally by executing:

```bash
uvicorn main:app --reload
```

This will start the `uvicorn` server, and you can access the application at http://localhost:8000 in your web browser.
