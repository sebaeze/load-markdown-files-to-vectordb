# Markdown to Pinecone VectorDB Loader

## Overview

This application is designed to process local markdown files, generate vector embeddings for their content using Ollama, and then store these embeddings in a Pinecone vector database.

The script will:
1.  Load all `.md` files from a specified directory (`markdown_files`).
2.  Split the document content into manageable chunks.
3.  Generate embeddings for each chunk using a model served by Ollama (e.g., `nomic-embed-text`).
4.  Upsert these chunks and their embeddings into a specified Pinecone index under the namespace `ns-markdown-docs`.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

*   **Node.js and npm**: Download and install from nodejs.org.
*   **Git**: For cloning the repository.
*   **Ollama**:
    *   Installed and running. You can find installation instructions on the Ollama website.
    *   The embedding model specified in the script (default: `nomic-embed-text`) must be pulled. You can pull it using:
        ```bash
        ollama pull nomic-embed-text
        ```
*   **Pinecone Account**:
    *   An active Pinecone account.
    *   An existing Pinecone index. **This script does not create the index if it's missing.**
    *   The Pinecone index must be configured with the correct dimension for the embedding model you are using. For `nomic-embed-text`, the dimension is **768**.

## Setup

1.  **Clone the Repository**:
    Open your terminal and run:
    ```bash
    git clone https://github.com/sebaeze/load-markdown-files-to-vectordb.git
    cd load-markdown-files-to-vectordb
    ```

2.  **Install Dependencies**:
    Navigate to the project directory and install the necessary Node.js packages:
    ```bash
    npm install
    ```

3.  **Configure Environment Variables**:
    Create a `.env` file in the root of the project directory. Copy the following content into it and replace the placeholder values with your actual credentials and settings:

    ```env
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT"
    PINECONE_INDEX_NAME="your-pinecone-index-name"
    OLLAMA_BASE_URL="http://localhost:11434"
    ```

    *   `PINECONE_API_KEY`: Your API key from the Pinecone console.
    *   `PINECONE_ENVIRONMENT`: The environment of your Pinecone index (e.g., `gcp-starter`, `us-west1-gcp`).
    *   `PINECONE_INDEX_NAME`: The name of your **existing** Pinecone index.
    *   `OLLAMA_BASE_URL`: The base URL of your running Ollama instance. Defaults to `http://localhost:11434` if Ollama is running locally.

## Usage

1.  **Prepare Markdown Files**:
    Place your markdown (`.md`) files into the `markdown_files` directory located at the root of the project. If this directory does not exist, the script will create it automatically when run.

2.  **Run the Ingestion Script**:
    Execute the script using npm:
    ```bash
    npm start
    ```
    *Note: This command assumes your `package.json` has a `start` script configured. For example:*
    *   For development (running TypeScript directly): `"start": "ts-node src/ingest.ts"`
    *   For production (after compiling TypeScript to JavaScript, e.g., with `tsc`): `"start": "node dist/ingest.js"`
    *   *You may need to install `ts-node` globally (`npm install -g ts-node`) or as a dev dependency if you use the `ts-node` approach.*

    Upon execution, the script will:
    *   Log its progress to the console, including the number of files found and chunks created.
    *   Process each markdown file, generate embeddings via Ollama, and load them into your Pinecone index.
    *   Print a success message upon completion or any errors encountered during the process.

## How It Works

1.  **Configuration**: Reads API keys, Pinecone details, and Ollama URL from the `.env` file.
2.  **Directory Setup**: Checks for the `markdown_files` directory and creates it if it's not present.
3.  **Document Loading**: Uses LangChain's `DirectoryLoader` with a `TextLoader` for `.md` files to load all markdown documents from the `markdown_files` directory.
4.  **Text Splitting**: Employs `RecursiveCharacterTextSplitter` to break down the loaded documents into smaller, overlapping chunks suitable for embedding.
5.  **Embedding Generation**: Initializes `OllamaEmbeddings` to connect to your Ollama instance and generate vector embeddings for each text chunk using the specified model (e.g., `nomic-embed-text`).
6.  **Pinecone Initialization**: Sets up the Pinecone client using your API key.
7.  **Data Ingestion**: Uses `PineconeStore.fromDocuments` to efficiently batch and upsert the document chunks and their embeddings into the designated Pinecone index and namespace (`ns-markdown-docs`).
