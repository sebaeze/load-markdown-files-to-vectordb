import { Pinecone } from "@pinecone-database/pinecone";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { PineconeStore } from "@langchain/pinecone";
import * as dotenv from "dotenv";
import * as fs from "fs";
import * as path from "path";

dotenv.config();

const PINECONE_API_KEY:string = process.env.PINECONE_API_KEY||"";
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT||"";
const PINECONE_INDEX_NAME:string = process.env.PINECONE_INDEX_NAME||"";
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL;

if (!PINECONE_API_KEY || !PINECONE_ENVIRONMENT || !PINECONE_INDEX_NAME) {
  throw new Error(
    "Pinecone API key, environment, or index name not provided in .env"
  );
}

if (!OLLAMA_BASE_URL) {
  throw new Error("Ollama base URL not provided in .env");
}


export async function ingestMarkdownFiles(directoryPath: string) {
  try {
    console.log(`Loading markdown files from: ${directoryPath}`);

    // Ensure the directory exists
    if (!fs.existsSync(directoryPath)) {
      console.error(`Directory not found: ${directoryPath}`);
      return;
    }

    // 1. List all .md files from the folder
    const loader = new DirectoryLoader(directoryPath, {
      ".md": (path) => new TextLoader(path),
    });

    const rawDocuments = await loader.load();
    console.log(`Found ${rawDocuments.length} markdown files.`);

    // 2. Split documents into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const docs = await textSplitter.splitDocuments(rawDocuments);
    console.log(`Split documents into ${docs.length} chunks.`);

    // 3. Initialize Embeddings (using Ollama's nomic-embed-text)
    const embeddings = new OllamaEmbeddings({
      model: process.env.MODEL_NAME||"nomic-embed-text",
      baseUrl: OLLAMA_BASE_URL,
    });

    // 4. Initialize Pinecone
    const pinecone = new Pinecone({
      apiKey: PINECONE_API_KEY
    });

    const index = pinecone.Index(PINECONE_INDEX_NAME);

    // Check if the index exists, create if not
    const indexList = await pinecone.listIndexes();
    const indexExists = indexList.indexes?.some(
      (idx) => idx.name === PINECONE_INDEX_NAME
    );
    //
    console.log(`Using existing Pinecone index: ${PINECONE_INDEX_NAME}`);
    //
    console.log("Loading documents into Pinecone...");
    await PineconeStore.fromDocuments(docs, embeddings, {
      pineconeIndex: index,
      namespace: "ns-markdown-docs"
    });

    console.log("Markdown files successfully loaded into Pinecone!");
  } catch (error) {
    console.error("Error ingesting markdown files:", error);
  }
}
//