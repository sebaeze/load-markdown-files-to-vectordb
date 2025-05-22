import { Pinecone } from "@pinecone-database/pinecone";
//import { DirectoryLoader } from "@langchain/community/document_loaders/fs/directory";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
//import { TextLoader } from "@langchain/community/document_loaders/fs/text";
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


async function ingestMarkdownFiles(directoryPath: string) {
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
      model: "nomic-embed-text", // Or another embedding model available on your Ollama server
      baseUrl: OLLAMA_BASE_URL,
    });

    // If using Google Generative AI Embeddings, replace the above with:
    /*
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: GOOGLE_API_KEY,
      model: "text-embedding-004", // Ensure this model is available and you have access
    });
    console.log("Using Google Generative AI Embeddings.");
    */
    console.log("Using Ollama Embeddings with model: nomic-embed-text");

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
// Specify the folder containing your markdown files
const MARKDOWN_FOLDER = path.resolve(__dirname, "../markdown_files");
//
if (!fs.existsSync(MARKDOWN_FOLDER)) {
  fs.mkdirSync(MARKDOWN_FOLDER);
  console.log(`Created directory: ${MARKDOWN_FOLDER}`);
}
//
ingestMarkdownFiles(MARKDOWN_FOLDER);
//