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
    console.log("Initializing Pinecone store for sequential ingestion...");
    const pineconeStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: index,
      namespace: process.env.PINECONE_NAMESPACE || "ns-markdown-docs",
    });

    console.log(
      `Starting sequential ingestion of ${docs.length} document chunks...`
    );
    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      // Log more detailed information for each chunk
      const source = doc.metadata?.source || "N/A";
      const contentLength = doc.pageContent.length;
      console.log(
        `Ingesting chunk ${i + 1}/${
          docs.length
        }. Source: ${source}, Length: ${contentLength} chars.`
      );
      try {
        // addDocuments expects an array of Document objects
        await pineconeStore.addDocuments([doc]);
      } catch (e) {
        console.error(`Error ingesting chunk ${i + 1}/${docs.length} (Source: ${source}):`, e);
        // Depending on requirements, you might want to re-throw the error
        // or collect failed chunks for later processing. Here, we just log and continue.
      }
    }
    console.log("Markdown files successfully loaded into Pinecone!");
  } catch (error) {
    console.error("Error ingesting markdown files:", error);
  }
}
//