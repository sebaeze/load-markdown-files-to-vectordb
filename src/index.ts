/*
*
*/
import path from "path";
import { ingestMarkdownFiles } from "./ingest";
import * as fs from 'fs';
//
const MARKDOWN_FOLDER = path.resolve(__dirname, "../markdown_files");
//
if (!fs.existsSync(MARKDOWN_FOLDER)) {
  fs.mkdirSync(MARKDOWN_FOLDER);
  console.log(`Created directory: ${MARKDOWN_FOLDER}`);
}
ingestMarkdownFiles(MARKDOWN_FOLDER);
//