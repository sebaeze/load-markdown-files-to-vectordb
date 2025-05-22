/*
*
*/
import path from "path";
import { ingestMarkdownFiles } from "./ingest";
import * as fs from 'fs';
//
let FOLDER = "" ;
if ( path.isAbsolute(process.env.MARKDOWN_FOLDER||"") ){
    FOLDER = process.env.MARKDOWN_FOLDER||"";
} else {
    FOLDER = path.resolve(__dirname, process.env.MARKDOWN_FOLDER||"");
}
//
if (!fs.existsSync(FOLDER)) {
  fs.mkdirSync(FOLDER);
  console.log(`Created directory: ${FOLDER}`);
}
//
ingestMarkdownFiles(FOLDER);
//