from os import wait
from utility import extract_json_from_codeblock, call_llm, semantic_chunking
from io import UnsupportedOperation
import nltk
from nltk.tokenize import sent_tokenize
from pocketflow import AsyncParallelBatchNode, AsyncFlow, Node, Flow
from eigen_client.client import Client
from eigen_client.data_types import Document

RAW_TEXT_EXTENSIONS = ("md", "txt")


# --- Document Level Flow ---
class SentenceSplitter(Node): 
    def prep(self, shared):
        filename = self.params["filename"]

        extension = filename.split('.')[-1].lower()

        if extension in RAW_TEXT_EXTENSIONS:
            # load the file
            with open(filename) as f:
                return f.read()

        elif extension == "pdf":
            import pymupdf
            raw_text = ""
            with pymupdf.open(filename) as doc:  # open document
                text = chr(12).join([page.get_text() for page in doc])
                return text
        else:
            raise UnsupportedOperation("Passed an unsupported file extension")


    def exec(self, prep_res):
        sentences = nltk.sent_tokenize(prep_res)
        return sentences

    def post(self, shared, prep_res, exec_res):
        shared["sentences"] = exec_res
        # Pass the filename to the end
        shared["sentences"].append(self.params["filename"])
        
class DocumentMaker(Node):
    def prep(self, shared):
        sentences = shared["sentences"]
        # We know it's there since previous node
        filename = sentences.pop() 
        return (filename, sentences)

    def exec(self, prep_res):
        import uuid
        filename, sentences = prep_res
        file_directory = "stuff" #self.params["file_directory"]

        metadata = {
                    "file_directory": file_directory,
                    "filename" : filename, 
                    "sentence_length" : len(sentences),
                    "chunk_count": 0, # this will be overwritten
                }

        # Proceed to the actual semantic chunking
        chunks, chunk_count = semantic_chunking(sentences)
        metadata["chunk_count"] = chunk_count
        documents = [ Document(id=uuid.uuid4(), data=chunk, metadata={**metadata, "chunk_number": chunk_number}) 
                     for (chunk, chunk_number) in chunks ]
        return documents

    def post(self, shared, prep_res, exec_res):
        shared["documents"] = exec_res


class VectorStoreWriter(Node):
    def prep(self, shared):
        return shared["client"], shared["documents"]

    def exec(self, prep_res):
        client, documents = prep_res
        if not documents:
            raise ValueException("VectorStoreWriter received empty documents list")

        filename = documents[0].metadata.get("filename")
        index = client.create_index_from_model(
            index_name=filename,
            model_name="mxbai-embed-large:335m",
            model_provider="ollama",
        )
        index.upsert_docs(documents)



# Downloading the punkt stuff if needed
nltk.download("punkt")
nltk.download("punkt_tab")
# Testing the logic
node = SentenceSplitter()
node.params["filename"] = "frankenstein.txt"

node2 = DocumentMaker()

node3 = VectorStoreWriter()
node >> node2 >> node3
flow = Flow(start=node)

client = Client(
    url="http://localhost:8080",
    api_key="eigendb-***",
)

shared = {"client" : client}

flow.run(shared)
