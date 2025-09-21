from io import UnsupportedOperation
import nltk
from pocketflow import AsyncParallelBatchNode, AsyncFlow, Node
from eigen_client.data_types import Document
# Setting up punkt (for sentence splitting)
nltk.download("punkt")

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
            import fitz
            raw_text = ""
            with fitz.open(filename) as pdf_doc:
                for page in pdf_doc:
                    raw_text = page.get_text()
                return raw_text
        else:
            raise UnsupportedOperation("Passed an unsupported file extension")



    def exec(self, prep_res):
        sentences = nltk.sent_tokenize(prep_res)
        return sentences

    def post(self, shared, prep_res, exec_res):
        shared["sentences"] = exec_res
        
class ChunkMaker(Node):
    pass

class Embedder(Node):
    pass

class VectorStoreWriter(Node):
    pass




