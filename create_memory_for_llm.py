# Load raw PDF(s)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
documents = load_pdf_files(data=DATA_PATH)
print("Length of PDF Pages: ", len(documents))
# Create Chunks
# Create Vector Embeddings
# Store Embeddings in FAISS