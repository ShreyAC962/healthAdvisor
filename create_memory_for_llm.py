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
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks = create_chunks(extracted_data=documents)
print("Length of Text chunks: ", len(text_chunks))

# Create Vector Embeddings
# Store Embeddings in FAISS