from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load PDF files
DATA_PATH = 'data/'


def load_pdf_files(data):
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()  # gives you a list of documents with metadata and content
    return documents


documents = load_pdf_files(data=DATA_PATH)

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)  # to keep the context intact it will have buffer of 20 chunks before and after
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = create_chunks(extracted_data=documents)

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store Embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())
