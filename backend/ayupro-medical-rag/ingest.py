import os
import io
import boto3
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
DB_DIR = "./chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not BUCKET_NAME:
    print("❌ Error: S3_BUCKET_NAME is not set in your .env file!")
    exit(1)

s3_client = boto3.client('s3')

print("🔄 Initializing HuggingFace Embedding Model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print(f"📂 Initializing/Loading Chroma vector database at {DB_DIR}...")
vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model
)

def chunk_text(text, chunk_size=600, overlap=100):
    """Splits text into smaller chunks with an overlap to maintain context."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]

print(f"📥 Scanning S3 bucket `{BUCKET_NAME}` for PDFs to ingest...")

try:
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    
    if 'Contents' not in response:
        print("❌ S3 bucket is empty or no files found.")
        exit(1)

    documents = []
    for obj in response['Contents']:
        file_key = obj['Key']
        
        if file_key.lower().endswith('.pdf'):
            print(f"📄 Processing PDF from S3: {file_key}")
            
            # 1. Stream PDF directly from S3 into memory
            pdf_object = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
            pdf_stream = io.BytesIO(pdf_object['Body'].read())
            
            # 2. Extract text per page
            reader = PdfReader(pdf_stream)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                
                if page_text:
                    chunks = chunk_text(page_text)
                    for chunk in chunks:
                        # 3. Wrap in LangChain Document format with metadata for citations
                        doc = Document(
                            page_content=chunk,
                            metadata={"source": file_key, "page": page_num + 1}
                        )
                        documents.append(doc)

    if documents:
        print(f"🚀 Embedding and adding {len(documents)} chunks to Chroma vector store...")
        vectordb.add_documents(documents)
        print("✅ Ingestion complete! Chroma DB successfully populated from S3 PDFs.")
    else:
        print("⚠️ No valid text content found in the S3 PDFs to ingest.")

except Exception as e:
    print(f"❌ Error during S3 ingestion pipeline: {e}")