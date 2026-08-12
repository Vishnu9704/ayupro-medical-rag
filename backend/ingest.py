import os
import io
import boto3
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DB_DIR = "./chroma_db"
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "your-medical-pdfs-bucket")

print("⚡ Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True}
)

# 1. Check if database already exists
db_exists = os.path.exists(DB_DIR) and os.listdir(DB_DIR)

if db_exists:
    print(f"📂 Existing vector database found at '{DB_DIR}'. Checking for new files in S3...")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Extract already-processed source filenames from Chroma metadata
    existing_metadata = vectordb.get(include=["metadatas"])
    processed_sources = set()
    if existing_metadata and existing_metadata.get("metadatas"):
        for meta in existing_metadata["metadatas"]:
            if meta and "source" in meta:
                processed_sources.add(meta["source"])
else:
    print("📂 No existing database found. Building a fresh one from scratch...")
    vectordb = None
    processed_sources = set()

# 2. Connect to S3 and list files
print("🔄 Connecting to AWS S3...")
s3_client = boto3.client('s3')

try:
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
    if 'Contents' not in response:
        print(f"❌ Error: S3 bucket '{S3_BUCKET_NAME}' is empty.")
        exit(1)
        
    s3_pdf_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].lower().endswith('.pdf')]
    
    # 3. Filter out files that are already inside Chroma
    new_files = [file_key for file_key in s3_pdf_files if file_key not in processed_sources]
    
    if not new_files and db_exists:
        print("✅ Up to date! No new PDFs found in S3. Skipping ingestion.")
        exit(0)
        
    print(f"📄 Found {len(new_files)} new PDF(s) to process: {new_files}")
    
    all_new_documents = []
    
    # 4. Stream only the new PDFs
    for file_key in new_files:
        print(f"📥 Streaming and parsing new file '{file_key}' from S3...")
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        pdf_stream = io.BytesIO(s3_object['Body'].read())
        
        reader = PdfReader(pdf_stream)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                doc = Document(
                    page_content=text,
                    metadata={"source": file_key, "page": page_num + 1}
                )
                all_new_documents.append(doc)
                
    if not all_new_documents:
        print("❌ Error: No text could be extracted from the new PDFs.")
        exit(1)

    # 5. Chunk the new documents
    print("🔪 Splitting new documents into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_new_documents)
    print(f"🔪 Generated {len(chunks)} new text chunks.")

    # 6. Add to database (creates DB if missing, or appends cleanly if it exists)
    if not db_exists:
        print("⚡ Creating new vector database with embeddings...")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
    else:
        print("⚡ Appending new chunks to existing vector database...")
        vectordb.add_documents(chunks)

    print("✅ Incremental ingestion complete! New vectors successfully added.")

except Exception as e:
    print(f"❌ Error during ingestion pipeline: {e}")
    exit(1)