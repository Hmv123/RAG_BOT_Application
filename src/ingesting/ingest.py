import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from llama_index.core.node_parser import SentenceSplitter

# ------------------------------
# Import & configure logging
# ------------------------------
# setup_logger → writes logs to logs/ingest.log in the root folder
# install_global_exception_hook → ensures uncaught exceptions are logged
from src.config.logger_config import setup_logger, install_global_exception_hook
logger = setup_logger(__name__, log_file="ingest.log")
install_global_exception_hook(logger)

# ------------------------------
# Load environment variables
# ------------------------------
# dotenv looks for a `.env` file in the current directory and loads variables from it
# This allows you to keep API keys and endpoints outside the code (better security)
load_dotenv()

# ------------------------------
# Azure configuration values
# ------------------------------
# These values are fetched from the .env file
# They include endpoints, API keys, and container/index names needed for services
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")           # Endpoint of the Azure AI Search service
search_key = os.getenv("AZURE_SEARCH_API_KEY")                 # API key for Azure AI Search
search_index = os.getenv("AZURE_SEARCH_INDEX")                 # Name of the search index

aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")             # Endpoint of Azure OpenAI resource
aoai_key = os.getenv("AZURE_OPENAI_API_KEY")                   # API key for Azure OpenAI
aoai_embed_model = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")  # Deployment name of embedding model
aoai_api_version = os.getenv("OPENAI_API_VERSION")             # API version for Azure OpenAI

if not search_endpoint or not search_key or not search_index:
    logger.error("Missing Azure Search configuration in .env")
    raise ValueError("Azure Search settings must be set in environment variables")

if not aoai_endpoint or not aoai_key or not aoai_embed_model:
    logger.error("Missing Azure OpenAI configuration in .env")
    raise ValueError("Azure OpenAI settings must be set in environment variables")

logger.info("Starting ingestion process")
logger.debug(f"Using index: {search_index}, endpoint: {search_endpoint}")

# ------------------------------
# Step 1: Download PDFs from Azure Blob Storage
# ------------------------------
# Create a local folder to temporarily store downloaded PDFs
local_dir = "./data"

# ------------------------------
# Step 2: Load documents from the local folder
# ------------------------------
logger.info(" Reading PDFs from local data folder...")
try:
    documents = SimpleDirectoryReader(
        local_dir,
        file_extractor={".pdf": PDFReader()}   # Only .pdf files are handled with PDFReader
    ).load_data()
    logger.info(f"Loaded {len(documents)} documents")
except Exception as e:
    logger.exception("Failed while reading documents")
    raise

# ------------------------------
# Step 3: Setup embedding model (Azure OpenAI)
# ------------------------------
# Embeddings turn text into vectors (lists of numbers) that capture meaning/semantics.
# AzureOpenAIEmbedding is a wrapper around the Azure OpenAI embeddings API.
logger.info("Initializing Azure OpenAI embedding model...")
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-large",                        # Embedding model family (3072 dimensions)
    deployment_name=aoai_embed_model,                      # Your deployment name in Azure portal
    api_key=aoai_key,                                      # Azure OpenAI API key
    azure_endpoint=aoai_endpoint,                          # Endpoint for the resource
    api_version=aoai_api_version,                          # API version string
)

# ------------------------------
# Step 4: Setup Azure AI Search Vector Store
# ------------------------------
logger.info("Connecting to Azure AI Search vector store...")
search_client = SearchClient(
    endpoint=search_endpoint,               # Service endpoint
    index_name=search_index,                # Index name inside Azure AI Search
    credential=AzureKeyCredential(search_key)  # Auth with API key
)

vector_store = AzureAISearchVectorStore(
    search_or_index_client=search_client,  # The connected client
    id_field_key="doc_id",                     # Maps unique ID of each chunk
    chunk_field_key="content",             # Field to store actual text chunk
    embedding_field_key="embedding",       # Field to store the generated embedding vector
    metadata_string_field_key="metadata",  # Field for extra info (e.g., source filename, page number)
    doc_id_field_key="doc_id",             # Field for original doc ID (groups chunks together)
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ------------------------------
# Step 5: Split documents into smaller chunks
# ------------------------------
logger.info("Splitting documents into smaller chunks...")
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
nodes = splitter.get_nodes_from_documents(documents)
logger.info(f"Split {len(documents)} documents into {len(nodes)} chunks")

# ------------------------------
# Step 6: Build & push index into Azure AI Search
# ------------------------------
logger.info("Building index and pushing to Azure AI Search...")
try:
    index = VectorStoreIndex(
        nodes,                           # List of text chunks
        storage_context=storage_context, # Where to save them (Azure AI Search index)
        embed_model=embed_model          # Embedding model to generate semantic vectors
    )
    logger.info(f"Successfully pushed {len(nodes)} chunks into Azure AI Search index '{search_index}'")
except Exception as e:
    logger.exception("Failed while building/pushing index")
    raise
