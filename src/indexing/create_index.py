import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, SimpleField, SearchableField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile
)

# ------------------------------
# Import & configure logging
# ------------------------------
# setup_logger → writes logs to logs/index.log in the root folder
# install_global_exception_hook → ensures uncaught exceptions are logged
from src.config.logger_config import setup_logger, install_global_exception_hook
logger = setup_logger(__name__, log_file="index.log")
install_global_exception_hook(logger)

# ------------------------------
# Step 1: Load environment variables
# ------------------------------
# dotenv reads `.env` file and sets environment variables.
# Keeps secrets (keys, endpoints) out of code for security.
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")  # Endpoint of Azure Cognitive Search
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")    # API key for Azure Cognitive Search
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "client-manual-index")  # Index name

if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
    logger.error(" Missing Azure Search configuration in .env")
    raise ValueError("Azure Search endpoint and key must be set in environment variables")

logger.info(" Starting index creation process")
logger.debug(f"Using endpoint: {AZURE_SEARCH_ENDPOINT}, index name: {AZURE_SEARCH_INDEX}")

# ------------------------------
# Step 2: Initialize Search Index Client
# ------------------------------
# SearchIndexClient is used to manage indexes (create, delete, update).
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

# ------------------------------
# Step 3: Define Index Schema
# ------------------------------
# Fields:
# - doc_id: unique key per document (string)
# - content: searchable text (PDF text chunks)
# - embedding: vector field for semantic search (dimension 1536 = OpenAI embedding size)
# - metadata: extra info (e.g., filename, page no.)
fields = [
    SimpleField(name="doc_id", type=SearchFieldDataType.String, key=True),  # primary key
    SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),  # full-text search
    SearchField(  # vector embedding field
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,       # depends on embedding model used
        vector_search_profile_name="my-vector-config"
    ),
    SimpleField(name="metadata", type=SearchFieldDataType.String, filterable=True, facetable=True)  # metadata storage
]

# Vector search config (using HNSW algorithm for fast similarity search)
vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")],
    profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-hnsw")]
)

# Build the index definition
index = SearchIndex(
    name=AZURE_SEARCH_INDEX,
    fields=fields,
    vector_search=vector_search
)

# ------------------------------
# Step 4: Create or Update Index
# ------------------------------
# - If index already exists, delete it and recreate (fresh start).
# - Else, create a new index.
try:
    existing_indexes = list(index_client.list_index_names())
    if AZURE_SEARCH_INDEX in existing_indexes:
        logger.info(f" Index '{AZURE_SEARCH_INDEX}' already exists. Deleting old version…")
        index_client.delete_index(AZURE_SEARCH_INDEX)

    logger.info(f"Creating index '{AZURE_SEARCH_INDEX}'...")
    index_client.create_index(index)
    logger.info(f"Index '{AZURE_SEARCH_INDEX}' created successfully!")

except Exception as e:
    # Any error here (network, permissions, invalid schema) is logged with full traceback
    logger.exception(f" Failed to create/update index: {e}")
    raise
