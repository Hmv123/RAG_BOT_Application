import os
from dotenv import load_dotenv
import streamlit as st
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import copy

# ==========================================================
# 0. Import & configure logging
# ==========================================================
# setup_logger → writes logs to logs/chat.log in the root folder
# install_global_exception_hook → ensures uncaught exceptions are logged
from src.config.logger_config import setup_logger, install_global_exception_hook
logger = setup_logger(__name__, log_file="chat.log")
install_global_exception_hook(logger)


# ==========================================================
# 1. Load environment variables from .env file
# ==========================================================
# These are secret keys and service endpoints stored outside the code 
# to avoid hardcoding credentials.
# load_dotenv() reads the .env file and sets the values in environment variables.
load_dotenv()

# Fetch Azure OpenAI service settings from environment
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")               # full endpoint URL, e.g., https://xxx.openai.azure.com
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")                     # primary/secondary key for Azure OpenAI
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat")  # deployment name for chat model
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "embed") # deployment name for embedding model

# Fetch Azure Cognitive Search service settings from environment
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")               # endpoint for Cognitive Search
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")                 # API key for Cognitive Search
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")                     # index name inside Cognitive Search

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
    logger.error(" Missing Azure OpenAI configuration in .env")
    raise ValueError("Azure OpenAI settings must be set in environment variables")

if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY or not AZURE_SEARCH_INDEX:
    logger.error("Missing Azure Search configuration in .env")
    raise ValueError("Azure Search settings must be set in environment variables")

logger.info("Starting chatbot app (Streamlit + Azure OpenAI + Cognitive Search)")


# ==========================================================
# 2. Initialize Azure OpenAI client
# ==========================================================
# AzureOpenAI client wraps API calls to Azure-hosted OpenAI models.
# It needs endpoint, key, and correct API version.
# NOTE: api_version must match your Azure OpenAI resource version.
aoai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-12-01-preview",   # ensure this matches supported API version
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)
logger.info("Azure OpenAI client initialized successfully")


# ==========================================================
# 3. Initialize Azure Cognitive Search client
# ==========================================================
# SearchClient allows us to query documents in Azure Cognitive Search.
# Requires endpoint, index name, and an API key credential.
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)
logger.info(f" Connected to Azure Cognitive Search index: {AZURE_SEARCH_INDEX}")


# ==========================================================
# 4. Retrieve top-k relevant chunks via vector search
# ==========================================================
def get_top_chunks(query, top_k=10):
    """
    Perform semantic search using embeddings:
    1. Convert user query into vector (embedding) using Azure OpenAI.
    2. Send that vector to Azure Cognitive Search.
    3. Retrieve top-k most similar text chunks (documents).
    """
    logger.info(f" Retrieving top {top_k} chunks for query: {query}")

    try:
        # ---- Step 1: Generate embedding for the user query ----
        query_embedding = aoai_client.embeddings.create(
            input=query,                           # text to embed
            model=AZURE_OPENAI_EMBED_DEPLOYMENT    # embedding deployment name in Azure
        ).data[0].embedding                         # extract embedding vector (list of floats)

        # ---- Step 2: Search Cognitive Search using vector similarity ----
        results = search_client.search(
            search_text="",   # left empty → only vector similarity is considered
            vector_queries=[{
                "kind": "vector",              # REQUIRED → tells service it's a vector search
                "vector": query_embedding,     # the embedding vector to search with
                "fields": "embedding",         # index field where embeddings are stored
                "k": top_k                     # number of top results to return
            }],
            select=["content", "metadata", "doc_id"]  # only fetch these fields from documents
        )

        # ---- Step 3: Collect retrieved chunks with source metadata ----
        chunks = []
        for r in results:
            content = r.get("content")       # actual chunk text
            metadata = r.get("metadata", "") # optional metadata (e.g., filename, page #)
            doc_id = r.get("doc_id", "")     # unique doc ID
            if content:
                chunks.append(f"{content}\n(Source: {metadata or doc_id})")

        logger.info(f" Retrieved {len(chunks)} chunks for query: {query}")
        return chunks

    except Exception as e:
        logger.exception("Error occurred during vector search")
        return []


# ==========================================================
# 5. Generate an AI-powered answer using retrieved chunks
# ==========================================================
def generate_answer(user_query, chat_history):
    """
    Generate chatbot response:
    - Search Cognitive Search for relevant context
    - Construct prompt with chat history + retrieved context
    - Call Azure OpenAI chat model to generate an answer
    """
    logger.info(f"Generating answer for user query: {user_query}")

    # ---- Step 1: Retrieve supporting context from Cognitive Search ----
    context_chunks = get_top_chunks(user_query, top_k=10)
    context_text = "\n".join(context_chunks) if context_chunks else ""

    # ---- Step 2: Copy chat history ----
    messages = copy.deepcopy(chat_history)

    # ---- Step 3: Add system instruction ----
    messages.append({
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the provided context to answer questions accurately. "
            "If the context is insufficient, give the most careful, concise answer possible. "
            "Always cite the sources when available. Do not hallucinate beyond the context."
        )
    })

    # ---- Step 4: Add user query (with context attached) ----
    messages.append({
        "role": "user",
        "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"
    })

    # ---- Step 5: Generate AI response from Azure OpenAI ----
    try:
        response = aoai_client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,   # which deployment to use
            messages=messages,                    # chat history + context
            temperature=0.2,                      # low temp → less randomness, more factual
            max_tokens=1000                       # limit to avoid overly long responses
        )
        answer = response.choices[0].message.content
        logger.info("Successfully generated AI response")

    except Exception as e:
        logger.exception("Failed to generate AI response")
        answer = f"Error generating response: {e}"

    # ---- Step 6: Update chat history with new turn ----
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history


# ==========================================================
# 6. Streamlit User Interface (UI)
# ==========================================================
# Streamlit is used here to build a simple chatbot front-end.
st.set_page_config(page_title="Citi", page_icon="🏦")
st.title("Client Manual Chatbot 🏦")

# ---- Initialize session state for storing chat history ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    logger.info("Initialized empty chat history")

# ---- Render chat history in UI ----
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# ---- Input box at bottom of page ----
if prompt := st.chat_input("Ask about the Client Manual…"):
    logger.info(f" User asked: {prompt}")
    _, st.session_state.chat_history = generate_answer(prompt, st.session_state.chat_history)
    st.rerun()

