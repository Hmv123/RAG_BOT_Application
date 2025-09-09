This project is a Retrieval-Augmented Generation (RAG) chatbot using:

Azure Cognitive Search → Stores and retrieves document embeddings

Azure OpenAI → Generates embeddings + answers with GPT models

LlamaIndex → Handles document parsing, chunking, and pushing to the index

Streamlit → Simple web interface for chatbot

-------------------Setup Instructions-------------------------------------------
1.Clone repository (or create project folder)
git clone <your-repo-url> rag-bot
cd rag-bot
2.Create a .env file in the root folder
Note: Do not commit .env to GitHub. Add it to .gitignore.
3.Install dependencies 
***Using Pip
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .\.venv\Scripts\Activate.ps1 # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

***Using conda
conda env create -f environment.yaml
conda activate rag-bot
------------------------Running instructions-----------------
Step 1 – Create index in Azure Cognitive Search
python src/indexing/create_index.py
Step 2 – Load PDFs and push embeddings
python src/ingesting/Ingest.py
Step 3 – Run the chatbot UI
streamlit run src/query/query.py
Opens a local web app (default: http://localhost:8501)
Ask questions → answers are grounded in your indexed documents

------------------Notes-------------
PDF Loading: Right now, files are read from the ./data/ folder.
If you want to read directly from Azure Blob Storage, add azure-storage-blob to requirements.txt and update Ingest.py.
Azure SDK Version Pinning:
azure-search-documents==11.5.3 is pinned because your code uses the vector_queries syntax, which matches 11.5.x.
Upgrading to >=11.6 will require code changes.
Updating Dependencies:pip install -r requirements.txt --upgrade
--------------------------------------
