---------------Overview----------
This project is a Retrieval-Augmented Generation (RAG) chatbot using:
Azure Cognitive Search → Stores and retrieves document embeddings
Azure OpenAI → Generates embeddings + answers with GPT models
LlamaIndex → Handles document parsing, chunking, and pushing to the index
Streamlit → Simple web interface for chatbot

/*********Project flow***************/

RAG_BOT_Application/
│
├── .gitignore
├── requirements.txt
├── environment.yaml
├── README.md
│
├── logs/
│   ├── index.log
│   ├── ingestion.log
│   └── query.log
│
├── app/
│   └── Bash_scripts/
│       ├── run_all.sh
│       ├── run_index.sh
│       ├── run_ingest.sh
│       └── run_chatbot.sh
│
├── data/                       # Local folder to hold downloaded PDFs
│
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── logger_config.py        # Logger configuration
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── create_index.py         # Search index creation
│   │
│   ├── ingesting/
│   │   ├── __init__.py
│   │   └── ingestion.py            # Loading the index by reading pdf's
│   │
│   ├── query/
│   │   ├── __init__.py
│   │   └── query.py                # Retrieval and querying
│   │
│   └── __init__.py                 # Marks src as a package
│
└── .env                            # Environment variables (not tracked in git)


-------------------Setup Instructions-------------------------------------------
1.Clone repository (or create project folder)
git clone <your-repo-url> rag-bot
cd rag-bot
2.Create a .env file in the root folder
Note: Do not commit .env to GitHub. Add it to .gitignore.

3.Install dependencies --Below are the 2 options
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
Run the Bash scripts.
If you are running it for the first time--run_all.sh
Else
run only the run_chatbot.sh
------------------Notes-------------
PDF Loading: Right now, files are read from the ./data/ folder.
If you want to read directly from Azure Blob Storage, add azure-storage-blob to requirements.txt and update Ingest.py.
Azure SDK Version Pinning:
azure-search-documents==11.5.3 is pinned because your code uses the vector_queries syntax, which matches 11.5.x.
Upgrading to >=11.6 will require code changes.
Updating Dependencies:pip install -r requirements.txt --upgrade
--------------------------------------
