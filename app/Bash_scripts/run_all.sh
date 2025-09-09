#!/bin/bash
# Run the create_index module
echo "1.Running Create Index..."
python -m src.indexing.create_index

# Run the load_index module
echo "2.Running Load Index..."
python -m src.ingesting.ingest


# Run the chatbot (Streamlit UI)
echo "3.Running Chatbot..."
python -m streamlit run src/query/query.py