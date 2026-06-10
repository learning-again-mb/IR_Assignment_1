# End-to-End Information Retrieval System using Streamlit

This submission implements an Information Retrieval system with a Streamlit front end.  
The code flow follows the webinar sample style:

1. Step 1: Read dataset
2. Step 2: Preprocess text
3. Step 3: Create indexes
4. Step 4: Process user queries
5. Step 5: Display comparison tables and inferences

## Features Covered

- Upload `.txt` document collection from Streamlit
- View uploaded or sample documents
- Tokenization
- Lowercasing
- Stop word removal
- Hyphen handling
- Stemming and lemmatization
- Inverted index creation
- Boolean AND query search
- Stemming vs lemmatization comparison
- Biword index phrase query search
- Positional index phrase query search
- False positive explanation for biword index
- Dictionary search using Binary Search Tree and B-Tree style sorted search
- Query search time and retrieval time comparison table
- Tolerant retrieval using:
  - Wildcard query
  - Edit distance spelling correction
  - K-gram index

## Installation Steps

Install Python 3.9 or above.

Install dependencies:

```bash
pip install streamlit pandas nltk
```

## Command to Run the App

```bash
streamlit run app.py
```

## Dataset

The app includes a `sample_docs` folder. You can also upload your own `.txt` files directly from the Streamlit sidebar.

## Files Submitted

- `app.py` - Streamlit application code
- `README.md` - setup and running instructions
- `requirements.txt` - dependency list
- `sample_docs/` - sample text document collection

## Note for BITS Lab Portal

Run the complete workflow only through the Streamlit front end. The app displays intermediate outputs such as tokens, indexes, query results, performance tables, and final inferences.
