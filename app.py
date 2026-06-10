# Example - End-to-end Information Retrieval system using Streamlit
# The code flow intentionally follows the webinar examples:
# Step 1: Read dataset
# Step 2: Preprocess text
# Step 3: Create indexes
# Step 4: Process user queries
# Step 5: Display experimental comparison and inference

import os
import re
import time
import math
import tempfile
from collections import defaultdict

import pandas as pd
import streamlit as st

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.metrics.distance import edit_distance

# Ensure required NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------------------------------------------------------
# Step 1: Read the dataset
# ---------------------------------------------------------
DATASET_DIR = "sample_docs"       # Default dataset folder; can be replaced using upload option


def read_files(dataset_dir):
    """Read all .txt files from the dataset folder."""
    corpus = {}
    if not os.path.exists(dataset_dir):
        return corpus

    for filename in sorted(os.listdir(dataset_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(dataset_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                corpus[filename] = file.read()
    return corpus


def read_uploaded_files(uploaded_files):
    """Read uploaded Streamlit files into a corpus dictionary."""
    corpus = {}
    for uploaded_file in uploaded_files:
        corpus[uploaded_file.name] = uploaded_file.read().decode('utf-8', errors='ignore')
    return corpus


# ---------------------------------------------------------
# Step 2: Preprocess the text
# ---------------------------------------------------------
def handle_hyphen(text, hyphen_option):
    if hyphen_option == "Split hyphenated words":
        return re.sub(r"[-–—]", " ", text)
    elif hyphen_option == "Join hyphenated words":
        return re.sub(r"(?<=\w)[-–—](?=\w)", "", text)
    else:
        return text


def preprocess_text(text, lower_case=True, remove_stop=True, hyphen_option="Split hyphenated words", method="Lemmatization"):
    """Tokenization, lowercasing, stopword removal, hyphen handling, stemming/lemmatization."""

    # Hyphen handling
    text = handle_hyphen(text, hyphen_option)

    # Lowercasing
    if lower_case:
        text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation and non-alphanumeric tokens
    tokens = [re.sub(r'\W+', '', token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token != ""]

    # Stopword removal
    if remove_stop:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

    # Stemming or Lemmatization
    if method == "Stemming":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    elif method == "Lemmatization":
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def preprocess_corpus(corpus, lower_case, remove_stop, hyphen_option, method):
    preprocessed_corpus = {}
    for doc_id, text in corpus.items():
        preprocessed_corpus[doc_id] = preprocess_text(text, lower_case, remove_stop, hyphen_option, method)
    return preprocessed_corpus


# ---------------------------------------------------------
# Step 3: Create inverted, biword and positional indexes
# ---------------------------------------------------------
def create_inverted_index(corpus):
    inverted_index = defaultdict(set)
    for doc_id, tokens in corpus.items():
        for token in tokens:
            inverted_index[token].add(doc_id)
    return {key: sorted(value) for key, value in sorted(inverted_index.items())}


def create_biword_index(corpus):
    biword_index = defaultdict(set)
    for doc_id, tokens in corpus.items():
        for i in range(len(tokens) - 1):
            biword = tokens[i] + " " + tokens[i + 1]
            biword_index[biword].add(doc_id)
    return {key: sorted(value) for key, value in sorted(biword_index.items())}


def create_positional_inverted_index(corpus):
    positional_index = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in corpus.items():
        for position, token in enumerate(tokens):
            positional_index[token][doc_id].append(position)
    return {term: dict(postings) for term, postings in sorted(positional_index.items())}


# ---------------------------------------------------------
# Step 4: Query processing functions
# ---------------------------------------------------------
def boolean_and_query(query, inverted_index, lower_case, remove_stop, hyphen_option, method):
    query_tokens = preprocess_text(query, lower_case, remove_stop, hyphen_option, method)
    if len(query_tokens) == 0:
        return [], query_tokens

    result = set(inverted_index.get(query_tokens[0], []))
    for token in query_tokens[1:]:
        result = result.intersection(set(inverted_index.get(token, [])))
    return sorted(result), query_tokens


def biword_phrase_query(query, biword_index, lower_case, remove_stop, hyphen_option, method):
    query_tokens = preprocess_text(query, lower_case, remove_stop, hyphen_option, method)
    if len(query_tokens) < 2:
        return [], query_tokens, []

    query_biwords = []
    for i in range(len(query_tokens) - 1):
        query_biwords.append(query_tokens[i] + " " + query_tokens[i + 1])

    result = set(biword_index.get(query_biwords[0], []))
    for biword in query_biwords[1:]:
        result = result.intersection(set(biword_index.get(biword, [])))

    return sorted(result), query_tokens, query_biwords


def positional_phrase_query(query, positional_index, lower_case, remove_stop, hyphen_option, method):
    query_tokens = preprocess_text(query, lower_case, remove_stop, hyphen_option, method)
    if len(query_tokens) == 0:
        return [], query_tokens

    candidate_docs = set(positional_index.get(query_tokens[0], {}).keys())
    for token in query_tokens[1:]:
        candidate_docs = candidate_docs.intersection(set(positional_index.get(token, {}).keys()))

    final_result = []
    for doc_id in candidate_docs:
        first_positions = positional_index[query_tokens[0]][doc_id]
        for start_pos in first_positions:
            matched = True
            for offset in range(1, len(query_tokens)):
                next_token = query_tokens[offset]
                if start_pos + offset not in positional_index[next_token][doc_id]:
                    matched = False
                    break
            if matched:
                final_result.append(doc_id)
                break

    return sorted(final_result), query_tokens


# ---------------------------------------------------------
# Step 5: Dictionary search using BST and B-Tree
# ---------------------------------------------------------
class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = BSTNode(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node, key):
        if key < node.key:
            if node.left is None:
                node.left = BSTNode(key)
            else:
                self._insert(node.left, key)
        elif key > node.key:
            if node.right is None:
                node.right = BSTNode(key)
            else:
                self._insert(node.right, key)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None:
            return False
        if node.key == key:
            return True
        elif key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)


class BTree:
    """Simple B-Tree-like dictionary wrapper using sorted keys and binary search.
    This is suitable for assignment-level comparison with BST search flow.
    """
    def __init__(self, keys):
        self.keys = sorted(keys)

    def search(self, key):
        low = 0
        high = len(self.keys) - 1
        while low <= high:
            mid = (low + high) // 2
            if self.keys[mid] == key:
                return True
            elif key < self.keys[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return False


def create_bst(dictionary_terms):
    bst = BinarySearchTree()
    for term in sorted(dictionary_terms):
        bst.insert(term)
    return bst


def compare_dictionary_search(query_list, dictionary_terms, inverted_index):
    bst = create_bst(dictionary_terms)
    btree = BTree(dictionary_terms)
    rows = []

    for query in query_list:
        query = query.strip().lower()
        if query == "":
            continue

        start = time.perf_counter()
        bst_found = bst.search(query)
        bst_search_time = (time.perf_counter() - start) * 1000000

        start = time.perf_counter()
        bst_docs = inverted_index.get(query, [])
        bst_retrieval_time = (time.perf_counter() - start) * 1000000

        start = time.perf_counter()
        btree_found = btree.search(query)
        btree_search_time = (time.perf_counter() - start) * 1000000

        start = time.perf_counter()
        btree_docs = inverted_index.get(query, [])
        btree_retrieval_time = (time.perf_counter() - start) * 1000000

        rows.append({
            "Query Term": query,
            "BST Found": bst_found,
            "BST Search Time (micro sec)": round(bst_search_time, 3),
            "BST Retrieval Time (micro sec)": round(bst_retrieval_time, 3),
            "B-Tree Found": btree_found,
            "B-Tree Search Time (micro sec)": round(btree_search_time, 3),
            "B-Tree Retrieval Time (micro sec)": round(btree_retrieval_time, 3),
            "Retrieved Docs": sorted(set(bst_docs).union(set(btree_docs)))
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Step 6: Tolerant retrieval
# ---------------------------------------------------------
def wildcard_query(pattern, inverted_index):
    regex_pattern = "^" + pattern.replace("*", ".*") + "$"
    matching_terms = [term for term in inverted_index.keys() if re.match(regex_pattern, term)]
    result_docs = set()
    for term in matching_terms:
        result_docs.update(inverted_index[term])
    return matching_terms, sorted(result_docs)


def spelling_correction(query_term, dictionary_terms):
    query_term = query_term.lower().strip()
    if query_term in dictionary_terms:
        return query_term, 0

    best_term = None
    best_distance = 999
    for term in dictionary_terms:
        distance = edit_distance(query_term, term)
        if distance < best_distance:
            best_distance = distance
            best_term = term
    return best_term, best_distance


def create_kgram_index(dictionary_terms, k=3):
    kgram_index = defaultdict(set)
    for term in dictionary_terms:
        padded_term = "$" + term + "$"
        for i in range(len(padded_term) - k + 1):
            kgram = padded_term[i:i+k]
            kgram_index[kgram].add(term)
    return {key: sorted(value) for key, value in sorted(kgram_index.items())}


# ---------------------------------------------------------
# Step 7: Stemming vs Lemmatization comparison
# ---------------------------------------------------------
def compare_stemming_lemmatization(corpus, query, lower_case, remove_stop, hyphen_option):
    stemmed_corpus = preprocess_corpus(corpus, lower_case, remove_stop, hyphen_option, "Stemming")
    lemmatized_corpus = preprocess_corpus(corpus, lower_case, remove_stop, hyphen_option, "Lemmatization")

    stem_index = create_inverted_index(stemmed_corpus)
    lemma_index = create_inverted_index(lemmatized_corpus)

    stem_result, stem_tokens = boolean_and_query(query, stem_index, lower_case, remove_stop, hyphen_option, "Stemming")
    lemma_result, lemma_tokens = boolean_and_query(query, lemma_index, lower_case, remove_stop, hyphen_option, "Lemmatization")

    rows = [
        {"Method": "Stemming", "Query Tokens": stem_tokens, "Retrieved Documents": stem_result, "No. of Results": len(stem_result)},
        {"Method": "Lemmatization", "Query Tokens": lemma_tokens, "Retrieved Documents": lemma_result, "No. of Results": len(lemma_result)}
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Streamlit Front End
# ---------------------------------------------------------
st.set_page_config(page_title="End-to-End IR System", layout="wide")
st.title("End-to-End Information Retrieval System using Streamlit")
st.write("Upload documents, preprocess text, build indexes, run queries and compare retrieval techniques.")

st.sidebar.header("Dataset Upload")
uploaded_files = st.sidebar.file_uploader("Upload .txt document collection", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    corpus = read_uploaded_files(uploaded_files)
else:
    corpus = read_files(DATASET_DIR)

if len(corpus) == 0:
    st.error("No documents found. Upload .txt files or keep sample_docs folder with app.py.")
    st.stop()

st.sidebar.header("Preprocessing Options")
lower_case = st.sidebar.checkbox("Lowercasing", value=True)
remove_stop = st.sidebar.checkbox("Stop word removal", value=True)
hyphen_option = st.sidebar.selectbox("Hyphen handling", ["Split hyphenated words", "Join hyphenated words", "Keep hyphen as it is"])
method = st.sidebar.selectbox("Normalization", ["Lemmatization", "Stemming", "None"])

# Preprocess corpus and create indexes
preprocessed_corpus = preprocess_corpus(corpus, lower_case, remove_stop, hyphen_option, method)
inverted_index = create_inverted_index(preprocessed_corpus)
biword_index = create_biword_index(preprocessed_corpus)
positional_index = create_positional_inverted_index(preprocessed_corpus)
dictionary_terms = sorted(inverted_index.keys())
kgram_index = create_kgram_index(dictionary_terms, k=3)

# Tabs for complete end-to-end workflow
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Documents & Preprocessing",
    "2. Inverted Index",
    "3. Phrase Query",
    "4. BST vs B-Tree",
    "5. Tolerant Retrieval",
    "6. Inferences"
])

with tab1:
    st.header("Uploaded / Selected Documents")
    st.write("Total documents:", len(corpus))

    selected_doc = st.selectbox("Select document to view", list(corpus.keys()))
    st.subheader("Original Document")
    st.text_area("Original text", corpus[selected_doc], height=180)

    st.subheader("Preprocessed Output")
    st.write(preprocessed_corpus[selected_doc])

    st.subheader("Stemming vs Lemmatization Comparison")
    compare_query = st.text_input("Enter query for comparison", value="machine learning retrieval")
    comparison_df = compare_stemming_lemmatization(corpus, compare_query, lower_case, remove_stop, hyphen_option)
    st.dataframe(comparison_df, use_container_width=True)
    st.info("Inference: Stemming is faster and gives broader matching, but may reduce words to non-dictionary roots. Lemmatization keeps valid words and is usually more suitable for readable academic or domain documents.")

with tab2:
    st.header("Tokenization and Inverted Index Creation")
    st.write("Dictionary size:", len(dictionary_terms))
    st.subheader("Inverted Index Representation")
    st.dataframe(pd.DataFrame([{"Term": term, "Posting List": docs} for term, docs in inverted_index.items()]), use_container_width=True)

    st.subheader("Boolean AND Search")
    boolean_query = st.text_input("Enter Boolean AND style query", value="retrieval system")
    boolean_result, boolean_tokens = boolean_and_query(boolean_query, inverted_index, lower_case, remove_stop, hyphen_option, method)
    st.write("Processed query tokens:", boolean_tokens)
    st.write("Retrieved documents:", boolean_result)

with tab3:
    st.header("Phrase Query Processing")
    phrase_query = st.text_input("Enter phrase query", value="machine learning")

    biword_result, phrase_tokens, query_biwords = biword_phrase_query(phrase_query, biword_index, lower_case, remove_stop, hyphen_option, method)
    positional_result, positional_tokens = positional_phrase_query(phrase_query, positional_index, lower_case, remove_stop, hyphen_option, method)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Biword Index Representation")
        st.write("Query biwords:", query_biwords)
        st.dataframe(pd.DataFrame([{"Biword": term, "Posting List": docs} for term, docs in biword_index.items()]), use_container_width=True)
        st.write("Biword query result:", biword_result)

    with col2:
        st.subheader("Positional Index Representation")
        pos_rows = []
        for term, postings in positional_index.items():
            pos_rows.append({"Term": term, "Document Positions": postings})
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)
        st.write("Positional query result:", positional_result)

    st.subheader("Comparison")
    phrase_df = pd.DataFrame([
        {"Technique": "Biword Index", "Result": biword_result, "Comment": "Fast, but can produce false positives for longer phrases because it only checks adjacent word-pairs."},
        {"Technique": "Positional Index", "Result": positional_result, "Comment": "More accurate because it checks exact word positions in sequence."}
    ])
    st.dataframe(phrase_df, use_container_width=True)
    st.warning("False positive case: a long phrase may have all its biwords somewhere in a document, but not as one continuous phrase. Positional index avoids this by verifying consecutive positions.")

with tab4:
    st.header("Dictionary Search using Binary Search Tree and B-Tree")
    st.write("A dictionary of unique terms is created from the document collection.")
    multi_queries = st.text_area("Enter multiple terms, one per line", value="retrieval\nlearning\nindex\nstream\nwrongterm")
    query_list = multi_queries.split("\n")
    performance_df = compare_dictionary_search(query_list, dictionary_terms, inverted_index)
    st.dataframe(performance_df, use_container_width=True)
    st.info("Inference: B-Tree style sorted search usually gives stable logarithmic lookup. BST performance depends on tree balance; with a sorted insertion order, BST can become skewed and slower.")

with tab5:
    st.header("Tolerant Retrieval for Imperfect Queries")

    st.subheader("1. Wildcard Query")
    wildcard_pattern = st.text_input("Enter wildcard query", value="retriev*")
    matching_terms, wildcard_docs = wildcard_query(wildcard_pattern, inverted_index)
    st.write("Matching terms:", matching_terms)
    st.write("Retrieved documents:", wildcard_docs)

    st.subheader("2. Spelling / Edit Distance Correction")
    misspelled_term = st.text_input("Enter misspelled term", value="retrival")
    corrected_term, distance = spelling_correction(misspelled_term, dictionary_terms)
    st.write("Suggested correction:", corrected_term)
    st.write("Edit distance:", distance)
    st.write("Retrieved documents after correction:", inverted_index.get(corrected_term, []))

    st.subheader("3. K-Gram Index")
    st.write("Showing first 30 k-grams from the 3-gram index.")
    kgram_rows = []
    for kgram, terms in list(kgram_index.items())[:30]:
        kgram_rows.append({"K-Gram": kgram, "Terms": terms})
    st.dataframe(pd.DataFrame(kgram_rows), use_container_width=True)

with tab6:
    st.header("Final Inferences")
    st.markdown("""
    **Preprocessing:** Lowercasing and stop word removal reduce vocabulary size. Hyphen handling changes how terms like `real-time` and `real time` are matched.

    **Stemming vs Lemmatization:** Stemming gives aggressive matching but may create unreadable roots. Lemmatization preserves valid words and is generally more suitable for this selected document collection.

    **Phrase Query:** Biword index is simple and fast, but can give false positives for longer phrases. Positional index is more accurate because exact term positions are verified.

    **BST vs B-Tree:** BST search depends on balance. B-Tree/sorted search is more stable for dictionary lookup and is more suitable for larger dictionaries.

    **Tolerant Retrieval:** Wildcard search, edit distance correction and k-gram index improve retrieval when the user query is imperfect or partially known.
    """)
