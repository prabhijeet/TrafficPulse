# llm/rag.py

'''
Day 8 — RAG Knowledge Base using ChromaDB.
Stores the traffic policy document as vector embeddings so the
LLM assistant can retrieve relevant policy sections before answering.

RAG = Retrieval-Augmented Generation.
Instead of putting the entire policy document in every prompt
(expensive and slow), we:
  1. Split the document into chunks
  2. Convert each chunk to a vector embedding
  3. Store in ChromaDB (local vector database)
  4. At query time, find the most relevant chunks
  5. Inject only those chunks into the LLM prompt

This grounds the LLM's answers in written policy — when asked
"why did the agent give INT_2 priority?" it can cite the actual
policy rule, not just reason from training data.

Two RAG uses in this project:
  P12 style (this file): grounds the dialogue chatbot in policy rules
  P2 style (future):     injects domain knowledge to help generate
                         digital twin model components automatically
'''

import os
import chromadb
from chromadb.utils import embedding_functions
from simulation.config import BASE_DIR

# paths
POLICY_FILE   = os.path.join(BASE_DIR, 'docs', 'traffic_policy.md')
CHROMA_DIR    = os.path.join(BASE_DIR, 'data', 'chroma_db')

# ChromaDB collection name
COLLECTION_NAME = 'traffic_policy'

# how many relevant chunks to retrieve per query
TOP_K = 3


def load_policy_document():
    '''
    Reads the traffic policy markdown file and returns its content.
    '''
    with open(POLICY_FILE, 'r') as f:
        return f.read()


def chunk_document(text, chunk_size=500, overlap=50):
    '''
    Splits the policy document into overlapping chunks.
    Smaller chunks = more precise retrieval.
    Overlap ensures context is not lost at chunk boundaries.

    chunk_size: approximate characters per chunk
    overlap   : characters shared between consecutive chunks
    '''
    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size

        # try to break at a newline for cleaner chunks
        if end < len(text):
            newline = text.rfind('\n', start, end)
            if newline > start + chunk_size // 2:
                end = newline

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def build_rag_index():
    '''
    Builds the ChromaDB vector index from the policy document.
    Call this once to create the index — subsequent calls
    check if it already exists and skip rebuilding.

    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2)
    which runs locally — no external API needed.
    '''
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # create persistent ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # check if collection already exists
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f'RAG index already exists — {COLLECTION_NAME}')
        collection = client.get_collection(COLLECTION_NAME)
        print(f'Documents in index: {collection.count()}')
        return collection

    # create new collection with default embedding function
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(
        name               = COLLECTION_NAME,
        embedding_function = ef,
        metadata           = {'description': 'TrafficPulse signal policy'}
    )

    # load and chunk the policy document
    print('Loading policy document...')
    policy_text = load_policy_document()
    chunks      = chunk_document(policy_text)

    print(f'Document split into {len(chunks)} chunks')
    print('Building vector embeddings...')

    # add chunks to ChromaDB
    collection.add(
        documents = chunks,
        ids       = [f'chunk_{i}' for i in range(len(chunks))]
    )

    print(f'RAG index built — {collection.count()} chunks indexed')
    return collection


def get_collection():
    '''
    Returns the ChromaDB collection.
    Builds the index if it doesn't exist yet.
    '''
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME not in existing:
        return build_rag_index()

    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(
        name               = COLLECTION_NAME,
        embedding_function = ef
    )


def retrieve(query, top_k=TOP_K):
    '''
    Retrieves the most relevant policy chunks for a given query.
    Returns a single string with all relevant chunks joined together.

    Args:
        query : the user's question or topic to search for
        top_k : number of chunks to retrieve (default 3)

    Returns:
        string of relevant policy sections
    '''
    collection = get_collection()

    results = collection.query(
        query_texts = [query],
        n_results   = min(top_k, collection.count())
    )

    chunks = results['documents'][0]
    return '\n\n---\n\n'.join(chunks)


def retrieve_with_metadata(query, top_k=TOP_K):
    '''
    Same as retrieve() but also returns distance scores.
    Lower distance = more relevant.
    Useful for debugging retrieval quality.
    '''
    collection = get_collection()

    results = collection.query(
        query_texts = [query],
        n_results   = min(top_k, collection.count()),
        include     = ['documents', 'distances']
    )

    chunks    = results['documents'][0]
    distances = results['distances'][0]

    return list(zip(chunks, distances))


def test_retrieval():
    '''
    Tests the RAG system with a few sample queries.
    Confirms that relevant policy sections are being retrieved.
    '''
    test_queries = [
        'Why does INT_2 have priority over other intersections?',
        'What is the minimum green time and why?',
        'How does the agent handle rush hour traffic?',
        'What happens when all intersections are equally congested?',
        'What is the maximum wait time a car can experience?',
    ]

    print('Testing RAG retrieval...')
    print()

    for query in test_queries:
        print(f'Query: {query}')
        results = retrieve_with_metadata(query, top_k=1)
        chunk, distance = results[0]
        print(f'Distance: {distance:.4f}')
        print(f'Retrieved: {chunk[:200]}...')
        print()


if __name__ == '__main__':
    print('TrafficPulse — Day 8 RAG Knowledge Base')
    print()

    # build the index
    build_rag_index()
    print()

    # test retrieval
    test_retrieval()

    print('RAG system ready.')
    print('Import retrieve() in assistant_api.py to use it.')