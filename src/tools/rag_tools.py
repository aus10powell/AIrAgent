"""RAG (Retrieval-Augmented Generation) tools for AirAgent."""

import spacy
from typing import List
from langchain_core.tools import Tool
from langchain_community.llms import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import OLLAMA_MODEL, DEFAULT_TEMPERATURE, DEFAULT_CHUNK_SIZE

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize LLM for RAG
llm = Ollama(model=OLLAMA_MODEL, temperature=DEFAULT_TEMPERATURE)


def simple_rag_with_ollama(
    query: str, corpus: str, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> str:
    """
    Lightweight RAG using TF-IDF and Ollama's generate endpoint with spaCy-based chunking.

    Args:
        query (str): Question to answer
        corpus (str): Text corpus to search through
        chunk_size (int): Maximum words per chunk

    Returns:
        str: Answer to the query based on the corpus
    """
    # Normalize corpus
    if isinstance(corpus, list):
        corpus = " ".join(corpus)

    doc = nlp(corpus)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Using sentence-based chunking to split text into manageable pieces while preserving semantic meaning.
    # Each chunk will contain complete sentences up to the specified chunk_size limit.
    chunks, current_chunk, current_len = [], [], 0
    for sentence in sentences:
        sent_len = len(sentence.split())
        # If the current chunk is too large, add it to the chunks list and start a new chunk
        if current_len + sent_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [], 0
        current_chunk.append(sentence)
        current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if not chunks:
        return "No content available to search through."

    # TF-IDF retrieval
    try:
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(chunks)
        query_vector = vectorizer.transform([query])
        best_doc = chunks[cosine_similarity(query_vector, doc_vectors).argmax()]
    except Exception as e:
        return f"Error during retrieval: {str(e)}"

    # Generate API payload (using prompt instead of messages)
    prompt = f"Context: {best_doc}\n\nQuestion: {query}"

    # Generate the response using Ollama
    try:
        response = llm.generate(prompts=[prompt])

        # Extract text from the response
        if hasattr(response, "generations") and response.generations:
            if response.generations[0] and hasattr(response.generations[0][0], "text"):
                return response.generations[0][0].text
            else:
                return "[Error] Couldn't extract text from response"
        else:
            return "[Error] No generations in response"
    except Exception as e:
        return f"[Error] {str(e)}"


def rag_wrapper(input_dict: dict) -> str:
    """
    Wrapper function for RAG to handle dictionary input.

    Args:
        input_dict (dict): Dictionary with 'query' and 'corpus' keys

    Returns:
        str: RAG response
    """
    query = input_dict.get("query", "")
    corpus = input_dict.get("corpus", "")

    if not query:
        return "No query provided."
    if not corpus:
        return "No corpus provided to search through."

    return simple_rag_with_ollama(query, corpus)


# Create the LangChain Tool
RAGTool = Tool(
    name="answer_listing_questions",
    func=rag_wrapper,
    description="Answers questions about specific listings using a RAG approach. Input should be a dictionary with 'query' and 'corpus' keys.",
)
