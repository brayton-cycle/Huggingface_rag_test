import logging
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_key:
    logger.error("Hugging Face API key not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    exit(1)


llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.7, "max_length": 512},
    huggingfacehub_api_token=hf_api_key
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def initialize_rag_model():
    """Initializes the RAG retriever model."""
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
    return tokenizer, retriever, model


def load_and_index_documents():
    """Loads documents, splits text, and indexes embeddings using FAISS."""
    if not os.path.exists("sample_docs.txt"):
        logger.error("File sample_docs.txt not found.")
        exit(1)
    
    loader = TextLoader("sample_docs.txt")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    
    db = FAISS.from_documents(texts, embeddings)
    return db


def perform_rag_query(query, db, retriever, tokenizer, model):
    """Performs RAG-based retrieval and response generation."""
    retrieved_docs = db.similarity_search(query)
    
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    context = retriever.retrieve(query, n_docs=5) 
    generated_ids = model.generate(input_ids, context_input_ids=context.input_ids)
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, retrieved_docs


def evaluate_rag(query, retrieved_docs, response):
    """Evaluates the response using fuzzy matching against retrieved docs."""
    scores = [fuzz.partial_ratio(doc.page_content, response) for doc in retrieved_docs]
    avg_score = sum(scores) / len(scores) if scores else 0

    logger.info(f"Query: {query}")
    logger.info(f"Generated Response: {response}")
    logger.info(f"Evaluation Score: {avg_score:.2f}")

    return avg_score


def main():
    """Runs the RAG pipeline and evaluates it."""
    db = load_and_index_documents()
    tokenizer, retriever, model = initialize_rag_model()
    
    query = "What is Retrieval-Augmented Generation?"
    response, retrieved_docs = perform_rag_query(query, db, retriever, tokenizer, model)
    
    evaluate_rag(query, retrieved_docs, response)


if __name__ == "__main__":
    main()
