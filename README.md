# Retrieval-Augmented Generation (RAG) Evaluation

## Overview

This project demonstrates a basic RAG evaluation process. The Retrieval-Augmented Generation (RAG) approach enhances language models by retrieving relevant information before generating responses. 

## Types of RAG Evaluation

1. **Faithfulness Evaluation**  
   - Measures if the generated response is factually correct based on retrieved documents.  

2. **Relevance Evaluation**  
   - Assesses if the retrieved documents are relevant to the query.  

3. **Fluency Evaluation**  
   - Checks if the response is grammatically and semantically correct.  

4. **Similarity-Based Evaluation**  
   - Compares the generated response with retrieved documents using string similarity metrics.  

## Chosen Evaluation Method

For this project, we use **Similarity-Based Evaluation** with fuzzy string matching (`fuzzywuzzy`) to measure how closely the generated response matches retrieved documents. This provides a simple and effective scoring method.

## Requirements

- Python 3.8+
- Install dependencies using:

  ```sh
  pip install -r requirements.txt
