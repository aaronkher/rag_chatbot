import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
from rag.document_vectorizer import DocumentVectorizer
from retriever.retriever import Retriever



if __name__ == "__main__":
    txt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag/document_1.txt'))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer = DocumentVectorizer(embedding_model=model)
    df = vectorizer.create_dataframe(txt_path)

    retriever = Retriever()
    query_embedding = retriever.get_sentence_embedding(sentence="Are Non-communicable diseases infectious?")
    top_k_chunks = retriever.get_top_k_similar_chunks(df, query_embedding, k=2)
    
    res = ""
    for chunk in top_k_chunks:
        text,similarity = chunk[0], chunk[1]
        if similarity > 0.2:
            res += text

    print(res)