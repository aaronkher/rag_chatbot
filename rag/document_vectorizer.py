import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

nltk.download('punkt')

class DocumentVectorizer:
    def __init__(self, embedding_model, chunk_size=200):
        self.chunk_size = chunk_size
        self.model = embedding_model

    def extract_text_from_txt(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def chunk_text(self, text):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(word_tokenize(sentence))
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def get_sentence_embedding(self, sentence):
        return np.asarray(self.model.encode(sentence))

    def vectorize_chunks(self, chunks):
        return [(chunk, self.get_sentence_embedding(chunk)) for chunk in chunks]

    def vectorize(self, txt_path):
        text = self.extract_text_from_txt(txt_path)
        chunks = self.chunk_text(text)
        vectors = self.vectorize_chunks(chunks)
        return vectors

    def create_dataframe(self, txt_path):
        vectors = self.vectorize(txt_path)
        data = {
            'chunk': [chunk for chunk, vector in vectors],
            'embedding': [vector for chunk, vector in vectors]
        }
        df = pd.DataFrame(data)
        return df

if __name__ == "__main__":
    pass
    # txt_path = 'rag/document_1.txt'
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # vectorizer = DocumentVectorizer(embedding_model=model)
    # df = vectorizer.create_dataframe(txt_path)
    # print(df)
    