from sentence_transformers import SentenceTransformer
import numpy as np


class Retriever:
    def __init__(self) -> None:
        pass

    def get_top_k_similar_chunks(self, df, query_embedding, k):
        similarities = []

        for index, row in df.iterrows():
            similarity = self.cosine_similarity(query_embedding, row["embedding"])
            similarities.append((row["chunk"], similarity))

        top_k_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

        return top_k_similarities

    def get_sentence_embedding(self, sentence: str, model_name: str = "all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_name)
        return np.asarray(model.encode(sentence))

    def cosine_similarity(self, embedding_1, embedding_2):
        dot_product = np.dot(embedding_1, embedding_2)
        norm_1 = np.linalg.norm(embedding_1)
        norm_2 = np.linalg.norm(embedding_2)
        return dot_product / (norm_1 * norm_2)





# retriever = Retriever()
# query_embedding = retriever.get_sentence_embedding(sentence="I've been feeling really tired and thirsty all the time. What could be wrong with me?")

