class vectorstore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_document(self, document, embedding):
        self.documents.append(document)
        self.embeddings.append(embedding)

    def search(self, query_embedding, top_k=5):
        # 计算查询向量与文档向量的相似度
        similarities = [
            self.cosine_similarity(query_embedding, emb) for emb in self.embeddings
        ]
        # 获取相似度最高的top_k个文档
        top_indices = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )[:top_k]
        return [self.documents[i] for i in top_indices]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_vec1 = sum(a * a for a in vec1) ** 0.5
        norm_vec2 = sum(b * b for b in vec2) ** 0.5
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)
