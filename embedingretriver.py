import json
import os
import uuid
from pathlib import Path
from typing import Any

import dotenv
from openai import OpenAI


dotenv.load_dotenv()


class EmbedingRetriever:
    """Simple in-memory embedding retriever with OpenAI embeddings."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        resolved_api_key = (
            api_key
            or os.getenv("EMBEDDING_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is not provided, please check settings.")

        resolved_base_url = (
            base_url
            or os.getenv("EMBEDDING_OPENAI_API_BASE")
            or os.getenv("OPENAI_API_BASE")
        )

        self.model = model
        self.request_timeout = request_timeout
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

        self._docs: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()

    def add_document(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Document text cannot be empty.")

        embedding = self._embed_one(text)
        doc_id = document_id or str(uuid.uuid4())

        self._docs.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "embedding": embedding,
            }
        )
        return doc_id

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        document_ids: list[str] | None = None,
        batch_size: int = 64,
    ) -> list[str]:
        cleaned_texts = [(t or "").strip() for t in texts]
        if not cleaned_texts:
            return []

        if any(not t for t in cleaned_texts):
            raise ValueError("Document text list contains empty item(s).")

        if metadatas is not None and len(metadatas) != len(cleaned_texts):
            raise ValueError("metadatas length must match texts length.")
        if document_ids is not None and len(document_ids) != len(cleaned_texts):
            raise ValueError("document_ids length must match texts length.")

        all_embeddings: list[list[float]] = []
        for i in range(0, len(cleaned_texts), max(1, batch_size)):
            batch = cleaned_texts[i : i + max(1, batch_size)]
            all_embeddings.extend(self._embed_many(batch))

        ids: list[str] = []
        for idx, text in enumerate(cleaned_texts):
            doc_id = document_ids[idx] if document_ids else str(uuid.uuid4())
            meta = metadatas[idx] if metadatas else {}
            self._docs.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": meta,
                    "embedding": all_embeddings[idx],
                }
            )
            ids.append(doc_id)

        return ids

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not self._docs:
            return []

        query_embedding = self._embed_one(query)

        scored: list[dict[str, Any]] = []
        for doc in self._docs:
            score = self._cosine_similarity(query_embedding, doc["embedding"])
            if min_score is None or score >= min_score:
                scored.append(
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "score": score,
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def save(self, file_path: str) -> None:
        payload = {
            "model": self.model,
            "documents": self._docs,
        }
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, file_path: str, merge: bool = False) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {file_path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        docs = payload.get("documents", [])
        if not isinstance(docs, list):
            raise ValueError("Invalid index format: 'documents' must be a list.")

        if not merge:
            self._docs = []

        for item in docs:
            if not isinstance(item, dict):
                continue
            if "id" not in item or "text" not in item or "embedding" not in item:
                continue
            self._docs.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "embedding": item["embedding"],
                }
            )

    def _embed_one(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            timeout=self.request_timeout,
        )
        return response.data[0].embedding

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            timeout=self.request_timeout,
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_vec1 = sum(a * a for a in vec1) ** 0.5
        norm_vec2 = sum(b * b for b in vec2) ** 0.5
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)
