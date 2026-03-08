import asyncio
from typing import Any, Iterator

from ChatModelAI import ChatModelAI
from embedingretriver import EmbedingRetriever
from mcpclient import MCPClient


class Agent:
    """High-level agent orchestrating LLM chat, MCP tools, and RAG retrieval."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are a helpful AI assistant.",
        context: str = "",
        mcp_command: str | None = None,
        mcp_args: list[str] | None = None,
        mcp_env: dict[str, str] | None = None,
        enable_rag: bool = False,
        embedding_model: str = "text-embedding-3-small",
        rag_top_k: int = 3,
        request_timeout: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.base_context = context
        self.rag_top_k = rag_top_k
        self.request_timeout = request_timeout

        self.mcp_client: MCPClient | None = None
        if mcp_command:
            self.mcp_client = MCPClient(command=mcp_command, args=mcp_args, env=mcp_env)

        self.retriever: EmbedingRetriever | None = None
        if enable_rag:
            self.retriever = EmbedingRetriever(
                model=embedding_model,
                request_timeout=request_timeout,
            )

        self.chat_model: ChatModelAI | None = None
        self._started = False

    async def astart(self) -> None:
        """Initialize optional MCP and create ChatModelAI instance."""
        tools: list[dict[str, Any]] = []
        tool_registry: dict[str, Any] = {}

        if self.mcp_client is not None:
            await self.mcp_client.connect()
            tools = self.mcp_client.get_openai_tools()
            tool_registry = self.mcp_client.build_tool_registry()

        self.chat_model = ChatModelAI(
            modelname=self.model_name,
            tools=tools,
            systemprompt=self.system_prompt,
            context=self.base_context,
            tool_registry=tool_registry,
            request_timeout=self.request_timeout,
        )
        self._started = True

    async def aclose(self) -> None:
        """Gracefully release resources."""
        if self.mcp_client is not None:
            await self.mcp_client.close()
        self._started = False

    def start(self) -> None:
        """Sync wrapper for astart()."""
        self._run_async(self.astart())

    def close(self) -> None:
        """Sync wrapper for aclose()."""
        self._run_async(self.aclose())

    def refresh_mcp_tools(self) -> None:
        """Refresh MCP tools and rebind tool schemas into ChatModelAI."""
        if self.mcp_client is None:
            return
        self._run_async(self._arefresh_mcp_tools())

    async def _arefresh_mcp_tools(self) -> None:
        if self.mcp_client is None:
            return
        await self.mcp_client.refresh_tools()
        if self.chat_model is None:
            return
        self.chat_model.tools = self.mcp_client.get_openai_tools()
        self.chat_model.tool_registry = self.mcp_client.build_tool_registry()

    def ingest_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        document_ids: list[str] | None = None,
        batch_size: int = 64,
    ) -> list[str]:
        """Add documents into the retriever index."""
        retriever = self._require_retriever()
        return retriever.add_documents(
            texts=texts,
            metadatas=metadatas,
            document_ids=document_ids,
            batch_size=batch_size,
        )

    def save_index(self, file_path: str) -> None:
        retriever = self._require_retriever()
        retriever.save(file_path)

    def load_index(self, file_path: str, merge: bool = False) -> None:
        retriever = self._require_retriever()
        retriever.load(file_path, merge=merge)

    def stream_chat(
        self, user_input: str, use_rag: bool = True
    ) -> Iterator[dict[str, Any]]:
        """Stream chat events from ChatModelAI, optionally with RAG context."""
        self._ensure_started()
        assert self.chat_model is not None

        final_user_input = user_input
        if use_rag and self.retriever is not None and self.retriever.size > 0:
            final_user_input = self._inject_retrieval_context(user_input)

        yield from self.chat_model.StreamChat(final_user_input)

    def get_status(self) -> dict[str, Any]:
        return {
            "started": self._started,
            "model": self.model_name,
            "mcp_enabled": self.mcp_client is not None,
            "mcp_tools": self.mcp_client.get_tool_names() if self.mcp_client else [],
            "rag_enabled": self.retriever is not None,
            "rag_documents": self.retriever.size if self.retriever else 0,
        }

    def _inject_retrieval_context(self, user_input: str) -> str:
        assert self.retriever is not None
        retrieved = self.retriever.retrieve(
            query=user_input, top_k=max(1, self.rag_top_k)
        )
        if not retrieved:
            return user_input

        context_blocks: list[str] = []
        for idx, item in enumerate(retrieved, start=1):
            context_blocks.append(
                f"[doc-{idx}] score={item['score']:.4f}\n{item['text']}"
            )

        retrieved_context = "\n\n".join(context_blocks)
        return (
            "请优先根据以下检索上下文回答；若上下文不足请明确说明，再给出通用答案。\n\n"
            f"检索上下文:\n{retrieved_context}\n\n"
            f"用户问题: {user_input}"
        )

    def _ensure_started(self) -> None:
        if self._started:
            return
        self.start()

    def _require_retriever(self) -> EmbedingRetriever:
        if self.retriever is None:
            raise RuntimeError(
                "RAG retriever is disabled. Set enable_rag=True in Agent."
            )
        return self.retriever

    @staticmethod
    def _run_async(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "Cannot run sync wrapper inside an existing event loop. "
            "Use async methods (astart/aclose) in async context."
        )
