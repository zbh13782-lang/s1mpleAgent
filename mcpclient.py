import asyncio
import inspect
from collections.abc import Callable
from typing import Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # pragma: no cover - handled at runtime
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]


class MCPClient:
    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.env = env

        self._stdio_ctx: Any | None = None
        self._stdio_streams: tuple[Any, Any] | None = None
        self._session: Any | None = None

        self._tools_by_name: dict[str, Any] = {}
        self._openai_tools_cache: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """Connect to MCP server and initialize a session."""
        self._ensure_mcp_available()
        if self._session is not None:
            return

        server_params = StdioServerParameters(  # type: ignore[operator]
            command=self.command,
            args=self.args,
            env=self.env,
        )

        self._stdio_ctx = stdio_client(server_params)  # type: ignore[operator]
        self._stdio_streams = await self._stdio_ctx.__aenter__()
        read_stream, write_stream = self._stdio_streams

        self._session = ClientSession(read_stream, write_stream)  # type: ignore[operator]
        await self._session.__aenter__()
        await self._session.initialize()
        await self.refresh_tools()

    async def close(self) -> None:
        session = self._session
        stdio_ctx = self._stdio_ctx

        self._session = None
        self._stdio_ctx = None
        self._stdio_streams = None

        if session is not None:
            await session.__aexit__(None, None, None)
        if stdio_ctx is not None:
            await stdio_ctx.__aexit__(None, None, None)

    async def refresh_tools(self) -> list[dict[str, Any]]:
        """Fetch latest MCP tools and rebuild OpenAI tool schemas."""
        session = self._require_session()
        listed = await session.list_tools()
        tools = getattr(listed, "tools", listed)

        self._tools_by_name = {}
        openai_tools: list[dict[str, Any]] = []

        for tool in tools:
            name = getattr(tool, "name", "")
            if not name:
                continue

            self._tools_by_name[name] = tool
            description = getattr(tool, "description", "") or ""
            input_schema = self._extract_input_schema(tool)

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": input_schema,
                    },
                }
            )

        self._openai_tools_cache = openai_tools
        return list(self._openai_tools_cache)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return list(self._openai_tools_cache)

    def get_tool_names(self) -> list[str]:
        return sorted(self._tools_by_name.keys())

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        session = self._require_session()
        if name not in self._tools_by_name:
            raise KeyError(f"Tool '{name}' not found in cached MCP tools")

        result = await session.call_tool(name=name, arguments=arguments or {})
        return self._normalize_tool_result(result)

    def build_tool_registry(self) -> dict[str, Callable[..., Any]]:
        registry: dict[str, Callable[..., Any]] = {}
        for name in self._tools_by_name:
            registry[name] = self._make_sync_tool_caller(name)
        return registry

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _make_sync_tool_caller(self, name: str) -> Callable[..., Any]:
        def _caller(**kwargs: Any) -> Any:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.call_tool(name, kwargs))
            raise RuntimeError(
                "build_tool_registry() sync wrapper cannot run inside an existing event loop. "
                "Use await call_tool(...) in async context."
            )

        return _caller

    def _extract_input_schema(self, tool: Any) -> dict[str, Any]:
        input_schema = getattr(tool, "inputSchema", None)
        if input_schema is None:
            input_schema = getattr(tool, "input_schema", None)

        if input_schema is None:
            return {"type": "object", "properties": {}}

        if isinstance(input_schema, dict):
            return input_schema

        if hasattr(input_schema, "model_dump") and callable(input_schema.model_dump):
            return input_schema.model_dump()  # type: ignore[no-any-return]

        if hasattr(input_schema, "dict") and callable(input_schema.dict):
            return input_schema.dict()  # type: ignore[no-any-return]

        if inspect.isclass(input_schema) and hasattr(input_schema, "model_json_schema"):
            return input_schema.model_json_schema()  # type: ignore[no-any-return]

        return {"type": "object", "properties": {}}

    def _normalize_tool_result(self, result: Any) -> Any:
        if result is None:
            return ""

        content = getattr(result, "content", None)
        if content is None:
            if hasattr(result, "model_dump") and callable(result.model_dump):
                return result.model_dump()  # type: ignore[no-any-return]
            return result

        normalized_parts: list[Any] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                normalized_parts.append(text)
                continue

            if hasattr(item, "model_dump") and callable(item.model_dump):
                normalized_parts.append(item.model_dump())
                continue

            normalized_parts.append(item)

        if len(normalized_parts) == 1 and isinstance(normalized_parts[0], str):
            return normalized_parts[0]
        return normalized_parts

    def _require_session(self) -> Any:
        if self._session is None:
            raise RuntimeError("MCP session not connected. Call connect() first.")
        return self._session

    def _ensure_mcp_available(self) -> None:
        if (
            ClientSession is None
            or StdioServerParameters is None
            or stdio_client is None
        ):
            raise RuntimeError(
                "Missing MCP SDK. Install dependency 'mcp' first, then retry."
            )
