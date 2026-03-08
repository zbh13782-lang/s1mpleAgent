import json
import os
import time
from typing import Any, Callable, Iterator

import dotenv
from openai import OpenAI

# COPILOT_EDIT_MARK: 2026-03-07 ChatModelAI updated
dotenv.load_dotenv()


class ChatModelAI:
    def __init__(
        self,
        modelname: str,
        tools: list[dict[str, Any]] | None,
        systemprompt: str,
        context: str,
        tool_registry: dict[str, Callable[..., Any]] | None = None,
        max_history_messages: int = 20,
        request_timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not provided,Please check settings.")

        base_url = os.getenv("OPENAI_API_BASE")

        self.modelname = modelname
        self.tools = tools or []
        self.tool_registry = tool_registry or {}
        self.max_history_messages = max_history_messages
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.llm = OpenAI(api_key=api_key, base_url=base_url)

        self.messages: list[dict[str, Any]] = []
        if systemprompt:
            self.messages.append({"role": "system", "content": systemprompt})
        if context:
            self.messages.append({"role": "system", "content": context})

    # 返回事件流
    def StreamChat(self, user_input: str) -> Iterator[dict[str, Any]]:
        if user_input:
            self.messages.append({"role": "user", "content": user_input})

        self._trim_messages()
        content = ""
        tool_calls: list[dict[str, Any]] = []
        for event in self._iter_stream_completion_events():
            if event["type"] == "delta":
                yield event
                continue
            content = event["content"]
            tool_calls = event["tool_calls"]

        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        self.messages.append(assistant_message)

        tool_results: list[dict[str, Any]] = []
        if tool_calls:
            yield {"type": "tool_calls", "tool_calls": tool_calls}
            tool_results = self._execute_tool_calls(tool_calls)
            yield {"type": "tool_results", "tool_results": tool_results}

            final_content = ""
            for event in self._iter_stream_completion_events():
                if event["type"] == "delta":
                    yield event
                    continue
                final_content = event["content"]

            self.messages.append({"role": "assistant", "content": final_content})
            yield {
                "type": "done",
                "content": final_content,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
            }
            return

        yield {
            "type": "done",
            "content": content,
            "tool_calls": [],
            "tool_results": [],
        }

    def _iter_stream_completion_events(self) -> Iterator[dict[str, Any]]:
        response = self._request_with_retry(stream=True)
        content_parts: list[str] = []
        tool_call_state: dict[int, dict[str, Any]] = {}

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                content_parts.append(delta.content)
                yield {"type": "delta", "content": delta.content}

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_state:
                        tool_call_state[idx] = {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    if tc.id:
                        tool_call_state[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_call_state[idx]["function"]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_call_state[idx]["function"]["arguments"] += (
                            tc.function.arguments
                        )

        tool_calls = [tool_call_state[i] for i in sorted(tool_call_state.keys())]
        yield {
            "type": "complete",
            "content": "".join(content_parts),
            "tool_calls": tool_calls,
        }

    def _stream_completion(self) -> tuple[str, list[dict[str, Any]]]:
        content = ""
        tool_calls: list[dict[str, Any]] = []
        for event in self._iter_stream_completion_events():
            if event["type"] != "delta":
                content = event["content"]
                tool_calls = event["tool_calls"]
        return content, tool_calls

    def _completion_once(self) -> str:
        response = self._request_with_retry(stream=False)
        message = response.choices[0].message
        return message.content or ""

    def _request_with_retry(self, stream: bool):
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.llm.chat.completions.create(
                    model=self.modelname,
                    messages=self.messages,
                    tools=self.tools or None,
                    stream=stream,
                    timeout=self.request_timeout,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"Chat completion failed after retries: {last_error}")

    def _execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in tool_calls:
            name = call["function"]["name"]
            args_raw = call["function"]["arguments"] or "{}"

            try:
                parsed_args = json.loads(args_raw)
            except json.JSONDecodeError:
                parsed_args = {"raw_arguments": args_raw}

            fn = self.tool_registry.get(name)
            if not fn:
                output = f"Tool '{name}' is not registered."
            else:
                try:
                    if isinstance(parsed_args, dict):
                        result = fn(**parsed_args)
                    else:
                        result = fn(parsed_args)
                    output = (
                        result
                        if isinstance(result, str)
                        else json.dumps(result, ensure_ascii=False)
                    )
                except Exception as exc:
                    output = f"Tool '{name}' execution error: {exc}"

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": output,
                }
            )
            results.append({"tool": name, "output": output})

        return results

    # 修剪上下文避免太长了
    def _trim_messages(self) -> None:
        if self.max_history_messages <= 0:
            return

        system_messages = [m for m in self.messages if m.get("role") == "system"]
        non_system_messages = [m for m in self.messages if m.get("role") != "system"]

        if len(non_system_messages) > self.max_history_messages:
            non_system_messages = non_system_messages[-self.max_history_messages :]

        self.messages = system_messages + non_system_messages
