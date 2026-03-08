import argparse
import os
from pathlib import Path

from Agent import Agent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run s1mpleAgent in CLI mode.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model name")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
        help="System prompt for the assistant",
    )
    parser.add_argument("--context", default="", help="Extra context for the assistant")

    parser.add_argument(
        "--enable-rag", action="store_true", help="Enable embedding retriever"
    )
    parser.add_argument("--rag-index", default="", help="RAG index file path to load")

    parser.add_argument("--mcp-command", default="", help="MCP server command")
    parser.add_argument(
        "--mcp-args",
        nargs="*",
        default=[],
        help="Arguments for MCP server command",
    )
    return parser


def _print_help_banner(enable_rag: bool) -> None:
    print("s1mpleAgent CLI started. Type 'exit' or 'quit' to stop.")
    if enable_rag:
        print("RAG commands: /ingest <text>, /save <index_path>")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing. Please set it in environment or .env file.")
        return 1

    agent = Agent(
        model_name=args.model,
        system_prompt=args.system_prompt,
        context=args.context,
        mcp_command=args.mcp_command or None,
        mcp_args=args.mcp_args,
        enable_rag=args.enable_rag,
    )

    try:
        agent.start()

        if args.enable_rag and args.rag_index:
            rag_path = Path(args.rag_index)
            if rag_path.exists():
                agent.load_index(str(rag_path), merge=False)
                print(f"Loaded RAG index: {rag_path}")
            else:
                print(f"RAG index file not found, skip loading: {rag_path}")

        _print_help_banner(args.enable_rag)

        while True:
            user_input = input("You> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            if args.enable_rag and user_input.startswith("/ingest "):
                text = user_input[len("/ingest ") :].strip()
                if not text:
                    print("Usage: /ingest <text>")
                    continue
                doc_id = agent.ingest_documents([text])[0]
                print(f"Ingested doc id: {doc_id}")
                continue

            if args.enable_rag and user_input.startswith("/save "):
                path = user_input[len("/save ") :].strip()
                if not path:
                    print("Usage: /save <index_path>")
                    continue
                agent.save_index(path)
                print(f"Saved index to: {path}")
                continue

            print("AI> ", end="", flush=True)
            for event in agent.stream_chat(user_input, use_rag=args.enable_rag):
                event_type = event.get("type")
                if event_type == "delta":
                    print(event.get("content", ""), end="", flush=True)
                elif event_type == "tool_calls":
                    print(f"\n[tool_calls] {event.get('tool_calls', [])}")
                elif event_type == "tool_results":
                    print(f"\n[tool_results] {event.get('tool_results', [])}")
            print()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        agent.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
