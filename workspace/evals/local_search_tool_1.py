"""rLLM Tool wrapper for the local dense retrieval server."""

from __future__ import annotations

import os
from typing import Any

import requests

from rllm.tools.tool_base import Tool, ToolOutput


class LocalRetrievalTool(Tool):
    DESCRIPTION = "Search for information using a dense retrieval server with Wikipedia corpus"

    def __init__(
        self,
        name: str = "local_search",
        description: str | None = None,
        server_url: str | None = None,
        timeout: float = 3600.0,
        max_results: int = 10,
        summarize: int = 0,
    ) -> None:
        self.server_url = (
            server_url or os.environ.get("RETRIEVAL_SERVER_URL") or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.summarize = summarize
        super().__init__(name=name, description=description or self.DESCRIPTION)

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to retrieve relevant documents",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def forward(self, query: str, top_k: int | None = None, **_: Any) -> ToolOutput:
        try:
            results = self._retrieve(query=query, top_k=top_k or self.max_results, summarize=self.summarize)
            return ToolOutput(name=self.name or "local_search", output=results)
        except Exception as exc:
            return ToolOutput(
                name=self.name or "local_search",
                error=f"{type(exc).__name__} - {exc}",
            )

    def _retrieve(self, query: str, top_k: int, summarize: int) -> str:
        if summarize > 0:
            payload = {"query": query, "top_k": min(int(top_k), 50)}
            response = requests.post(
                f"{self.server_url}/retrieve", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                return "No relevant documents found for the query."
            format_results = self._format_results(results[:top_k])
            documents = []
            for result in results[:top_k]:
                content = result.get("content", "")
                if isinstance(content, dict):
                    content = content.get("contents", content.get("content", str(content)))
                content = str(content).strip()
                documents.append({"content": str(content)})

            payload_summary = {
                "documents": documents,
                "max_length": 512
            }
            response = requests.post(
                f"{self.server_url}/summarize", json=payload_summary, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            summary = data.get("summary", [])
            return summary
        else:
            payload = {"query": query, "top_k": min(int(top_k), 50)}
            response = requests.post(
                f"{self.server_url}/retrieve", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
        if not results:
            return "No relevant documents found for the query."
        return self._format_results(results[:top_k])

    @staticmethod
    def _format_results(results: list[dict[str, Any]]) -> str:
        formatted: list[str] = []
        for i, result in enumerate(results, 1):
            doc_id = result.get("id", f"doc_{i}")
            score = float(result.get("score", 0.0) or 0.0)
            content = result.get("content", "")
            if isinstance(content, dict):
                content = content.get("contents", content.get("content", str(content)))
            content = str(content).strip()
            if len(content) > 1200:
                content = content[:1200] + "..."
            formatted.append(
                f'<result id="{doc_id}" score="{score:.3f}">\n{content}\n</result>'
            )
        return "\n\n".join(formatted)


__all__ = ["LocalRetrievalTool"]
