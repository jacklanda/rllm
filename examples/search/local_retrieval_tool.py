#!/usr/bin/env python3

import os
import logging
from typing import Any, Optional

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class LocalRetrievalTool(Tool):
    """
    A tool for dense search using the local retrieval server.

    This tool connects to a locally running dense retrieval server (launched via retrieval_launch.sh)
    and performs dense retrieval using E5 embeddings on the indexed Wikipedia corpus.
    """

    NAME = "local_search"
    DESCRIPTION = "Search for information using a dense retrieval server with Wikipedia corpus"

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        server_url: str = None,
        timeout: float = 3600.0,
        max_results: int = 10,
    ):
        """
        Initialize the Local Retrieval Tool.

        Args:
            name: Tool name
            description: Tool description
            server_url: URL of the local retrieval server (if None, checks RETRIEVAL_SERVER_URL env var)
            timeout: Request timeout in seconds
            max_results: Maximum number of results to return
        """
        # Use environment variable if server_url not provided
        if server_url is None:
            server_url = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000")

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30),
        )

        # Suppress httpx INFO logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        super().__init__(name=name, description=description)

        # Test server connection
        # self._test_connection()

    def _test_connection(self):
        """Test connection to the retrieval server."""
        try:
            response = self.client.get(f"{self.server_url}/health")
            if response.status_code == 200:
                # logger.info(f"Successfully connected to retrieval server at {self.server_url}")
                pass
            else:
                logger.warning(f"Retrieval server returned status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to retrieval server: {e}")

    @property
    def json(self):
        """Return tool JSON schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query to retrieve relevant documents"},
                        "top_k": {"type": "integer", "description": f"Number of results to return (default: {self.max_results})", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                },
            },
        }

    # P0-1: minimum useful passage size. step-0 evals showed 22-51% of
    # web_search responses came back with < 20 words (just token lists
    # like "Brob-Bron | Denys Fisher"), which killed multi-hop tasks
    # (musique 9.9%, 2wiki 40%). We now prefer the long ``document``
    # field over ``chunk_text`` fragments and drop results below a
    # minimum word threshold.
    _MIN_DOC_WORDS = 25
    _summarize_disabled_reason: str | None = None

    def _extract_doc_text(self, result: dict[str, Any]) -> str | None:
        """Pull the longest-available textual field out of a retrieval result."""
        candidates = []
        doc = result.get("document")
        if isinstance(doc, str) and doc.strip():
            candidates.append(doc)
        content = result.get("content")
        if isinstance(content, dict):
            for key in ("original_text", "chunk_text", "text"):
                v = content.get(key)
                if isinstance(v, str) and v.strip():
                    candidates.append(v)
        elif isinstance(content, str) and content.strip():
            candidates.append(content)
        for key in ("chunk_text", "text", "passage"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                candidates.append(v)
        if not candidates:
            return None
        # Prefer the longest candidate — passages beat ``content.chunk_text``
        # tokens which are often 1-3 entity strings.
        return max(candidates, key=lambda s: len(s.split()))

    def _format_search_results(self, results: list[dict[str, Any]], query: Optional[str] = None) -> list[str]:
        """Format search results for LLM consumption."""
        if not results:
            return ["No relevant documents found."]

        documents: list[str] = []
        skipped_short = 0
        for result in results:
            content = self._extract_doc_text(result)
            if not content:
                continue
            if len(content.split()) < self._MIN_DOC_WORDS:
                skipped_short += 1
                continue
            documents.append(content)
            if len(documents) >= self.max_results:
                break

        if not documents:
            # Fall back: take top candidate even if short so the model
            # still sees something, but tag it so the rollout knows.
            for result in results[: self.max_results]:
                content = self._extract_doc_text(result)
                if content:
                    documents.append(content)
            if not documents:
                return ["No relevant documents found. Try a more specific query with named entities, dates, or numbers."]
            documents.append("[retriever returned only low-content fragments; issue a more specific query with named entities, dates, or numbers]")
        elif skipped_short:
            documents.append(f"[{skipped_short} short fragments were filtered; narrow the query if you need more detail]")

        return documents

    @classmethod
    def _disable_summarization(cls, reason: str) -> None:
        if cls._summarize_disabled_reason is None:
            cls._summarize_disabled_reason = reason
            logger.warning(
                f"{reason} — disabling retrieval summarization for this process; "
                "falling back to chunked docs without summarization"
            )

    def forward(self, query: str, top_k: int | None = None, *args, **kwargs: Any) -> ToolOutput:
        """
        Execute a search query using the dense retrieval server.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            ToolOutput: Search results or error message
        """
        try:
            # Use provided parameters or defaults
            top_k = top_k or self.max_results

            # Prepare request payload. Ask for more candidates than we
            # plan to show so the min-word filter in
            # ``_format_search_results`` has headroom (P0-1).
            payload = {
                "query": query,
                "top_k": max(top_k, 15),
                "description": "",
                "args": [],  # Add empty args
                "kwargs": {},  # Add empty kwargs
            }

            # Make request to retrieval server
            response = self.client.post(f"{self.server_url}/retrieve", json=payload)

            if not response.is_success:
                error_msg = f"Retrieval server error: {response.status_code}"
                if response.content:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('error', 'Unknown error')}"
                    except Exception:
                        error_msg += f" - {response.text}"

                return ToolOutput(name=self.name, error=error_msg)

            # Parse response
            response_data = response.json()
            results = response_data.get("results", [])

            if not results:
                return ToolOutput(name=self.name, output="No relevant documents found for the query.")

            # Format results
            documents = self._format_search_results(results, query)

            # Evidence-mode vs summary-mode (fix #4).
            #
            # Default behaviour is raw top-k passages. Set
            # ``RLLM_RETRIEVAL_SUMMARIZE=1`` to request abstractive
            # summaries via the server's ``/summarize`` endpoint.
            # If that service is unavailable (unreachable, non-200,
            # empty response, or raises), we auto fall back to the
            # chunked setup of retrieval docs without summarization.
            use_summary_requested = (
                os.environ.get("RLLM_RETRIEVAL_SUMMARIZE", "0") == "1"
                and self._summarize_disabled_reason is None
            )
            content = "\n\n".join(documents)
            summary_used = False
            if use_summary_requested:
                try:
                    payload = {
                        "documents": [{"content": d} for d in documents],
                        "max_length": 256,
                    }
                    response = self.client.post(f"{self.server_url}/summarize", json=payload)
                    if response.status_code == 200:
                        summary_data = response.json()
                        candidate = summary_data.get("summary", "").split("# Summary:", 1)[-1].strip()
                        if candidate:
                            content = candidate
                            summary_used = True
                        else:
                            self._disable_summarization("Summarize endpoint returned empty content")
                    else:
                        self._disable_summarization(f"Summarize endpoint returned status {response.status_code}")
                except Exception as e:
                    self._disable_summarization(f"Summarize service unavailable ({e})")
                    content = "\n\n".join(documents)

            # Cap total content by the *effective* mode, not the requested
            # one: a summary fallback that yields chunked passages should
            # get the 2048-word chunked budget, not the 256-word summary
            # budget (which would truncate most of the evidence).
            word_budget = 256 if summary_used else 2048
            words = content.split()
            if len(words) >= word_budget:
                content = " ".join(words[:word_budget]) + "..."
            summary = content

            # Create metadata for potential downstream use
            metadata = {"query": query, "num_results": len(results), "retriever_type": "dense", "server_url": self.server_url, "summary": summary}

            return ToolOutput(name=self.name, output=summary, metadata=metadata)

        except httpx.TimeoutException:
            return ToolOutput(name=self.name, error=f"Request timeout after {self.timeout} seconds. Please check if the retrieval server is running.")
        except httpx.ConnectError:
            return ToolOutput(name=self.name, error=f"Could not connect to retrieval server at {self.server_url}. Please ensure the server is running.")
        except Exception as e:
            return ToolOutput(name=self.name, error=f"Unexpected error: {str(e)}")

    def close(self):
        """Explicitly close the HTTP client and release connections."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
            self.client = None

    def __del__(self):
        """Clean up HTTP client."""
        try:
            self.close()
        except Exception:
            pass


# Convenience function for tool registry
def create_local_retrieval_tool(server_url: str = "http://127.0.0.1:8000", max_results: int = 3) -> LocalRetrievalTool:
    """
    Create a LocalRetrievalTool instance with specified configuration.

    Args:
        server_url: URL of the dense retrieval server
        max_results: Maximum number of results to return

    Returns:
        LocalRetrievalTool instance
    """
    return LocalRetrievalTool(server_url=server_url, max_results=max_results)
