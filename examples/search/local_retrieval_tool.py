#!/usr/bin/env python3

import os
import logging
import re
from typing import Any, Optional

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class LocalRetrievalTool(Tool):
    """
    A tool for lexical search using the local retrieval server.

    This tool connects to a retrieval server and performs lexical retrieval by
    default. Set ``retrieval_mode`` to use a different server-supported mode.
    """

    NAME = "local_search"
    DESCRIPTION = "Search for information using a lexical retrieval server with Wikipedia corpus"

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        server_url: str = None,
        timeout: float = 3600.0,
        max_results: int = 10,
        retrieval_mode: str | None = None,
        retrieval_max_words: int | None = None,
    ):
        """
        Initialize the Local Retrieval Tool.

        Args:
            name: Tool name
            description: Tool description
            server_url: URL of the local retrieval server (if None, checks RETRIEVAL_SERVER_URL env var)
            timeout: Request timeout in seconds
            max_results: Maximum number of results to return
            retrieval_mode: Retrieval mode to request from the server. Defaults to lexical.
            retrieval_max_words: Maximum words per returned passage requested from the server.
        """
        # Use environment variable if server_url not provided
        if server_url is None:
            server_url = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000")
        if retrieval_mode is None:
            retrieval_mode = os.environ.get("RLLM_RETRIEVAL_MODE", "lexical")
        if retrieval_max_words is None:
            retrieval_max_words = int(os.environ.get("RLLM_RETRIEVAL_MAX_WORDS", "64"))

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.retrieval_mode = str(retrieval_mode or "lexical").strip() or "lexical"
        self.retrieval_max_words = int(retrieval_max_words)
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
    _summarize_probe_calls = 0

    @classmethod
    def _log_summarization_probe(
        cls,
        *,
        env_value: str,
        requested: bool,
        attempted: bool,
        summary_used: bool,
        num_results: int,
        num_documents: int,
        query: str,
    ) -> None:
        cls._summarize_probe_calls += 1
        log_every = max(1, int(os.environ.get("RLLM_RETRIEVAL_SUMMARY_LOG_EVERY", "1")))
        if cls._summarize_probe_calls % log_every != 0:
            return
        logger.warning(
            "[retrieval-summary-probe] call=%s env_RLLM_RETRIEVAL_SUMMARIZE=%r " "requested=%s attempted=%s summary_used=%s disabled_reason=%r " "num_results=%s num_documents=%s query=%r",
            cls._summarize_probe_calls,
            env_value,
            requested,
            attempted,
            summary_used,
            cls._summarize_disabled_reason,
            num_results,
            num_documents,
            str(query)[:200],
        )

    @staticmethod
    def _normalize_doc_signature(text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized[:500]

    def _extract_doc_title(self, result: dict[str, Any], content: str | None = None) -> str:
        for key in ("title", "source", "url"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        nested = result.get("content")
        if isinstance(nested, dict):
            for key in ("title", "source", "url"):
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if content:
            first_line = content.strip().splitlines()[0].strip()
            if first_line and len(first_line.split()) <= 16:
                return first_line
        return "Untitled"

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

    def _format_search_results(self, results: list[dict[str, Any]], query: Optional[str] = None) -> tuple[list[str], dict[str, Any]]:
        """Format search results for LLM consumption."""
        if not results:
            return ["No relevant documents found."], {"num_unique": 0, "num_duplicates": 0, "num_short_filtered": 0}

        documents: list[str] = []
        seen_signatures: set[str] = set()
        duplicate_count = 0
        skipped_short = 0
        for result in results:
            content = self._extract_doc_text(result)
            if not content:
                continue
            signature_source = " ".join(
                str(v)
                for v in (
                    result.get("title"),
                    result.get("url"),
                    content,
                )
                if v
            )
            signature = self._normalize_doc_signature(signature_source)
            if signature and signature in seen_signatures:
                duplicate_count += 1
                continue
            if signature:
                seen_signatures.add(signature)
            if len(content.split()) < self._MIN_DOC_WORDS:
                skipped_short += 1
                continue
            title = self._extract_doc_title(result, content)
            documents.append(f"[Result {len(documents) + 1}] Title: {title}\nSnippet: {content.strip()}")
            if len(documents) >= self.max_results:
                break

        if not documents:
            return [
                "No usable evidence was found for this query. The returned passages were duplicates, too short, or too generic. "
                "Do not submit an answer from this result. Rewrite the query with a specific title, quoted phrase, named entity, date, number, or one clue from the question, then call web_search again."
            ], {"num_unique": 0, "num_duplicates": duplicate_count, "num_short_filtered": skipped_short}
        unique_count = len(documents)
        if skipped_short:
            documents.append(f"[{skipped_short} short fragments were filtered; narrow the query if you need more detail]")
        if duplicate_count:
            documents.append(f"[{duplicate_count} duplicate passages were removed before display]")

        return documents, {"num_unique": unique_count, "num_duplicates": duplicate_count, "num_short_filtered": skipped_short}

    @classmethod
    def _disable_summarization(cls, reason: str) -> None:
        if cls._summarize_disabled_reason is None:
            cls._summarize_disabled_reason = reason
            logger.warning(f"{reason} — disabling retrieval summarization for this process; " "falling back to chunked docs without summarization")

    def forward(self, query: str, top_k: int | None = None, *args, **kwargs: Any) -> ToolOutput:
        """
        Execute a search query using the retrieval server.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            ToolOutput: Search results or error message
        """
        try:
            # Use provided parameters or defaults
            top_k = top_k or self.max_results

            # Prepare request payload following the retrieval server API:
            # {"query": ..., "top_k": ..., "max_words": ..., "mode": ...}.
            payload = {
                "query": query,
                "top_k": top_k,
                "max_words": self.retrieval_max_words,
                "mode": self.retrieval_mode,
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
            documents, format_metadata = self._format_search_results(results, query)

            # Evidence-mode vs summary-mode (fix #4).
            #
            # Default behaviour is raw top-k passages. Set
            # ``RLLM_RETRIEVAL_SUMMARIZE=1`` to request abstractive
            # summaries via the server's ``/summarize`` endpoint.
            # Temporary strict summary mode: keep attempting /summarize when
            # requested and surface failures instead of silently falling back.
            summarize_env_value = os.environ.get("RLLM_RETRIEVAL_SUMMARIZE", "0")
            use_summary_requested = summarize_env_value == "1"
            content = "\n\n".join(documents)
            summary_used = False
            summarize_attempted = False
            if use_summary_requested:
                summarize_attempted = True
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
                            self._log_summarization_probe(
                                env_value=summarize_env_value,
                                requested=use_summary_requested,
                                attempted=summarize_attempted,
                                summary_used=summary_used,
                                num_results=len(results),
                                num_documents=len(documents),
                                query=query,
                            )
                            return ToolOutput(name=self.name, error="Summarize endpoint returned empty content")
                    else:
                        self._log_summarization_probe(
                            env_value=summarize_env_value,
                            requested=use_summary_requested,
                            attempted=summarize_attempted,
                            summary_used=summary_used,
                            num_results=len(results),
                            num_documents=len(documents),
                            query=query,
                        )
                        return ToolOutput(name=self.name, error=f"Summarize endpoint returned status {response.status_code}")
                except Exception as e:
                    self._log_summarization_probe(
                        env_value=summarize_env_value,
                        requested=use_summary_requested,
                        attempted=summarize_attempted,
                        summary_used=summary_used,
                        num_results=len(results),
                        num_documents=len(documents),
                        query=query,
                    )
                    return ToolOutput(name=self.name, error=f"Summarize service unavailable ({e})")
            self._log_summarization_probe(
                env_value=summarize_env_value,
                requested=use_summary_requested,
                attempted=summarize_attempted,
                summary_used=summary_used,
                num_results=len(results),
                num_documents=len(documents),
                query=query,
            )

            # Cap total content by the *effective* mode, not the requested
            # one: a summary fallback that yields chunked passages should
            # get the 512-word chunked budget, not the 256-word summary
            # budget (which would truncate most of the evidence).
            word_budget = 256 if summary_used else 512
            words = content.split()
            if len(words) >= word_budget:
                content = " ".join(words[:word_budget]) + " ..."
            summary = content

            # Create metadata for potential downstream use
            metadata = {
                "query": query,
                "num_results": len(results),
                "retriever_type": self.retrieval_mode,
                "server_url": self.server_url,
                "summary": summary,
                "summary_used": summary_used,
                **format_metadata,
            }

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
def create_local_retrieval_tool(
    server_url: str = "http://127.0.0.1:8000",
    max_results: int = 3,
    retrieval_mode: str | None = None,
    retrieval_max_words: int | None = None,
) -> LocalRetrievalTool:
    """
    Create a LocalRetrievalTool instance with specified configuration.

    Args:
        server_url: URL of the dense retrieval server
        max_results: Maximum number of results to return
        retrieval_mode: Retrieval mode to request from the server
        retrieval_max_words: Maximum words per returned passage requested from the server

    Returns:
        LocalRetrievalTool instance
    """
    return LocalRetrievalTool(
        server_url=server_url,
        max_results=max_results,
        retrieval_mode=retrieval_mode,
        retrieval_max_words=retrieval_max_words,
    )
