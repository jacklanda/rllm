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
        max_results: int = 3,
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
        self.client = httpx.Client(timeout=timeout)

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

    def _format_search_results(self, results: list[dict[str, Any]], query: Optional[str] = None) -> list[str]:
        """Format search results for LLM consumption."""
        if not results:
            return "No relevant documents found."

        content = None
        documents = []
        for i, result in enumerate(results[: self.max_results], 1):
            # Extract key information
            # doc_id = result.get("id", f"doc_{i}")
            # content = result.get("content", "").get("original_text")  # use full text
            try:
                if "document" in result and "score" in result:
                    content = result.get("document")  # use full document text
                    # score = result.get("score", 0.0)
                elif "content" in result and "chunk_text" in result["content"]:
                    content = result.get("content").get("chunk_text")  # use chunked text
                    # score = result.get("score", 0.0)
                elif "chunk_text" in result:
                    content = result.get("chunk_text")
                else:
                    raise ValueError("Unknown result format")
            except Exception as _:
                logger.warning(f"Error parsing content {content}")
                content = "Nothing retrieved, please tweak your search query and search again."
            else:
                if not content:
                    logger.warning(f"Error parsing content {content}")
                    content = "Nothing retrieved, please tweak your search query and search again."

            documents.append(content)

        return documents

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

            # Prepare request payload
            payload = {
                "query": query,
                "top_k": min(top_k, 5),
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

            # Truncate content if too long (keep first 512 characters)
            if True:  # TODO: replace the condition to check if summarization is enabled
                try:
                    payload = {
                        "documents": [
                            {
                                "content": document,
                            }
                            for document in documents
                        ],
                        # "query": query or "Summarize the above document.",
                        "max_length": 256,
                    }
                    response = self.client.post(f"{self.server_url}/summarize", json=payload)
                    if response.status_code == 200:
                        summary = response.json()
                        content = summary.get("summary", "").split("# Summary:", 1)[-1].strip()
                except Exception as e:
                    logger.warning(f"Error during summarization: {e}")

            if len(content.split()) >= 128:
                summary = " ".join(content.split()[:128]) + "..."
            else:
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

    def __del__(self):
        """Clean up HTTP client."""
        try:
            if hasattr(self, "client"):
                self.client.close()
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
