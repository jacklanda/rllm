#!/usr/bin/env python3
"""
Simple script to query the dense retrieval server.
"""

import requests
import json
import sys

# Server configuration
SERVER_URL = "http://127.0.0.1:23456"

def search(query: str, top_k: int = 5):
    """
    Send a search query to the retrieval server.
    
    Args:
        query: Search query string
        top_k: Number of results to retrieve (default: 5)
    
    Returns:
        JSON response with search results
    """
    # Prepare request payload - match what the server expects
    payload = {
        "query": query,
        "top_k": min(top_k, 50),
        "description": "",
        "args": [],      # Add empty args
        "kwargs": {}     # Add empty kwargs
    }
    
    try:
        # Make POST request to retrieval server
        response = requests.post(
            f"{SERVER_URL}/retrieve",
            json=payload,
            timeout=30
        )
        
        # Print detailed error info if request fails
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
        
        response.raise_for_status()
        
        # Parse and return results
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error querying server: {e}", file=sys.stderr)
        return None


def print_results(results):
    """Pretty print search results."""
    if not results:
        print("No results found.")
        return
    
    print(f"\nQuery: {results.get('query', 'N/A')}")
    print(f"Method: {results.get('method', 'N/A')}")
    print(f"Number of results: {results.get('num_results', 0)}")
    print("\n" + "="*80)
    
    for i, item in enumerate(results.get('results', []), 1):
        doc = item.get('document', {})
        score = item.get('score', 0.0)
        
        print(f"\nResult #{i} (score: {score:.4f})")
        print(f"Title: {doc.get('title', 'N/A')}")
        
        # Handle text display
        text = doc.get('text', '') or doc.get('contents', 'N/A')
        if len(text) > 200:
            print(f"Text: {text[:200]}...")
        else:
            print(f"Text: {text}")
        print("-"*80)


def main():
    # Check if server is healthy
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        health.raise_for_status()
        health_data = health.json()
        print("✓ Server is healthy")
        print(f"  Corpus size: {health_data.get('corpus_size', 'N/A')}")
        print(f"  Method: {health_data.get('retrieval_method', 'N/A')}")
        print(f"  Top-K: {health_data.get('topk', 'N/A')}")
        print(f"  Model: {health_data.get('model_path', 'N/A')}")
    except Exception as e:
        print(f"✗ Server health check failed: {e}", file=sys.stderr)
        return
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Python programming language",
        "Climate change effects"
    ]
    
    # Process queries
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Searching for: '{query}'")
        print('='*80)
        
        results = search(query, top_k=3)
        if results:
            print_results(results)
        
        print()


if __name__ == "__main__":
    # You can also run with custom query from command line
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        top_k = 5
        
        print(f"Searching for: '{query}'")
        results = search(query, top_k=top_k)
        if results:
            print_results(results)
    else:
        # Run example queries
        main()
