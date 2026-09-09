"""
web_search_client.py

MCP client wrapper for DocMind that launches the DuckDuckGo MCP server
as a subprocess and exposes a simple async `search()` function your
RAG pipeline can call when a question needs live web results instead
of (or alongside) retrieved PDF chunks.

Requires:
    pip install mcp
    uv pip install "duckduckgo-mcp-server[browser]"   # the server itself

The DuckDuckGo server is launched via `uvx`, so `uv` must be on PATH
(same as your weather.py MCP project).
"""

import asyncio
import json
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ---------------------------------------------------------------------------
# Server launch config — mirrors what you'd put in a claude_desktop_config.json
# "mcpServers" entry, just expressed in Python instead of JSON.
# ---------------------------------------------------------------------------
DDG_SERVER_PARAMS = StdioServerParameters(
    command="uvx",
    args=["--with", "duckduckgo-mcp-server[browser]", "duckduckgo-mcp-server"],
)


class WebSearchClient:
    """
    Thin async wrapper around an MCP stdio session for the DuckDuckGo server.

    Usage:
        async with WebSearchClient() as client:
            results = await client.search("current inflation rate Sri Lanka")
    """

    def __init__(self, server_params: StdioServerParameters = DDG_SERVER_PARAMS):
        self.server_params = server_params
        self._exit_stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None

    async def __aenter__(self) -> "WebSearchClient":
        self._exit_stack = AsyncExitStack()

        # Launch the server subprocess and open the stdio transport
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(self.server_params)
        )
        read_stream, write_stream = stdio_transport

        # Open an MCP session over that transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # MCP handshake: initialize, then discover available tools
        await self.session.initialize()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exit_stack:
            await self._exit_stack.aclose()

    async def list_tools(self) -> list[str]:
        """Return the names of tools this server exposes (e.g. 'search', 'fetch_content')."""
        result = await self.session.list_tools()
        return [tool.name for tool in result.tools]

    async def search(self, query: str, max_results: int = 5) -> str:
        """
        Call the server's 'search' tool and return a formatted string of
        results ready to be injected into an LLM prompt.
        """
        result = await self.session.call_tool(
            "search",
            arguments={"query": query, "max_results": max_results},
        )

        # MCP tool results come back as a list of content blocks (usually text)
        text_chunks = [block.text for block in result.content if block.type == "text"]
        return "\n".join(text_chunks)

    async def fetch_content(self, url: str) -> str:
        """Call the server's 'fetch_content' tool to pull full page text from a URL."""
        result = await self.session.call_tool("fetch_content", arguments={"url": url})
        text_chunks = [block.text for block in result.content if block.type == "text"]
        return "\n".join(text_chunks)


# ---------------------------------------------------------------------------
# Standalone test — run this file directly to confirm the wiring works
# before integrating it into DocMind's Streamlit app.
# ---------------------------------------------------------------------------
async def _test():
    async with WebSearchClient() as client:
        tools = await client.list_tools()
        print("Available tools:", tools)

        results = await client.search("latest AI model releases 2026")
        print("\n--- Search results ---\n")
        print(results)


if __name__ == "__main__":
    asyncio.run(_test())