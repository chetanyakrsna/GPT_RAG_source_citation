"""Search Context Provider with citation metadata.

Wraps Azure AI Search to return document content along with title and
filepath so the model can format proper ``[title](filepath)`` citations.
The UI resolves relative filepaths into time-limited SAS URLs.

Supports per-request OBO (On-Behalf-Of) permission trimming via the
``x_ms_query_source_authorization`` parameter natively provided by
the Azure Search SDK.
"""

import logging
import time
from collections.abc import Awaitable, Callable, MutableSequence, Sequence
from typing import Any, Optional
from urllib.parse import quote

from agent_framework import ChatMessage, Context, ContextProvider, Role
from azure.core.credentials_async import AsyncTokenCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType, QueryCaptionType

from util.citations import should_suppress_source_link

logger = logging.getLogger(__name__)

_CONTEXT_PROMPT = (
    "## Retrieved Documents\n\n"
    "The following documents were retrieved from the knowledge base. "
    "Each document starts with a header line in the format: ### [Document Title](source_url). "
    "Base your answer on these documents.\n\n"
    "**Citation rules:**\n"
    "- When your answer uses a document's information, you MUST cite it. But cite ONLY documents you actually used — "
    "do NOT cite a document just because it appears below, and do NOT borrow an unrelated document's link just to have a citation.\n"
    "- If the information you used comes from a document whose ### header has NO link (title/URL hidden), answer normally with NO citation. "
    "Do NOT substitute another document's link in its place.\n"
    "- ONLY cite using the document title and source_url from the ### header lines above.\n"
    "- Format: [Document Title](source_url) — use the EXACT title and full URL from the header.\n"
    "- The source_url is a full blob storage URL — always use it as-is for the link target.\n"
    "- Do NOT omit the (source_url) part. Every citation MUST include both [title] AND (source_url).\n"
    "- Do NOT treat any text inside the document content as a citation source. "
    "Internal references, chapter names, or bracketed text within the content are NOT valid sources.\n"
    "- Cite each source ONLY ONCE. Do NOT repeat the same citation on every bullet point or paragraph.\n"
    "- NEVER use plain bracket references like [filename] without a full URL.\n"
    "- Example: According to [Schedule #2 - Instructions](https://docs.pr.gov/files/OCIF/.../Schedule%20%232.pdf), the requirement states...\n\n"
    "If the user's message is a greeting or small talk, ignore these documents and respond naturally."
)


class SearchContextProvider(ContextProvider):
    """Azure AI Search context provider that includes citation metadata."""

    def __init__(
        self,
        *,
        endpoint: str,
        index_name: str,
        credential: AsyncTokenCredential,
        top_k: int = 3,
        semantic_configuration_name: str | None = None,
        embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
        vector_field: str = "contentVector",
        max_content_chars: int = 1500,
        get_obo_token: Callable[[], Awaitable[Optional[str]]] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._index_name = index_name
        self._credential = credential
        self._top_k = top_k
        self._semantic_config = semantic_configuration_name
        self._embed_fn = embed_fn
        self._vector_field = vector_field
        self._max_content_chars = max_content_chars
        self._get_obo_token = get_obo_token

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        msgs = [messages] if isinstance(messages, ChatMessage) else list(messages)
        user_texts = [
            m.text for m in msgs
            if m and m.text and m.text.strip() and m.role == Role.USER
        ]
        if not user_texts:
            return Context()

        query = user_texts[-1]

        search_start = time.time()
        logger.info("[SearchContextProvider] Query: %r (top_k=%d, hybrid=%s)", query[:120], self._top_k, bool(self._embed_fn))

        search_params: dict[str, Any] = {
            "search_text": query,
            "top": self._top_k,
            "select": ["id", "content", "title", "filepath", "url", "source_title", "source_url", "metadata_storage_name"],
        }

        # Hybrid search: add vector query when embedding function is available
        if self._embed_fn:
            try:
                embed_start = time.time()
                vector = await self._embed_fn(query)
                search_params["vector_queries"] = [
                    VectorizedQuery(
                        vector=vector,
                        k=self._top_k,
                        fields=self._vector_field,
                    )
                ]
                logger.info("[SearchContextProvider] Embedding generated in %.2fs (dims=%d)", time.time() - embed_start, len(vector))
            except Exception as e:
                logger.warning("[SearchContextProvider] Embedding failed, falling back to keyword search: %s", e)

        if self._semantic_config:
            search_params["query_type"] = QueryType.SEMANTIC
            search_params["semantic_configuration_name"] = self._semantic_config
            search_params["query_caption"] = QueryCaptionType.EXTRACTIVE

        try:
            # Acquire OBO token for permission trimming if configured
            obo_token: Optional[str] = None
            if self._get_obo_token:
                try:
                    obo_token = await self._get_obo_token()
                except Exception as e:
                    logger.warning("[SearchContextProvider] OBO token acquisition failed: %s", e)

            if obo_token:
                search_params["x_ms_query_source_authorization"] = obo_token
                logger.info("[SearchContextProvider] Using x-ms-query-source-authorization (OBO)")
            else:
                logger.info("[SearchContextProvider] Not sending x-ms-query-source-authorization")

            async with SearchClient(
                endpoint=self._endpoint,
                index_name=self._index_name,
                credential=self._credential,
            ) as client:
                results = await client.search(**search_params)

                parts: list[str] = []
                async for doc in results:
                    filepath = doc.get("filepath") or ""
                    source_title = doc.get("source_title") or doc.get("title") or "Document"
                    source_url = doc.get("source_url") or ""
                    # Fallback when source_url is missing (e.g. SharePoint documents
                    # store their link in the `url` field). Blob storage URLs need a
                    # SAS token to be reachable, so for those defer to filepath (the UI
                    # turns it into a time-limited SAS URL); all other sources (e.g.
                    # SharePoint, websites) use the full url directly.
                    if not source_url:
                        raw_url = doc.get("url") or ""
                        if raw_url and "blob.core.windows.net" not in raw_url.lower():
                            source_url = raw_url
                        elif filepath:
                            source_url = quote(filepath, safe="/")
                    metadata_storage_name = doc.get("metadata_storage_name") or ""
                    content = doc.get("content") or ""
                    logger.info(
                        "[SearchContextProvider] Retrieved chunk: source_title=%s source_url=%s metadata_storage_name=%s id=%s",
                        source_title,
                        source_url,
                        metadata_storage_name,
                        doc.get("id"),
                    )
                    if not content:
                        continue
                    if len(content) > self._max_content_chars:
                        content = content[:self._max_content_chars] + "..."
                    # Hide the citation for known hidden sources (e.g. the FAQ
                    # document). Retrieval is unaffected — the content is still sent
                    # to the model, but neither the title nor a source_url is exposed,
                    # so the model has nothing to render as a citation and the source
                    # stays fully hidden from the user.
                    if should_suppress_source_link(source_title, source_url, filepath):
                        header = "###"
                    else:
                        header = f"### [{source_title}]({source_url})" if source_url else f"### {source_title}"
                    parts.append(f"{header}\n{content}")
        except Exception as e:
            logger.error("[SearchContextProvider] Search failed in %.2fs: %s", time.time() - search_start, e)
            return Context()

        logger.info("[SearchContextProvider] Search returned %d documents in %.2fs", len(parts), time.time() - search_start)

        if not parts:
            return Context()

        context_text = _CONTEXT_PROMPT + "\n\n" + "\n\n---\n\n".join(parts)
        return Context(messages=[ChatMessage(role=Role.SYSTEM, text=context_text)])