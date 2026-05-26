"""Quick test: verify source_title and source_url are populated in the search index."""

import asyncio
from azure.identity.aio import AzureCliCredential
from azure.search.documents.aio import SearchClient

# --- Update these values ---
SEARCH_ENDPOINT = "https://srch-lejbubzk5gqf2.search.windows.net"
INDEX_NAME = "ragindex-lejbubzk5gqf2"
# ---------------------------

async def main():
    credential = AzureCliCredential()
    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=credential,
    )

    try:
        results = await client.search(
            search_text="*",
            select=["id", "metadata_storage_name", "source_title", "source_url"],
            top=10,
        )

        count = 0
        async for doc in results:
            count += 1
            print(f"\n--- Document {count} ---")
            print(f"  id:                    {doc.get('id', '')[:60]}")
            print(f"  metadata_storage_name: {doc.get('metadata_storage_name', '')}")
            print(f"  source_title:          {doc.get('source_title', '') or '(empty)'}")
            print(f"  source_url:            {doc.get('source_url', '') or '(empty)'}")

        if count == 0:
            print("No documents found in the index.")
        else:
            print(f"\n--- Total: {count} documents shown ---")

    finally:
        await client.close()
        await credential.close()

if __name__ == "__main__":
    asyncio.run(main())
