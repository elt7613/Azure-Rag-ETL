"""Create or update the Azure AI Search index. Run once before first ingestion."""
from rag.targets.azure_search import ensure_index

if __name__ == "__main__":
    ensure_index()
    print("index created or updated")
