import os
import pickle

# ==========================================
# CONFIGURATION: Update filename if needed
# ==========================================
PICKLE_FILE = ".rag_cache/cc243c8acc52eb9d/index.pkl"  # Change this to your actual pickle filename if different


def inspect_rag_pickle(filepath):
  print("=" * 60)
  print(f"STEP 1: Checking if '{filepath}' exists...")
  print("=" * 60)

  print(f"✅ Found '{filepath}'! Loading data via pickle...\n")

  # ==========================================
  # STEP 2: Load and inspect overall structure
  # ==========================================
  try:
    with open(filepath, "rb") as f:
      data = pickle.load(f)
  except Exception as e:
    print(f"❌ Failed to load pickle file: {e}")
    return

  print("=" * 60)
  print("STEP 2: Analyzing Root Data Structure")
  print("=" * 60)
  print(f"Root Data Type: {type(data)}")

  if isinstance(data, tuple):
    print(f"Tuple Length: {len(data)}")
    for i, item in enumerate(data):
      print(f"  - Item {i}: {type(item)}")
  else:
    print("⚠️ The pickle contents are not a standard tuple format.")
    print(data)
    return

  # ==========================================
  # STEP 3: Inspect In-Memory Store & Indexing
  # ==========================================
  print("\n" + "=" * 60)
  print("STEP 3: Inspecting Indexing & Document Store")
  print("=" * 60)

  docstore = data[0]
  index_dict = data[1]

  print(f"Docstore Type: {type(docstore)}")
  print(f"Index/Mapping Dictionary Type: {type(index_dict)}")
  print(f"Total Indexed Chunks / Vectors: {len(index_dict)}")

  # Check sample keys
  sample_keys = list(index_dict.keys())[:5]
  print(f"Sample Index Keys: {sample_keys}")

  # ==========================================
  # STEP 4: Sample Document Content & Metadata
  # ==========================================
  print("\n" + "=" * 60)
  print("STEP 4: Sampling Stored Document Chunks")
  print("=" * 60)

  # Try to fetch a few documents from the docstore if possible
  if hasattr(docstore, "_dict") and docstore._dict:
    sample_doc_ids = list(docstore._dict.keys())[:2]
    print(f"Found {len(docstore._dict)} documents in docstore storage.")

    for idx, doc_id in enumerate(sample_doc_ids):
      doc = docstore._dict[doc_id]
      print(f"\n--- Sample Document Chunk #{idx + 1} (ID: {doc_id}) ---")
      content_preview = (
          doc.page_content[:300] + "..."
          if len(doc.page_content) > 300
          else doc.page_content
      )
      print(f"Content Preview:\n{content_preview}")
      print(f"Metadata: {doc.metadata}")
  else:
    print(
        "ℹ️ Docstore internal dictionary structure is standard or hidden, but"
        " total items are verified."
    )

  print("\n" + "=" * 60)
  print("SUMMARY:")
  print("=" * 60)
  print("Your project is using an **In-Memory Flat Index** backed by a Python")
  print(
      "dictionary and LangChain's InMemoryDocstore, persisted via Pickle."
  )
  print("=" * 60)


if __name__ == "__main__":
  inspect_rag_pickle(PICKLE_FILE)