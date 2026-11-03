from app.rag.ingestion import process_pdf
from app.rag.embeddings import create_embeddings
from app.rag.vector_store import (
    create_collection,
    add_chunks,
    search
)


# -------------------------
# Document 1
# -------------------------

chunks_1 = process_pdf(
    "test.pdf",
    "test.pdf",
    "doc_001"
)

print(
    "Document 1 chunks:",
    len(chunks_1)
)


# -------------------------
# Document 2
# -------------------------

chunks_2 = process_pdf(
    "test1.pdf",
    "test1.pdf",
    "doc_002"
)

print(
    "Document 2 chunks:",
    len(chunks_2)
)


# -------------------------
# Combine documents
# -------------------------

all_chunks = chunks_1 + chunks_2

print(
    "Total chunks:",
    len(all_chunks)
)


# -------------------------
# Create embeddings
# -------------------------

texts = [
    chunk["text"]
    for chunk in all_chunks
]

embeddings = create_embeddings(
    texts
)

print(
    "Total embeddings:",
    len(embeddings)
)


# -------------------------
# Store in Qdrant
# -------------------------

create_collection()

add_chunks(
    all_chunks,
    embeddings
)


# -------------------------
# Search
# -------------------------

question = "What is the main method discussed in these papers?"

question_embedding = create_embeddings(
    [question]
)[0]

results = search(
    question_embedding,
    top_k=5
)


print("\nSEARCH RESULTS\n")

for result in results:

    print(
        "Document:",
        result.payload["document_name"]
    )

    print(
        "Document ID:",
        result.payload["document_id"]
    )

    print(
        "Page:",
        result.payload["page_number"]
    )

    print(
        "Score:",
        result.score
    )

    print(
        "Text:",
        result.payload["text"][:200]
    )

    print("--------------------")