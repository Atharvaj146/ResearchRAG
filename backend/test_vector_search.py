from app.rag.ingestion import process_pdf
from app.rag.embeddings import create_embeddings
from app.rag.vector_store import (
    create_collection,
    add_chunks,
    search
)


chunks = process_pdf(
    "test.pdf",
    "test.pdf",
    "doc_001"
)

print("Chunks:", len(chunks))


texts = [chunk["text"] for chunk in chunks]

embeddings = create_embeddings(texts)

print("Embeddings created:", len(embeddings))


create_collection()

add_chunks(
    chunks,
    embeddings
)

print("Chunks stored in Qdrant")


question = "What is this document about?"

question_embedding = create_embeddings(
    [question]
)[0]


results = search(
    question_embedding,
    top_k=5
)


print("\nSEARCH RESULTS\n")

for result in results:

    print("Score:", result.score)

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
        "Chunk ID:",
        result.payload["chunk_id"]
    )

    print(
        "Text:",
        result.payload["text"][:200]
    )

    print("--------------------")