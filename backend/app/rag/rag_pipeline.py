from app.rag.embeddings import create_embeddings
from app.rag.hybrid_search import hybrid_search
from app.rag.reranker import rerank_results
from app.services.llm import (
    generate_answer,
    rewrite_query
)


def ask_question(question, chat_history=None):

    if chat_history is None:
        chat_history = []

    # -----------------------------------------
    # STEP 1: Rewrite the question
    # -----------------------------------------

    if chat_history:

        search_query = rewrite_query(
            question,
            chat_history
        )

    else:

        search_query = question

    print("\nOriginal Question:")
    print(question)

    print("\nSearch Query:")
    print(search_query)

    # -----------------------------------------
    # STEP 2: Create query embedding
    # -----------------------------------------

    query_embedding = create_embeddings(
        [search_query]
    )[0]

    # -----------------------------------------
    # STEP 3: Hybrid retrieval
    # -----------------------------------------

    retrieved_results = hybrid_search(
        query_embedding,
        search_query,
        top_k=20
    )

    # -----------------------------------------
    # STEP 4: Cross-encoder reranking
    # -----------------------------------------

    reranked_results = rerank_results(
        search_query,
        retrieved_results,
        top_k=5
    )

    # -----------------------------------------
    # STEP 5: Build context
    # -----------------------------------------

    context_parts = []

    for item in reranked_results:

        result = item["result"]

        text = result.payload["text"]
        document_name = result.payload["document_name"]
        page_number = result.payload["page_number"]

        context_parts.append(
            f"Source: {document_name}, Page {page_number}\n"
            f"{text}"
        )

    context = "\n\n".join(context_parts)

    # -----------------------------------------
    # STEP 6: Generate answer
    # -----------------------------------------

    answer = generate_answer(
        question,
        context
    )

    # -----------------------------------------
    # STEP 7: Prepare citations
    # -----------------------------------------

    sources = []

    for item in reranked_results:

        result = item["result"]

        sources.append(
            {
                "document_name": result.payload["document_name"],
                "document_id": result.payload["document_id"],
                "page_number": result.payload["page_number"],
                "chunk_id": result.payload["chunk_id"],
                "score": item["score"]
            }
        )

    # -----------------------------------------
    # STEP 8: Return result
    # -----------------------------------------

    return {
        "question": question,
        "search_query": search_query,
        "answer": answer,
        "sources": sources
    }