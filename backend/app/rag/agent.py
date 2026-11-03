from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.rag.embeddings import create_embeddings

from app.rag.hybrid_search import (
    hybrid_search,
    get_documents_for_session
)

from app.rag.reranker import rerank_results

from app.services.llm import (
    generate_answer,
    generate_general_answer,
    rewrite_query
)


# ==================================================
# RESEARCH STATE
# ==================================================

class ResearchState(TypedDict):

    question: str

    session_id: str

    chat_history: list

    rewritten_query: str

    needs_retrieval: bool

    is_summary_request: bool

    context: str

    answer: str

    retrieval_attempt: int

    retrieval_quality: float

    sources: list


# ==================================================
# NODE 1
# DECIDE WHAT THE AGENT SHOULD DO
# ==================================================

def decide_retrieval(state: ResearchState):

    question = state["question"].lower().strip()

    conversational_phrases = [
        "hello",
        "hi",
        "hey",
        "how are you",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "bye",
        "goodbye"
    ]

    is_conversational = any(
        phrase in question
        for phrase in conversational_phrases
    )

    summary_phrases = [
        "summarize",
        "summarise",
        "summary",
        "give me a summary",
        "give me an overview",
        "overview",
        "compare",
        "comparison",
        "compare the papers",
        "compare these papers",
        "compare the documents",
        "compare these documents"
    ]

    is_summary_request = any(
        phrase in question
        for phrase in summary_phrases
    )

    needs_retrieval = not is_conversational

    print("\n==============================")
    print("AGENT DECISION")
    print("==============================")

    print(
        "Retrieval required:",
        needs_retrieval
    )

    print(
        "Summary/comparison request:",
        is_summary_request
    )

    return {
        "needs_retrieval": needs_retrieval,
        "is_summary_request": is_summary_request
    }


# ==================================================
# NODE 2
# QUERY REWRITING
# ==================================================

def rewrite_user_query(
    state: ResearchState
):

    question = state["question"]

    chat_history = state.get(
        "chat_history",
        []
    )

    # No history → original question
    if not chat_history:

        return {
            "rewritten_query": question
        }

    print("\n==============================")
    print("QUERY REWRITING")
    print("==============================")

    print(
        "Original question:",
        question
    )

    rewritten = rewrite_query(
        question,
        chat_history
    )

    print(
        "Rewritten query:",
        rewritten
    )

    return {
        "rewritten_query": rewritten
    }


# ==================================================
# DOCUMENT-LEVEL SUMMARY RETRIEVAL
# ==================================================

def retrieve_document_summaries(
    session_id
):

    documents = get_documents_for_session(
        session_id
    )

    context_parts = []

    sources = []

    print("\n==============================")
    print("DOCUMENT SUMMARY RETRIEVAL")
    print("==============================")

    print(
        "Documents found:",
        len(documents)
    )

    for document in documents:

        document_name = document[
            "document_name"
        ]

        chunks = document[
            "chunks"
        ]

        print(
            f"Document: {document_name}"
        )

        print(
            f"Chunks available: {len(chunks)}"
        )

        selected_chunks = chunks[:8]

        context_parts.append(
            f"\n"
            f"========================================\n"
            f"DOCUMENT: {document_name}\n"
            f"========================================\n"
        )

        for chunk in selected_chunks:

            payload = chunk.payload

            context_parts.append(
                f"\n"
                f"Page {payload['page_number']}:\n"
                f"{payload['text']}\n"
            )

            sources.append({
                "document_name":
                    document_name,

                "document_id":
                    payload["document_id"],

                "page_number":
                    payload["page_number"],

                "chunk_id":
                    payload["chunk_id"],

                "score":
                    0.0
            })

    context = "\n".join(
        context_parts
    )

    return context, sources


# ==================================================
# NODE 3
# MULTI-DOCUMENT SUMMARY
# ==================================================

def summarize_documents(
    state: ResearchState
):

    question = state[
        "question"
    ]

    session_id = state[
        "session_id"
    ]

    print("\n==============================")
    print("MULTI-DOCUMENT SUMMARY")
    print("==============================")

    context, sources = (
        retrieve_document_summaries(
            session_id
        )
    )

    print(
        "Context characters:",
        len(context)
    )

    answer = generate_answer(
        question,
        context
    )

    return {
        "context": context,
        "sources": sources,
        "answer": answer
    }


# ==================================================
# NODE 4
# NORMAL DOCUMENT RETRIEVAL
# ==================================================

def retrieve_documents(
    state: ResearchState
):

    question = state.get(
        "rewritten_query"
    ) or state["question"]

    session_id = state[
        "session_id"
    ]

    attempt = state.get(
        "retrieval_attempt",
        0
    )

    print("\n==============================")
    print("RETRIEVAL")
    print("==============================")

    print(
        f"Retrieval attempt: "
        f"{attempt + 1}"
    )

    print(
        "Search query:",
        question
    )

    print(
        "Session ID:",
        session_id
    )

    query_embedding = create_embeddings(
        [question]
    )[0]

    candidate_count = (
        20
        if attempt == 0
        else 40
    )

    retrieved_results = hybrid_search(
        query_embedding,
        question,
        session_id,
        top_k=candidate_count
    )

    print(
        "Retrieved candidates:",
        len(retrieved_results)
    )

    reranked_results = rerank_results(
        question,
        retrieved_results,
        top_k=5
    )

    if reranked_results:

        scores = [
            item["score"]
            for item in reranked_results
        ]

        best_score = max(scores)

    else:

        best_score = -999.0

    print(
        f"Best reranker score: "
        f"{best_score:.4f}"
    )

    context_parts = []

    sources = []

    for item in reranked_results:

        result = item[
            "result"
        ]

        text = result.payload[
            "text"
        ]

        document_name = (
            result.payload[
                "document_name"
            ]
        )

        document_id = (
            result.payload[
                "document_id"
            ]
        )

        page_number = (
            result.payload[
                "page_number"
            ]
        )

        chunk_id = (
            result.payload[
                "chunk_id"
            ]
        )

        context_parts.append(
            f"Source: "
            f"{document_name}, "
            f"Page {page_number}\n"
            f"{text}"
        )

        sources.append({

            "document_name":
                document_name,

            "document_id":
                document_id,

            "page_number":
                page_number,

            "chunk_id":
                chunk_id,

            "score":
                float(
                    item["score"]
                )
        })

    context = "\n\n".join(
        context_parts
    )

    return {

        "context":
            context,

        "sources":
            sources,

        "retrieval_attempt":
            attempt + 1,

        "retrieval_quality":
            best_score
    }


# ==================================================
# NODE 5
# GENERATE RESEARCH ANSWER
# ==================================================

def answer_from_documents(
    state: ResearchState
):

    question = state["question"]

    context = state["context"]

    print("\n==============================")
    print("GENERATING RESEARCH ANSWER")
    print("==============================")

    answer = generate_answer(
        question,
        context
    )

    return {
        "answer": answer
    }


# ==================================================
# NODE 6
# DIRECT ANSWER
# ==================================================

def direct_answer(
    state: ResearchState
):

    question = state["question"]

    print("\n==============================")
    print("GENERATING DIRECT ANSWER")
    print("==============================")

    answer = generate_general_answer(
        question
    )

    return {
        "answer": answer
    }


# ==================================================
# ROUTER 1
# ==================================================

def route_after_decision(
    state: ResearchState
):

    if state[
        "is_summary_request"
    ]:

        return "summary"

    if state[
        "needs_retrieval"
    ]:

        return "rewrite"

    return "direct_answer"


# ==================================================
# ROUTER 2
# ==================================================

def route_after_retrieval(
    state: ResearchState
):

    quality = state[
        "retrieval_quality"
    ]

    attempt = state[
        "retrieval_attempt"
    ]

    print("\n==============================")
    print("AGENT QUALITY CHECK")
    print("==============================")

    print(
        f"Retrieval quality: "
        f"{quality:.4f}"
    )

    if quality >= 0:

        print(
            "Context looks sufficiently relevant."
        )

        return "answer"

    if attempt < 2:

        print(
            "Context is weak."
        )

        print(
            "Agent is retrying retrieval..."
        )

        return "retry"

    print(
        "Maximum retrieval attempts reached."
    )

    return "answer"


# ==================================================
# BUILD LANGGRAPH
# ==================================================

graph_builder = StateGraph(
    ResearchState
)


# ==================================================
# ADD NODES
# ==================================================

graph_builder.add_node(
    "decide",
    decide_retrieval
)

graph_builder.add_node(
    "rewrite",
    rewrite_user_query
)

graph_builder.add_node(
    "retrieve",
    retrieve_documents
)

graph_builder.add_node(
    "summary",
    summarize_documents
)

graph_builder.add_node(
    "answer",
    answer_from_documents
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)


# ==================================================
# START → DECIDE
# ==================================================

graph_builder.add_edge(
    START,
    "decide"
)


# ==================================================
# DECIDE → SUMMARY / REWRITE / DIRECT
# ==================================================

graph_builder.add_conditional_edges(

    "decide",

    route_after_decision,

    {

        "summary":
            "summary",

        "rewrite":
            "rewrite",

        "direct_answer":
            "direct_answer"
    }
)


# ==================================================
# REWRITE → RETRIEVE
# ==================================================

graph_builder.add_edge(
    "rewrite",
    "retrieve"
)


# ==================================================
# RETRIEVE → ANSWER / RETRY
# ==================================================

graph_builder.add_conditional_edges(

    "retrieve",

    route_after_retrieval,

    {

        "answer":
            "answer",

        "retry":
            "retrieve"
    }
)


# ==================================================
# SUMMARY → END
# ==================================================

graph_builder.add_edge(
    "summary",
    END
)


# ==================================================
# ANSWER → END
# ==================================================

graph_builder.add_edge(
    "answer",
    END
)


# ==================================================
# DIRECT ANSWER → END
# ==================================================

graph_builder.add_edge(
    "direct_answer",
    END
)


# ==================================================
# COMPILE
# ==================================================

research_agent = graph_builder.compile()