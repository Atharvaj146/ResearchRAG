from sentence_transformers import CrossEncoder


# Load the reranking model
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_results(
    query,
    results,
    top_k=5
):

    # Create question-document pairs
    pairs = []

    for result in results:

        text = result.payload["text"]

        pairs.append(
            (query, text)
        )


    # Calculate relevance scores
    scores = reranker_model.predict(
        pairs
    )


    # Combine results with scores
    ranked_results = []

    for result, score in zip(
        results,
        scores
    ):

        ranked_results.append(
            {
                "result": result,
                "score": float(score)
            }
        )


    # Sort by reranker score
    ranked_results.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    # Return top K
    return ranked_results[:top_k]