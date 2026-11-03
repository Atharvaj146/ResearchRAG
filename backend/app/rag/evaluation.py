def calculate_hit_at_k(
    retrieved_documents,
    expected_document,
    k
):
    """
    Checks whether the expected document
    appears within the first K retrieved documents.
    """

    top_documents = retrieved_documents[:k]

    for document in top_documents:

        if document == expected_document:

            return 1

    return 0


def calculate_keyword_match(
    answer,
    expected_keywords
):
    """
    Calculates the percentage of expected
    keywords found in the generated answer.
    """

    if not expected_keywords:

        return 0.0

    answer_lower = answer.lower()

    matched = 0

    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:

            matched += 1

    return matched / len(
        expected_keywords
    )


def evaluate_answer(
    answer,
    expected_keywords
):
    """
    Basic answer evaluation.
    """

    keyword_score = calculate_keyword_match(
        answer,
        expected_keywords
    )

    return {
        "keyword_match_score":
            round(
                keyword_score,
                3
            ),

        "keyword_match_percentage":
            round(
                keyword_score * 100,
                2
            )
    }


def evaluate_retrieval(
    retrieved_documents,
    expected_document
):
    """
    Calculates Hit@1, Hit@3 and Hit@5.
    """

    return {

        "hit_at_1":
            calculate_hit_at_k(
                retrieved_documents,
                expected_document,
                1
            ),

        "hit_at_3":
            calculate_hit_at_k(
                retrieved_documents,
                expected_document,
                3
            ),

        "hit_at_5":
            calculate_hit_at_k(
                retrieved_documents,
                expected_document,
                5
            )
    }