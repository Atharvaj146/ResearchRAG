from app.rag.evaluation import (
    evaluate_retrieval,
    evaluate_answer
)


# ==================================================
# RETRIEVAL TEST
# ==================================================

retrieved_documents = [

    "paper_a.pdf",
    "paper_b.pdf",
    "paper_c.pdf",
    "paper_d.pdf",
    "paper_e.pdf"
]

expected_document = "paper_b.pdf"


retrieval_result = evaluate_retrieval(
    retrieved_documents,
    expected_document
)


print("\n==============================")
print("RETRIEVAL EVALUATION")
print("==============================")

print(
    "Hit@1:",
    retrieval_result["hit_at_1"]
)

print(
    "Hit@3:",
    retrieval_result["hit_at_3"]
)

print(
    "Hit@5:",
    retrieval_result["hit_at_5"]
)


# ==================================================
# ANSWER TEST
# ==================================================

answer = """
The proposed method achieved 47 successful
assignments out of 50 experiments. The method
uses task decomposition, allocation and sequential
planning with feedback.
"""


expected_keywords = [
    "47",
    "50",
    "task decomposition",
    "allocation",
    "sequential planning"
]


answer_result = evaluate_answer(
    answer,
    expected_keywords
)


print("\n==============================")
print("ANSWER EVALUATION")
print("==============================")

print(
    "Keyword match:",
    answer_result[
        "keyword_match_percentage"
    ],
    "%"
)