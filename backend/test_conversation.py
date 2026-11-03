from app.rag.rag_pipeline import ask_question


chat_history = []


# =========================================
# QUESTION 1
# =========================================

question_1 = (
    "What was the success count of the proposed "
    "method compared with the two baseline methods?"
)


result_1 = ask_question(
    question_1,
    chat_history
)


print("\n==============================")
print("QUESTION 1")
print("==============================")

print(result_1["question"])

print("\nSEARCH QUERY:")
print(result_1["search_query"])

print("\nANSWER:")
print(result_1["answer"])

print("\nSOURCES:")

for source in result_1["sources"]:
    print(
        f"- {source['document_name']} "
        f"| Page {source['page_number']} "
        f"| Score {source['score']:.4f}"
    )


# =========================================
# ADD QUESTION 1 TO CHAT HISTORY
# =========================================

chat_history.append(
    {
        "role": "user",
        "content": question_1
    }
)

chat_history.append(
    {
        "role": "assistant",
        "content": result_1["answer"]
    }
)


# =========================================
# QUESTION 2
# =========================================

question_2 = (
    "How did it compare with the random approach?"
)


result_2 = ask_question(
    question_2,
    chat_history
)


print("\n==============================")
print("QUESTION 2")
print("==============================")

print(result_2["question"])

print("\nSEARCH QUERY:")
print(result_2["search_query"])

print("\nANSWER:")
print(result_2["answer"])

print("\nSOURCES:")

for source in result_2["sources"]:
    print(
        f"- {source['document_name']} "
        f"| Page {source['page_number']} "
        f"| Score {source['score']:.4f}"
    )