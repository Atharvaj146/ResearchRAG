from app.rag.agent import research_agent


# =========================================
# TEST 1
# RESEARCH QUESTION
# =========================================

question_1 = (
    "What was the success count of the proposed "
    "method compared with the two baseline methods?"
)


state_1 = {
    "question": question_1,
    "needs_retrieval": False,
    "context": "",
    "answer": "",
    "retrieval_attempt": 0,
    "retrieval_quality": 0.0
}


result_1 = research_agent.invoke(
    state_1
)


print("\n==============================")
print("TEST 1 RESULT")
print("==============================")

print("\nQUESTION:")
print(result_1["question"])

print("\nANSWER:")
print(result_1["answer"])


# =========================================
# TEST 2
# NORMAL CONVERSATION
# =========================================

question_2 = "Hello, how are you?"


state_2 = {
    "question": question_2,
    "needs_retrieval": False,
    "context": "",
    "answer": "",
    "retrieval_attempt": 0,
    "retrieval_quality": 0.0
}


result_2 = research_agent.invoke(
    state_2
)


print("\n==============================")
print("TEST 2 RESULT")
print("==============================")

print("\nQUESTION:")
print(result_2["question"])

print("\nANSWER:")
print(result_2["answer"])