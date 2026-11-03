from app.rag.rag_pipeline import ask_question


question = "What was the success count of the proposed method compared with the two baseline methods?"

result = ask_question(question)


print("\nQUESTION:")
print(result["question"])


print("\nANSWER:")
print(result["answer"])


print("\nSOURCES:")

for source in result["sources"]:

    print(
        f"- {source['document_name']} "
        f"| Page {source['page_number']} "
        f"| Score {source['score']:.4f}"
    )