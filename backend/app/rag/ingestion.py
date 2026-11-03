import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf(
    file_path,
    document_name,
    document_id,
    session_id
):
    document = fitz.open(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        if not text.strip():
            continue

        page_chunks = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(page_chunks):

            chunks.append({
                "session_id": session_id,
                "document_id": document_id,
                "document_name": document_name,
                "page_number": page_number + 1,
                "chunk_id": (
                    f"{document_id}_"
                    f"{page_number + 1}_"
                    f"{chunk_index}"
                ),
                "text": chunk_text
            })

    document.close()

    return chunks