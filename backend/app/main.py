import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag.agent import research_agent
from app.rag.ingestion import process_pdf
from app.rag.embeddings import create_embeddings
from app.rag.vector_store import create_collection, add_chunks


app = FastAPI(
    title="ResearchRAG API",
    description="Agentic Multi-Document Research Assistant",
    version="1.0.0"
)


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# UPLOAD DIRECTORY
# ==================================================

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ==================================================
# CURRENT SESSION
# ==================================================

SESSION_ID = str(
    uuid.uuid4()
)

print("\n===================================")
print("ResearchRAG Session Started")
print("Session ID:", SESSION_ID)
print("===================================\n")


# ==================================================
# REQUEST MODELS
# ==================================================

class QuestionRequest(BaseModel):

    question: str

    chat_history: list = []


# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():

    return {
        "message":
            "ResearchRAG backend is running",

        "session_id":
            SESSION_ID
    }


# ==================================================
# UPLOAD PDF
# ==================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    document_id = str(
        uuid.uuid4()
    )

    safe_filename = os.path.basename(
        file.filename
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{document_id}_{safe_filename}"
    )

    try:

        # ------------------------------------------
        # Save PDF
        # ------------------------------------------

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # ------------------------------------------
        # Extract + chunk
        # ------------------------------------------

        chunks = process_pdf(
            file_path=file_path,
            document_name=safe_filename,
            document_id=document_id,
            session_id=SESSION_ID
        )


        if not chunks:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No readable text was found "
                    "in this PDF."
                )
            )


        # ------------------------------------------
        # Embeddings
        # ------------------------------------------

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = create_embeddings(
            texts
        )


        # ------------------------------------------
        # Qdrant
        # ------------------------------------------

        create_collection()

        add_chunks(
            chunks,
            embeddings
        )


        print("\nDOCUMENT UPLOADED")
        print("----------------------------")

        print(
            "Session:",
            SESSION_ID
        )

        print(
            "Document:",
            safe_filename
        )

        print(
            "Document ID:",
            document_id
        )

        print(
            "Chunks:",
            len(chunks)
        )

        print("----------------------------")


        return {

            "message":
                "Document uploaded successfully.",

            "session_id":
                SESSION_ID,

            "document_id":
                document_id,

            "document_name":
                safe_filename,

            "chunks_created":
                len(chunks)
        }


    except HTTPException:

        raise


    except Exception as error:

        print(
            "UPLOAD ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==================================================
# ASK QUESTION
# ==================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    # ----------------------------------------------
    # Use conversation supplied by frontend
    # ----------------------------------------------

    chat_history = request.chat_history


    print("\n==============================")
    print("NEW QUESTION")
    print("==============================")

    print(
        "Question:",
        question
    )

    print(
        "Previous messages:",
        len(chat_history)
    )


    # ----------------------------------------------
    # Build LangGraph state
    # ----------------------------------------------

    state = {

        "question":
            question,

        "session_id":
            SESSION_ID,

        "chat_history":
            chat_history,

        "rewritten_query":
            "",

        "needs_retrieval":
            False,

        "is_summary_request":
            False,

        "is_ambiguous_question":
            False,

        "context":
            "",

        "answer":
            "",

        "retrieval_attempt":
            0,

        "retrieval_quality":
            0.0,

        "sources":
            []
    }


    # ----------------------------------------------
    # Run agent
    # ----------------------------------------------

    result = research_agent.invoke(
        state
    )


    answer = result[
        "answer"
    ]


    # ----------------------------------------------
    # Return response
    # ----------------------------------------------

    return {

        "question":
            question,

        "answer":
            answer,

        "sources":
            result.get(
                "sources",
                []
            ),

        "rewritten_query":
            result.get(
                "rewritten_query",
                question
            ),

        "session_id":
            SESSION_ID
    }