import { useState } from "react";


function App() {

  const [question, setQuestion] = useState("");

  const [messages, setMessages] = useState([]);

  const [sources, setSources] = useState([]);

  const [rewrittenQuery, setRewrittenQuery] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  const [uploading, setUploading] =
    useState(false);

  const [uploadMessage, setUploadMessage] =
    useState("");

  const [error, setError] =
    useState("");


  // ==================================================
  // UPLOAD DOCUMENTS
  // ==================================================

  const uploadDocuments = async (event) => {

    const files = Array.from(
      event.target.files
    );

    if (files.length === 0) {
      return;
    }


    const invalidFiles = files.filter(
      (file) =>
        !file.name
          .toLowerCase()
          .endsWith(".pdf")
    );


    if (invalidFiles.length > 0) {

      setError(
        "Only PDF files are supported."
      );

      return;
    }


    setUploading(true);

    setUploadMessage("");

    setError("");


    const successfulUploads = [];


    try {

      for (const file of files) {

        setUploadMessage(
          `Indexing ${file.name}...`
        );


        const formData =
          new FormData();

        formData.append(
          "file",
          file
        );


        const response =
          await fetch(
            "http://127.0.0.1:8000/upload",
            {
              method: "POST",
              body: formData,
            }
          );


        const data =
          await response.json();


        if (!response.ok) {

          throw new Error(
            `${file.name}: ${
              data.detail ||
              "Upload failed"
            }`
          );
        }


        successfulUploads.push({
          name:
            data.document_name,

          chunks:
            data.chunks_created,
        });
      }


      const summary =
        successfulUploads
          .map(
            (file) =>
              `${file.name} (${file.chunks} chunks)`
          )
          .join(", ");


      setUploadMessage(
        `${successfulUploads.length} document(s) indexed successfully: ${summary}`
      );


    } catch (error) {

      setError(
        error.message
      );


    } finally {

      setUploading(false);

      event.target.value = "";
    }
  };


  // ==================================================
  // ASK QUESTION
  // ==================================================

  const askQuestion = async () => {

    if (!question.trim()) {
      return;
    }


    setLoading(true);

    setError("");

    setSources([]);

    setRewrittenQuery("");


    const currentQuestion =
      question.trim();


    try {

      const response =
        await fetch(
          "http://127.0.0.1:8000/ask",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({

              question:
                currentQuestion,

              chat_history:
                messages.map(
                  (message) => ({

                    role:
                      message.role,

                    content:
                      message.content,
                  })
                ),
            }),
          }
        );


      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Failed to get answer"
        );
      }


      setMessages(
        (previousMessages) => [

          ...previousMessages,

          {
            role:
              "user",

            content:
              currentQuestion,
          },

          {
            role:
              "assistant",

            content:
              data.answer || "",
          },
        ]
      );


      setSources(
        data.sources || []
      );


      setRewrittenQuery(
        data.rewritten_query ||
        currentQuestion
      );


      setQuestion("");


    } catch (error) {

      setError(
        error.message
      );


    } finally {

      setLoading(false);
    }
  };


  // ==================================================
  // ENTER KEY
  // ==================================================

  const handleKeyDown = (
    event
  ) => {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {

      event.preventDefault();

      askQuestion();
    }
  };


  // ==================================================
  // NEW CHAT
  // ==================================================

  const startNewChat = () => {

    setMessages([]);

    setSources([]);

    setRewrittenQuery("");

    setQuestion("");

    setError("");
  };


  // ==================================================
  // UI
  // ==================================================

  return (

    <div className="app">


      {/* ============================================
          HEADER
      ============================================ */}

      <header className="header">

        <div className="header-content">

          <div>

            <h1>
              ResearchRAG
            </h1>

            <p>
              Agentic Multi-Document
              Research Assistant
            </p>

          </div>


          {messages.length > 0 && (

            <button
              className="new-chat-button"
              onClick={
                startNewChat
              }
            >
              New Chat
            </button>

          )}

        </div>

      </header>


      {/* ============================================
          MAIN
      ============================================ */}

      <main className="container">


        {/* ==========================================
            HERO
        ========================================== */}

        {messages.length === 0 && (

          <section className="hero">

            <span className="badge">
              AI Research Assistant
            </span>

            <h2>

              Research smarter.
              <br />

              Ask your documents anything.

            </h2>

            <p>

              Upload multiple research papers
              and have a conversation with your
              documents using hybrid retrieval,
              reranking and agentic reasoning.

            </p>

          </section>

        )}


        {/* ==========================================
            UPLOAD
        ========================================== */}

        <section className="upload-card">

          <h3>
            Upload Research Documents
          </h3>

          <p>
            Select one or multiple PDF research
            papers to add them to your knowledge base.
          </p>


          <label className="upload-button">

            {uploading
              ? "Indexing Documents..."
              : "Choose PDF Documents"
            }


            <input
              type="file"
              accept=".pdf"
              multiple
              onChange={
                uploadDocuments
              }
              disabled={
                uploading
              }
              hidden
            />

          </label>


          {uploadMessage && (

            <div className="success">

              {uploadMessage}

            </div>

          )}

        </section>


        {/* ==========================================
            CHAT
        ========================================== */}

        <section className="chat-card">


          {/* ========================================
              CONVERSATION
          ======================================== */}

          {messages.length > 0 && (

            <div className="conversation">

              {messages.map(
                (message, index) => (

                  <div
                    key={index}
                    className={
                      message.role ===
                      "user"
                        ? "message user-message"
                        : "message assistant-message"
                    }
                  >

                    <div className="message-label">

                      {message.role ===
                      "user"
                        ? "You"
                        : "ResearchRAG"
                      }

                    </div>


                    <div className="message-content">

                      {message.content}

                    </div>

                  </div>

                )
              )}

            </div>

          )}


          {/* ========================================
              REWRITTEN QUERY
          ======================================== */}

          {rewrittenQuery && (
            <div className="query-info">

              <strong>
                Search query used:
              </strong>

              <span>
                {rewrittenQuery}
              </span>

            </div>
          )}


          {/* ========================================
              QUESTION INPUT
          ======================================== */}

          <label className="question-label">

            {messages.length > 0
              ? "Continue the conversation"
              : "Research Question"
            }

          </label>


          <textarea

            value={question}

            onChange={(event) =>
              setQuestion(
                event.target.value
              )
            }

            onKeyDown={
              handleKeyDown
            }

            placeholder={
              messages.length > 0
                ? "Ask a follow-up question..."
                : "Ask a question about your research papers..."
            }

            rows={4}

          />


          <button

            onClick={
              askQuestion
            }

            disabled={
              loading ||
              !question.trim()
            }

          >

            {loading
              ? "Researching..."
              : "Ask ResearchRAG"
            }

          </button>


          {/* ========================================
              ERROR
          ======================================== */}

          {error && (

            <div className="error">

              <strong>
                Error:
              </strong>{" "}

              {error}

            </div>

          )}


          {/* ========================================
              SOURCES
          ======================================== */}

          {sources.length > 0 && (

            <div className="sources-section">

              <h3>
                Sources
              </h3>


              {sources.map(
                (source, index) => (

                  <div
                    className="source-card"
                    key={
                      source.chunk_id ||
                      index
                    }
                  >

                    <div>

                      <strong>
                        {
                          source.document_name
                        }
                      </strong>

                      <span>
                        Page{" "}
                        {
                          source.page_number
                        }
                      </span>

                    </div>


                    <small>

                      Chunk:{" "}
                      {
                        source.chunk_id
                      }

                    </small>

                  </div>

                )
              )}

            </div>

          )}

        </section>

      </main>

    </div>
  );
}


export default App;