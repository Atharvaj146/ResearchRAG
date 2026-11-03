from types import SimpleNamespace

from rank_bm25 import BM25Okapi

from app.rag.vector_store import (
    client,
    COLLECTION_NAME,
    search
)


def get_session_chunks(session_id):

    all_chunks = []

    offset = None

    while True:

        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in points:

            if point.payload.get(
                "session_id"
            ) == session_id:

                all_chunks.append(point)

        if next_offset is None:
            break

        offset = next_offset

    return all_chunks


def get_documents_for_session(
    session_id
):

    session_chunks = get_session_chunks(
        session_id
    )

    documents = {}

    for point in session_chunks:

        payload = point.payload

        document_id = payload[
            "document_id"
        ]

        if document_id not in documents:

            documents[document_id] = {

                "document_id":
                    document_id,

                "document_name":
                    payload[
                        "document_name"
                    ],

                "chunks":
                    []
            }

        documents[
            document_id
        ]["chunks"].append(point)

    return list(
        documents.values()
    )


def tokenize(text):

    return text.lower().split()


def normalize_scores(scores):

    if len(scores) == 0:

        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:

        return [
            1.0
            for _ in scores
        ]

    return [

        (score - minimum) /
        (maximum - minimum)

        for score in scores
    ]


def hybrid_search(
    query_embedding,
    query,
    session_id,
    top_k=5
):

    session_chunks = get_session_chunks(
        session_id
    )

    if not session_chunks:

        print(
            "No documents found for session:",
            session_id
        )

        return []


    # ----------------------------------------------
    # Vector search
    # ----------------------------------------------

    vector_results = search(
        query_embedding,
        top_k=20
    )


    # ----------------------------------------------
    # Session filtering
    # ----------------------------------------------

    vector_results = [

        result

        for result in vector_results

        if result.payload.get(
            "session_id"
        ) == session_id
    ]


    # ----------------------------------------------
    # BM25
    # ----------------------------------------------

    documents = [

        point.payload["text"]

        for point in session_chunks
    ]

    tokenized_documents = [

        tokenize(document)

        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    tokenized_query = tokenize(
        query
    )

    bm25_scores = bm25.get_scores(
        tokenized_query
    )


    # ----------------------------------------------
    # Normalize
    # ----------------------------------------------

    vector_scores = [

        result.score

        for result in vector_results
    ]

    normalized_vector_scores = (
        normalize_scores(
            vector_scores
        )
    )

    normalized_bm25_scores = (
        normalize_scores(
            bm25_scores
        )
    )


    # ----------------------------------------------
    # Combine
    # ----------------------------------------------

    combined_results = {}


    # Vector results

    for index, result in enumerate(
        vector_results
    ):

        point_id = str(
            result.id
        )

        combined_results[
            point_id
        ] = {

            "payload":
                result.payload,

            "score":
                0.5 *
                normalized_vector_scores[
                    index
                ],

            "vector_score":
                normalized_vector_scores[
                    index
                ],

            "bm25_score":
                0.0
        }


    # BM25 results

    for index, point in enumerate(
        session_chunks
    ):

        point_id = str(
            point.id
        )

        bm25_score = (
            normalized_bm25_scores[
                index
            ]
        )


        if point_id not in combined_results:

            combined_results[
                point_id
            ] = {

                "payload":
                    point.payload,

                "score":
                    0.5 *
                    bm25_score,

                "vector_score":
                    0.0,

                "bm25_score":
                    bm25_score
            }

        else:

            combined_results[
                point_id
            ]["score"] += (
                0.5 *
                bm25_score
            )

            combined_results[
                point_id
            ]["bm25_score"] = (
                bm25_score
            )


    # ----------------------------------------------
    # Sort
    # ----------------------------------------------

    sorted_results = sorted(

        combined_results.values(),

        key=lambda item:
            item["score"],

        reverse=True
    )


    # ----------------------------------------------
    # Top K
    # ----------------------------------------------

    final_results = []

    for result in sorted_results[
        :top_k
    ]:

        final_results.append(

            SimpleNamespace(

                payload=
                    result["payload"],

                score=
                    result["score"]
            )
        )


    print(
        f"Session retrieval: "
        f"{len(session_chunks)} chunks available"
    )

    print(
        f"Hybrid results: "
        f"{len(final_results)}"
    )

    return final_results