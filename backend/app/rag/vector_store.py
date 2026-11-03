import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


load_dotenv()


QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
)


COLLECTION_NAME = "research_documents"


def create_collection():

    collections = client.get_collections().collections

    existing_names = [
        collection.name
        for collection in collections
    ]

    if COLLECTION_NAME not in existing_names:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

        print(
            f"Collection '{COLLECTION_NAME}' created."
        )

    else:

        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )


def add_chunks(chunks, embeddings):

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=chunk
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(
        f"{len(points)} chunks stored in Qdrant."
    )


def search(query_embedding, top_k=5):

    results = (
        client
        .query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k
        )
        .points
    )

    return results


def delete_collection():

    client.delete_collection(
        collection_name=COLLECTION_NAME
    )

    print(
        f"Collection '{COLLECTION_NAME}' deleted."
    )