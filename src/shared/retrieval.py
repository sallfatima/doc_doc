"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Tuple, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from shared.configuration import BaseConfiguration
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

## Encoder constructors
def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Internal Pinecone utility
def _get_or_create_pinecone_vs(index_name: str, embedding_model: Embeddings) -> "PineconeVectorStore":

    pinecone_client = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )

    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )


## Retriever constructors
@asynccontextmanager
def make_text_indexer( configuration: BaseConfiguration, embedding_model: Embeddings) -> AsyncGenerator[Tuple[VectorStoreRetriever, object], None]:
    """Retriever on text index."""
    vectorstore = _get_or_create_pinecone_vs(
        os.environ["PINECONE_INDEX_NAME"], embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs=configuration.search_kwargs)
    yield retriever, vectorstore

@asynccontextmanager
def make_image_indexer(config: RunnableConfig,) -> AsyncGenerator[object, None]:
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    vectorstore = _get_or_create_pinecone_vs(
        os.environ["PINECONE_INDEX_IMAGE"], embedding_model
    )
    yield vectorstore


class MultiIndexRetriever(VectorStoreRetriever):
    def __init__(
        self,
        embed_fn: Embeddings,
        text_store: object,
        image_store: object,
        k: int = 10
    ):
        self.embed_fn = embed_fn
        self.text_store = text_store
        self.image_store = image_store
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_vec = self.embed_fn.embed_query(query)

        docs_text = self.text_store.similarity_search_by_vector(query_vec, k=self.k, score=True)
        docs_image = self.image_store.similarity_search_by_vector(query_vec, k=self.k, score=True)

        all_docs = []
        for score, doc in docs_text:
            doc.metadata["source_index"] = "text"
            doc.metadata["score"] = score
            all_docs.append((score, doc))

        for score, doc in docs_image:
            doc.metadata["source_index"] = "image"
            doc.metadata["score"] = score
            all_docs.append((score, doc))

        all_docs.sort(key=lambda tup: tup[0])  # sort by distance (lower is better)
        return [doc for _, doc in all_docs[:self.k]]


@asynccontextmanager
def make_retriever(
    config: RunnableConfig,
) -> AsyncGenerator[Tuple[VectorStoreRetriever, object], None]:
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)

    match configuration.retriever_provider:
        case "pinecone":
            text_store = _get_or_create_pinecone_vs(os.environ["PINECONE_INDEX_NAME"], embedding_model)
            image_store = _get_or_create_pinecone_vs(os.environ["PINECONE_INDEX_IMAGE"], embedding_model)
            multi_retriever = MultiIndexRetriever(embed_fn=embedding_model, text_store=text_store, image_store=image_store, k=configuration.search_kwargs.get("k", 10))
            yield multi_retriever, (text_store, image_store)

        case _:
            raise ValueError(
                "❌ Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
