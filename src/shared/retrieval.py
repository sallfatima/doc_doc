"""Manage the configuration of various retrievers with optimized namespace approach.

This module provides functionality to create and manage retrievers using
a single Pinecone index with namespaces for better performance and cost efficiency.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Dict, Any
import logging
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
def _get_or_create_pinecone_vs(index_name: str, embedding_model: Embeddings, namespace: str = "") -> "PineconeVectorStore":
    """Create or connect to Pinecone vectorstore with namespace support."""
    pinecone_client = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ.get("PINECONE_ENVIRONMENT", "")
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
        embedding=embedding_model,
        namespace=namespace
    )


class SmartRetriever:
    """
    Intelligent retriever optimized for async operations.
    No more blocking calls!
    """
    
    def __init__(
        self,
        embedding_model: Embeddings,
        index_name: str,
        search_kwargs: Dict[str, Any]
    ):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.search_kwargs = search_kwargs
        
        # Create vectorstores for each namespace - LAZY LOADING
        self._text_store = None
        self._image_store = None
    
    def _get_text_store(self):
        """Lazy loading of text store."""
        if self._text_store is None:
            self._text_store = _get_or_create_pinecone_vs(
                self.index_name, self.embedding_model, namespace=""
            )
        return self._text_store
    
    def _get_image_store(self):
        """Lazy loading of image store."""
        if self._image_store is None:
            self._image_store = _get_or_create_pinecone_vs(
                self.index_name, self.embedding_model, namespace="images"
            )
        return self._image_store
    
    def _is_image_focused_query(self, query: str) -> bool:
        """Determine if the query is primarily image-focused."""
        image_keywords = [
            'image', 'images', 'photo', 'photos', 'picture', 'pictures',
            'diagram', 'diagrams', 'figure', 'figures', 'chart', 'charts',
            'graph', 'graphs', 'screenshot', 'screenshots', 'visual', 'visuals',
            'illustration', 'illustrations', 'schéma', 'schémas', 'graphique',
            'montrer', 'afficher', 'voir', 'regarder', 'show', 'display', 'view'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in image_keywords)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async retrieval with proper thread handling for blocking operations.
        """
        is_image_focused = self._is_image_focused_query(query)
        
        # Adaptive search strategy
        if is_image_focused:
            text_k = max(1, self.search_kwargs.get("k", 4) // 3)
            image_k = self.search_kwargs.get("k", 6)
        else:
            text_k = self.search_kwargs.get("k", 8)
            image_k = max(1, self.search_kwargs.get("k", 3))
        
        all_docs = []
        
        try:
            # Use asyncio.to_thread for blocking operations
            async def search_text():
                if text_k > 0:
                    text_store = self._get_text_store()
                    text_retriever = text_store.as_retriever(
                        search_kwargs={"k": text_k, **{k: v for k, v in self.search_kwargs.items() if k != "k"}}
                    )
                    # Use to_thread for the blocking call
                    docs = await asyncio.to_thread(text_retriever.invoke, query)
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["source_type"] = "text"
                        doc.metadata["namespace"] = ""
                    
                    return docs
                return []
            
            async def search_images():
                if image_k > 0:
                    image_store = self._get_image_store()
                    image_retriever = image_store.as_retriever(
                        search_kwargs={"k": image_k, **{k: v for k, v in self.search_kwargs.items() if k != "k"}}
                    )
                    # Use to_thread for the blocking call
                    docs = await asyncio.to_thread(image_retriever.invoke, query)
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["source_type"] = "image"
                        doc.metadata["namespace"] = "images"
                    
                    return docs
                return []
            
            # Run searches concurrently
            text_docs, image_docs = await asyncio.gather(
                search_text(),
                search_images(),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(text_docs, Exception):
                logging.error(f"Text search failed: {text_docs}")
                text_docs = []
            
            if isinstance(image_docs, Exception):
                logging.error(f"Image search failed: {image_docs}")
                image_docs = []
            
            all_docs.extend(text_docs)
            all_docs.extend(image_docs)
        
        except Exception as e:
            logging.error(f"Error during smart retrieval: {e}")
        
        logging.info(f"SmartRetriever: {len([d for d in all_docs if d.metadata.get('source_type')=='text'])} text, "
                    f"{len([d for d in all_docs if d.metadata.get('source_type')=='image'])} images")
        
        return all_docs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Sync wrapper - should be avoided in async context."""
        return asyncio.run(self.aget_relevant_documents(query))
    
    async def ainvoke(self, query: str) -> List[Document]:
        """Async invoke method for compatibility."""
        return await self.aget_relevant_documents(query)
    
    def invoke(self, query: str) -> List[Document]:
        """Sync invoke - will use asyncio.to_thread when called from async context."""
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # If we are, this is problematic - log a warning
            logging.warning("Sync invoke called from async context - consider using ainvoke")
            return asyncio.run_coroutine_threadsafe(
                self.aget_relevant_documents(query), loop
            ).result()
        except RuntimeError:
            # No event loop running, safe to use sync
            return self.get_relevant_documents(query)

## Indexer functions (for compatibility with existing indexing code)
@asynccontextmanager
async def make_text_indexer(
    configuration: BaseConfiguration, 
    embedding_model: Embeddings
) -> AsyncGenerator[PineconeVectorStore, None]:
    """Text indexer using default namespace."""
    vectorstore = _get_or_create_pinecone_vs(
        os.environ["PINECONE_INDEX_NAME"], 
        embedding_model,
        namespace=""  # Default namespace for text
    )
    yield vectorstore


@asynccontextmanager
async def make_image_indexer(
    config: RunnableConfig,
) -> AsyncGenerator[PineconeVectorStore, None]:
    """Image indexer using 'images' namespace - COMPATIBLE WITH YOUR EXISTING CODE."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    vectorstore = _get_or_create_pinecone_vs(
        os.environ["PINECONE_INDEX_NAME"],  # Same index!
        embedding_model,
        namespace="images"  # Separate namespace for images
    )
    yield vectorstore


## Main retriever function
@asynccontextmanager
async def make_retriever(
    config: RunnableConfig,
) -> AsyncGenerator[SmartRetriever, None]:
    """Create the optimized SmartRetriever."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    
    smart_retriever = SmartRetriever(
        embedding_model=embedding_model,
        index_name=os.environ["PINECONE_INDEX_NAME"],
        search_kwargs=configuration.search_kwargs
    )
    
    yield smart_retriever