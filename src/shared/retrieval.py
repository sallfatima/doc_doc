import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
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


def get_pinecone_index_name() -> str:
    """Get the correct Pinecone index name from environment variables."""
    # Try different environment variable names
    index_name = (
        os.environ.get("PINECONE_INDEX_NAME") or 
        os.environ.get("PINECONE_INDEX") or 
        "index-text"  # Default to index-text for new installations
    )
    print(f"🔧 Using Pinecone index: {index_name}")
    return index_name


async def _get_or_create_pinecone_vs_async(index_name: str, embedding_model: Embeddings, namespace: str = "") -> "PineconeVectorStore":
    """Create or connect to Pinecone vectorstore with namespace support - FIXED VERSION."""
    
    def _sync_pinecone_operations():
        """Synchronous Pinecone operations to execute in a separate thread."""
        try:
            pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            
            # List existing indexes
            existing_indexes = pinecone_client.list_indexes().names()
            print(f"📋 Available indexes: {existing_indexes}")
            print(f"🔍 Looking for index: {index_name}")
            
            if index_name not in existing_indexes:
                print(f"🔄 Creating Pinecone index: {index_name}")
                
                # Create the index synchronously here
                pinecone_client.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws", 
                        region="us-east-1"
                    )
                )
                
                print(f"✅ Pinecone index created: {index_name}")
                
                # Wait for the index to be ready
                import time
                print("⏳ Waiting for index to be ready...")
                time.sleep(15)  # Give it more time to initialize properly
                
                # Verify the index was created
                updated_indexes = pinecone_client.list_indexes().names()
                if index_name in updated_indexes:
                    print(f"✅ Index {index_name} is now available")
                else:
                    print(f"⚠️ Index {index_name} creation may still be in progress")
                
            else:
                print(f"✅ Pinecone index exists: {index_name}")
            
            return pinecone_client
            
        except Exception as e:
            print(f"❌ Error in Pinecone operations: {e}")
            # Print more details about the error
            import traceback
            traceback.print_exc()
            raise
    
    def _sync_create_vectorstore(pc_client):
        """Create the vectorstore in a separate thread."""
        try:
            print(f"🔧 Creating vectorstore for index: {index_name}, namespace: '{namespace}'")
            
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embedding_model,
                namespace=namespace
            )
            
            print(f"✅ Vectorstore created successfully")
            return vectorstore
            
        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            raise
    
    # Execute Pinecone operations in separate threads
    pc_client = await asyncio.to_thread(_sync_pinecone_operations)
    vectorstore = await asyncio.to_thread(_sync_create_vectorstore, pc_client)
    
    return vectorstore


## INDEXERS FOR INDEXING
@asynccontextmanager
async def make_text_indexer(
    configuration: BaseConfiguration, 
    embedding_model: Embeddings
) -> AsyncGenerator[PineconeVectorStore, None]:
    """
    Text indexer for indexing in default namespace - FIXED VERSION
    """
    index_name = get_pinecone_index_name()
    vectorstore = await _get_or_create_pinecone_vs_async(
        index_name, 
        embedding_model,
        namespace=""  # Default namespace for text
    )
    print(f"📝 Text indexer created - Index: {index_name}, Namespace: '' (default)")
    yield vectorstore


@asynccontextmanager
async def make_image_indexer(
    config,  # Peut être RunnableConfig ou IndexConfiguration
) -> AsyncGenerator[PineconeVectorStore, None]:
    """
    Image indexer for indexing in 'images' namespace - FIXED VERSION
    Accepte maintenant les deux types de configuration
    """
    # CORRECTION: Gérer les deux types de configuration
    if hasattr(config, 'embedding_model'):
        # C'est déjà un IndexConfiguration ou BaseConfiguration
        configuration = config
        print("🔧 Using direct configuration object")
    else:
        # C'est un RunnableConfig, le convertir
        configuration = BaseConfiguration.from_runnable_config(config)
        print("🔧 Converting from RunnableConfig")
    
    embedding_model = make_text_encoder(configuration.embedding_model)
    index_name = get_pinecone_index_name()
    
    vectorstore = await _get_or_create_pinecone_vs_async(
        index_name,  # Same index as text!
        embedding_model,
        namespace="images"  # Separate namespace for images
    )
    print(f"🖼️ Image indexer created - Index: {index_name}, Namespace: 'images'")
    yield vectorstore

# SMART RETRIEVER FOR SEARCH (not for indexing)
class SmartRetriever:
    """
    SmartRetriever for search (not indexing)
    Uses both namespaces to retrieve text + images
    """
    
    def __init__(self, embedding_model: Embeddings, index_name: str, search_kwargs: Dict[str, Any]):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.search_kwargs = search_kwargs
        self._text_store = None
        self._image_store = None
        print(f"🚀 SmartRetriever initialized - Index: {index_name}")
    
    async def _ensure_stores_initialized(self):
        """Initialize stores asynchronously if not already done."""
        if self._text_store is None:
            self._text_store = await _get_or_create_pinecone_vs_async(
                self.index_name, self.embedding_model, namespace=""
            )
        
        if self._image_store is None:
            self._image_store = await _get_or_create_pinecone_vs_async(
                self.index_name, self.embedding_model, namespace="images"
            )
    
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
        is_image_focused = any(keyword in query_lower for keyword in image_keywords)
        print(f"🧠 Query analysis: '{query}' -> Image-focused: {is_image_focused}")
        return is_image_focused
    
    async def _async_search_text(self, query: str, k: int) -> List[Document]:
        """Asynchronous text search with asyncio.to_thread"""
        if k <= 0:
            return []
        
        await self._ensure_stores_initialized()
        
        def _sync_text_search():
            # Filter valid parameters for similarity_search
            valid_search_kwargs = {
                key: val for key, val in self.search_kwargs.items() 
                if key in ["k", "filter", "namespace", "include_metadata", "include_values"]
            }
            valid_search_kwargs["k"] = k
            
            text_retriever = self._text_store.as_retriever(search_kwargs=valid_search_kwargs)
            return text_retriever.invoke(query)
        
        docs = await asyncio.to_thread(_sync_text_search)
        
        # Add metadata
        for doc in docs:
            doc.metadata["source_type"] = "text"
            doc.metadata["namespace"] = ""
        
        print(f"✅ Text found: {len(docs)} documents")
        return docs
    
    async def _async_search_images(self, query: str, k: int) -> List[Document]:
        """Asynchronous image search with asyncio.to_thread"""
        if k <= 0:
            return []
        
        await self._ensure_stores_initialized()
        
        def _sync_image_search():
            # Filter valid parameters for similarity_search
            valid_search_kwargs = {
                key: val for key, val in self.search_kwargs.items() 
                if key in ["k", "filter", "namespace", "include_metadata", "include_values"]
            }
            valid_search_kwargs["k"] = k
            
            image_retriever = self._image_store.as_retriever(search_kwargs=valid_search_kwargs)
            return image_retriever.invoke(query)
        
        docs = await asyncio.to_thread(_sync_image_search)
        
        # Add metadata
        for doc in docs:
            doc.metadata["source_type"] = "image"
            doc.metadata["namespace"] = "images"
        
        if docs:
            print(f"✅ Images found: {len(docs)}")
            for i, doc in enumerate(docs[:3], 1):
                caption = doc.metadata.get('caption', doc.page_content)[:80]
                print(f"   {i}. {caption}...")
        else:
            print(f"⚠️ No images found for: '{query}'")
        
        return docs
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async retrieval with proper async handling
        """
        print(f"🔍 SEARCH: '{query}'")
        
        is_image_focused = self._is_image_focused_query(query)
        
        # Adaptive search strategy
        if is_image_focused:
            text_k = max(1, self.search_kwargs.get("k", 4) // 3)
            image_k = self.search_kwargs.get("k", 6)
            search_strategy = "image_focused"
        else:
            text_k = self.search_kwargs.get("k", 8)
            image_k = max(1, self.search_kwargs.get("k", 3))
            search_strategy = "text_focused"
        
        print(f"📊 Strategy: {search_strategy} | Text k={text_k}, Images k={image_k}")
        
        try:
            # Execute searches sequentially to avoid too many threads
            text_docs = await self._async_search_text(query, text_k)
            image_docs = await self._async_search_images(query, image_k)
            
            # Handle errors
            if isinstance(text_docs, Exception):
                print(f"❌ Text search failed: {text_docs}")
                text_docs = []
            
            if isinstance(image_docs, Exception):
                print(f"❌ Image search failed: {image_docs}")
                image_docs = []
            
            all_docs = text_docs + image_docs
            
            print(f"✅ RESULT: {len(text_docs)} text + {len(image_docs)} images = {len(all_docs)} total")
            return all_docs
        
        except Exception as e:
            print(f"💥 Error in retrieval: {e}")
            return []
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Sync method for compatibility"""
        return asyncio.run(self.aget_relevant_documents(query))
    
    async def ainvoke(self, query: str) -> List[Document]:
        """Async invoke for compatibility"""
        return await self.aget_relevant_documents(query)


## MAIN RETRIEVER FOR SEARCH
@asynccontextmanager
async def make_retriever(
    config: RunnableConfig,
) -> AsyncGenerator[SmartRetriever, None]:
    """
    Create SmartRetriever for SEARCH - FIXED VERSION
    """
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    index_name = get_pinecone_index_name()
    
    smart_retriever = SmartRetriever(
        embedding_model=embedding_model,
        index_name=index_name,
        search_kwargs=configuration.search_kwargs
    )
    
    yield smart_retriever