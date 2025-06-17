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


## ✅  Wrapper async pour les opérations Pinecone bloquantes
async def _get_or_create_pinecone_vs_async(index_name: str, embedding_model: Embeddings, namespace: str = "") -> "PineconeVectorStore":
    """Create or connect to Pinecone vectorstore with namespace support - VERSION ASYNC."""
    
    def _sync_pinecone_operations():
        """Opérations Pinecone synchrones à exécuter dans un thread séparé."""
        pinecone_client = Pinecone(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ.get("PINECONE_ENVIRONMENT", "")
        )

        # Vérifier si l'index existe
        existing_indexes = pinecone_client.list_indexes().names()
        
        if index_name not in existing_indexes:
            print(f"🔄 Creating Pinecone index: {index_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Pinecone index created: {index_name}")
        else:
            print(f"✅ Pinecone index exists: {index_name}")
        
        return True
    
    # ✅ Exécuter les opérations Pinecone dans un thread séparé
    await asyncio.to_thread(_sync_pinecone_operations)
    
    # Créer le vectorstore (cette partie n'est pas bloquante)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model,
        namespace=namespace
    )
    
    return vectorstore


## ✅ INDEXERS CORRIGES POUR L'INDEXATION
@asynccontextmanager
async def make_text_indexer(
    configuration: BaseConfiguration, 
    embedding_model: Embeddings
) -> AsyncGenerator[PineconeVectorStore, None]:
    """
    ✅ CORRECTION: Text indexer pour indexation dans namespace par défaut - VERSION ASYNC
    Utilisé par index_graph pour indexer le contenu textuel
    """
    vectorstore = await _get_or_create_pinecone_vs_async(
        os.environ["PINECONE_INDEX_NAME"], 
        embedding_model,
        namespace=""  # Namespace par défaut pour le texte
    )
    print(f"📝 Text indexer créé - Index: {os.environ['PINECONE_INDEX_NAME']}, Namespace: '' (default)")
    yield vectorstore


@asynccontextmanager
async def make_image_indexer(
    config: RunnableConfig,
) -> AsyncGenerator[PineconeVectorStore, None]:
    """
    ✅ CORRECTION: Image indexer pour indexation dans namespace 'images' - VERSION ASYNC
    Utilisé par image_indexer.py - COMPATIBLE avec votre code existant
    """
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    vectorstore = await _get_or_create_pinecone_vs_async(
        os.environ["PINECONE_INDEX_NAME"],  # Même index que le texte !
        embedding_model,
        namespace="images"  # Namespace séparé pour les images
    )
    print(f"🖼️ Image indexer créé - Index: {os.environ['PINECONE_INDEX_NAME']}, Namespace: 'images'")
    yield vectorstore


# ✅ SMART RETRIEVER POUR LA RECHERCHE (pas pour l'indexation)
class SmartRetriever:
    """
    SmartRetriever pour la recherche (pas l'indexation)
    Utilise les deux namespaces pour récupérer texte + images
    """
    
    def __init__(self, embedding_model: Embeddings, index_name: str, search_kwargs: Dict[str, Any]):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.search_kwargs = search_kwargs
        self._text_store = None
        self._image_store = None
        print(f"🚀 SmartRetriever initialisé - Index: {index_name}")
    
    async def _ensure_stores_initialized(self):
        """Initialise les stores de manière async si pas déjà fait."""
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
        print(f"🧠 Analyse query: '{query}' -> Image-focused: {is_image_focused}")
        return is_image_focused
    
    async def _async_search_text(self, query: str, k: int) -> List[Document]:
        """Asynchronous text search"""
        if k <= 0:
            return []
        
        await self._ensure_stores_initialized()
        
        # Exécuter la recherche dans un thread séparé
        def _sync_text_search():
            text_retriever = self._text_store.as_retriever(
                search_kwargs={"k": k, **{key: val for key, val in self.search_kwargs.items() if key != "k"}}
            )
            return text_retriever.invoke(query)
        
        docs = await asyncio.to_thread(_sync_text_search)
        
        # Add metadata
        for doc in docs:
            doc.metadata["source_type"] = "text"
            doc.metadata["namespace"] = ""
        
        print(f"✅ Texte trouvé: {len(docs)} documents")
        return docs
    
    async def _async_search_images(self, query: str, k: int) -> List[Document]:
        """Asynchronous image search"""
        if k <= 0:
            return []
        
        await self._ensure_stores_initialized()
        
        # Exécuter la recherche dans un thread séparé
        def _sync_image_search():
            image_retriever = self._image_store.as_retriever(
                search_kwargs={"k": k, **{key: val for key, val in self.search_kwargs.items() if key != "k"}}
            )
            return image_retriever.invoke(query)
        
        docs = await asyncio.to_thread(_sync_image_search)
        
        # Add metadata
        for doc in docs:
            doc.metadata["source_type"] = "image"
            doc.metadata["namespace"] = "images"
        
        if docs:
            print(f"✅ Images trouvées: {len(docs)}")
            for i, doc in enumerate(docs[:3], 1):
                caption = doc.metadata.get('caption', doc.page_content)[:80]
                print(f"   {i}. {caption}...")
        else:
            print(f"⚠️ Aucune image trouvée pour: '{query}'")
        
        return docs
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async retrieval with proper async handling
        """
        print(f"🔍 RECHERCHE: '{query}'")
        
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
        
        print(f"📊 Stratégie: {search_strategy} | Texte k={text_k}, Images k={image_k}")
        
        try:
            # Exécuter les recherches en parallèle de manière async
            text_docs, image_docs = await asyncio.gather(
                self._async_search_text(query, text_k),
                self._async_search_images(query, image_k),
                return_exceptions=True
            )
            
            # Gérer les exceptions
            if isinstance(text_docs, Exception):
                print(f"❌ Text search failed: {text_docs}")
                text_docs = []
            
            if isinstance(image_docs, Exception):
                print(f"❌ Image search failed: {image_docs}")
                image_docs = []
            
            all_docs = text_docs + image_docs
            
            print(f"✅ RÉSULTAT: {len(text_docs)} texte + {len(image_docs)} images = {len(all_docs)} total")
            return all_docs
        
        except Exception as e:
            print(f"💥 Erreur dans retrieval: {e}")
            return []
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Sync method for compatibility"""
        return asyncio.run(self.aget_relevant_documents(query))
    
    async def ainvoke(self, query: str) -> List[Document]:
        """Async invoke for compatibility"""
        return await self.aget_relevant_documents(query)


## ✅ RETRIEVER PRINCIPAL POUR LA RECHERCHE
@asynccontextmanager
async def make_retriever(
    config: RunnableConfig,
) -> AsyncGenerator[SmartRetriever, None]:
    """
    ✅ CORRECTION: Créer le SmartRetriever pour la RECHERCHE - VERSION ASYNC
    (Pas pour l'indexation - utilisez make_text_indexer et make_image_indexer pour ça)
    """
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    
    smart_retriever = SmartRetriever(
        embedding_model=embedding_model,
        index_name=os.environ["PINECONE_INDEX_NAME"],
        search_kwargs=configuration.search_kwargs
    )
    
    yield smart_retriever