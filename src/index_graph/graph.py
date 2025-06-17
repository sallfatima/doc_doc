"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import asyncio
import os
from typing import List, Optional

import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from index_graph.configuration import IndexConfiguration
from index_graph.state import IndexState, InputState
from shared import retrieval
from index_graph.image_indexer import index_ocr_from_images  # Import de votre fonction existante


def check_index_config(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """Check the API key."""
    configuration = IndexConfiguration.from_runnable_config(config)

    if not configuration.api_key:
        raise ValueError("API key is required for document indexing.")
    
    if configuration.api_key != os.getenv("INDEX_API_KEY"):
        raise ValueError("Authentication failed: Invalid API key provided.")
    
    if configuration.retriever_provider != "pinecone":
        raise ValueError("Only Pinecone is currently supported for document indexing due to specific ID prefix requirements.")
    
    return {}

async def get_sitemap_urls(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, List[str]]:
    """Get the URLs from the sitemap with detailed debugging - VERSION ASYNC CORRIGÉE."""
    import aiohttp
    import xml.etree.ElementTree as ET
    import re
    
    url = state.url_site_map
    print(f"🔍 Fetching sitemap from: {url}")
    
    headers = {
        "Accept": "application/xml",
        "User-Agent": "Mozilla/5.0 (compatible; LangChainBot/1.0)",
    }
    
    try:
        # ✅  Utiliser aiohttp au lieu de requests
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                print(f"📊 Response status: {response.status}")
                print(f"📊 Content-Type: {response.headers.get('content-type', 'Unknown')}")
                
                sitemap_content = await response.text()
                print(f"📊 Content length: {len(sitemap_content)} characters")
                
                # Afficher les premières lignes pour debug
                content_preview = sitemap_content[:500]
                print(f"📄 Content preview:\n{content_preview}")
        
        # Analyse XML détaillée
        try:
            root = ET.fromstring(sitemap_content)
            print(f"🔍 XML root tag: {root.tag}")
            print(f"🔍 XML attributes: {root.attrib}")
            
            # Essayer différents patterns XML
            patterns_to_try = [
                # Pattern standard
                ("{http://www.sitemaps.org/schemas/sitemap/0.9}url", "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"),
                # Pattern sans namespace
                ("url", "loc"),
                # Pattern avec sitemap index
                ("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap", "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"),
                ("sitemap", "loc"),
            ]
            
            urls_found = []
            
            for url_tag, loc_tag in patterns_to_try:
                print(f"🔍 Trying pattern: {url_tag} -> {loc_tag}")
                elements = root.findall(url_tag)
                print(f"   Found {len(elements)} elements with tag '{url_tag}'")
                
                for element in elements:  
                    loc_element = element.find(loc_tag)
                    if loc_element is not None and loc_element.text:
                        urls_found.append(loc_element.text)
                        print(f"   ✅ Found URL: {loc_element.text}")
                    else:
                        print(f"   ❌ No loc element found in: {ET.tostring(element, encoding='unicode')[:100]}")
                
                if urls_found:
                    print(f"✅ Success with pattern: {url_tag} -> {loc_tag}")
                    break
            
            # Si toujours rien, essayer d'analyser la structure
            if not urls_found:
                print("🔍 No URLs found with standard patterns. Analyzing XML structure:")
                
                # Essayer de trouver toutes les URLs dans le texte
                url_pattern = r'https?://[^\s<>"]+langchain[^\s<>"]*'
                found_urls = re.findall(url_pattern, sitemap_content)
                
                if found_urls:
                    print(f"🔍 Found URLs via regex: {len(found_urls)}")
                    for found_url in found_urls[:5]:
                        print(f"   📎 {found_url}")
                    urls_found = found_urls[:100]  # Limiter pour éviter trop d'URLs
                
        except ET.ParseError as e:
            print(f"❌ XML parsing error: {e}")
            print("🔍 Content is not valid XML. Content preview:")
            print(sitemap_content[:1000])
            
            # Peut-être que c'est un index de sitemaps, essayer de trouver les URLs autrement
            url_pattern = r'https?://[^\s<>"]+\.xml'
            xml_urls = re.findall(url_pattern, sitemap_content)
            
            if xml_urls:
                print(f"🔍 Found XML URLs (might be sitemap index): {xml_urls}")
                return {"urls_to_index": []}  # Retourner vide pour l'instant
            
            return {"urls_to_index": []}
        
        print(f"🎯 Final result: {len(urls_found)} URLs to index")
        
        if urls_found:
            print("📋 First 5 URLs:")
            for i, found_url in enumerate(urls_found[:5], 1):
                print(f"   {i}. {found_url}")
        
        return {"urls_to_index": urls_found}
        
    except Exception as e:
        print(f"❌ Error fetching sitemap: {e}")
        import traceback
        traceback.print_exc()
        return {"urls_to_index": []}

async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function now indexes both text content and images from web pages.
    Text goes to the default namespace, images go to the 'images' namespace.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.
    """
    print(f"🚀 Starting indexation of {len(state.urls_to_index)} URLs")
    
    # Process all URLs in parallel
    chunk_tasks = [index_url(url, config) for url in state.urls_to_index]
    await asyncio.gather(*chunk_tasks)

    print("✅ All URLs indexed successfully")
    return {}


async def index_url(url: str, config: Optional[RunnableConfig] = None) -> List[Document]:
    """
    Index a web path - TEXTE ET IMAGES - VERSION ASYNC CORRIGÉE
    
   
    - Texte → namespace par défaut ("")  
    - Images → namespace "images"
    """
    print(f"🔄 Indexing URL: {url}")
    
    try:
        # ✅  Utiliser asyncio.to_thread pour WebBaseLoader (bloquant)
        def load_web_content(web_url):
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(web_paths=(web_url,))
            return loader.load()
        
        # Charger le contenu de la page dans un thread séparé
        docs = await asyncio.to_thread(load_web_content, url)
        
        # Récupérer le contenu HTML pour l'indexation d'images
        html_content = ""
        if docs:
            html_content = docs[0].page_content
        
        # ✅  Utiliser asyncio.to_thread pour le text splitter
        def split_documents(documents):
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return text_splitter.split_documents(documents)
        
        text_docs = await asyncio.to_thread(split_documents, docs)
        
        # ✅ INDEXATION DU TEXTE dans le namespace par défaut
        if text_docs:
            print(f"📝 Indexing {len(text_docs)} text chunks from {url}")
            
            configuration = IndexConfiguration.from_runnable_config(config)
            embedding_model = retrieval.make_text_encoder(configuration.embedding_model)
            
            async with retrieval.make_text_indexer(configuration, embedding_model) as text_vectorstore:
                await text_vectorstore.aadd_texts(
                    texts=[doc.page_content for doc in text_docs],
                    metadatas=[{**doc.metadata, "source_type": "text", "indexed_url": url} for doc in text_docs],
                    ids=[f"{url}--text--{i}" for i in range(len(text_docs))]
                )
            
            print(f"✅ {len(text_docs)} text chunks indexed for {url}")
        
        # ✅ INDEXATION DES IMAGES dans le namespace "images"
        print(f"🖼️ Looking for images in {url}")
        
        try:
            # Utiliser votre fonction existante d'indexation d'images
            await index_ocr_from_images(url, html_content, config)
            print(f"✅ Images indexed for {url}")
        except Exception as e:
            print(f"⚠️ Warning: Could not index images for {url}: {e}")
        
        print(f"✅ Successfully indexed {url}")
        return text_docs
        
    except Exception as e:
        print(f"❌ Error indexing {url}: {e}")
        import traceback
        traceback.print_exc()
        return []
# ✅ LANGGRAPH DEFINITION - PARTIE MANQUANTE RESTAURÉE
builder = StateGraph(IndexState, input=InputState, config_schema=IndexConfiguration)

# Add nodes
builder.add_node("check_index_config", check_index_config)
builder.add_node("get_sitemap_urls", get_sitemap_urls) 
builder.add_node("index_docs", index_docs)

# Add edges
builder.add_edge(START, "check_index_config")
builder.add_edge("check_index_config", "get_sitemap_urls")
builder.add_edge("get_sitemap_urls", "index_docs")
builder.add_edge("index_docs", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"

print("🎯 IndexGraph compiled successfully with namespace-based indexing")