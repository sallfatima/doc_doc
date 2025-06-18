import asyncio
import os
import logging
from typing import List, Optional
from datetime import datetime, timedelta

import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pinecone import Index

from index_graph.configuration import IndexConfiguration
from index_graph.state import IndexState, InputState
from shared import retrieval
from index_graph.image_indexer import index_ocr_from_images


def check_index_config(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """Check the API key and configuration with enhanced debugging."""
    configuration = IndexConfiguration.from_runnable_config(config)

    print("🔧 Configuration Check:")
    print(f"   API Key present: {'Yes' if configuration.api_key else 'No'}")
    print(f"   Retriever provider: {configuration.retriever_provider}")
    
    # Debug environment variables
    print("🌍 Environment Variables:")
    pinecone_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_INDEX"]
    for var in pinecone_vars:
        value = os.environ.get(var)
        if value:
            display_value = f"{value[:8]}..." if "API_KEY" in var else value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ❌ {var}: Not set")
    
    # Determine the index name that will be used
    index_name = get_pinecone_index_name()
    print(f"🎯 Will use index: {index_name}")

    if not configuration.api_key:
        raise ValueError("API key is required for document indexing.")
    
    expected_api_key = os.getenv("INDEX_API_KEY")
    if not expected_api_key:
        print("⚠️ Warning: INDEX_API_KEY not set in environment")
    elif configuration.api_key != expected_api_key:
        raise ValueError("Authentication failed: Invalid API key provided.")
    
    if configuration.retriever_provider != "pinecone":
        raise ValueError("Only Pinecone is currently supported for document indexing due to specific ID prefix requirements.")
    
    return {}


def get_pinecone_index_name() -> str:
    """Get the correct Pinecone index name from environment variables."""
    # Try different environment variable names
    index_name = (
        os.environ.get("PINECONE_INDEX_NAME") or 
        os.environ.get("PINECONE_INDEX") or 
        "index-text"  # Default to index-text if not specified
    )
    print(f"🔧 Resolved index name: {index_name}")
    print(f"🔍 PINECONE_INDEX_NAME env: {os.environ.get('PINECONE_INDEX_NAME')}")
    print(f"🔍 PINECONE_INDEX env: {os.environ.get('PINECONE_INDEX')}")
    return index_name


async def get_sitemap_urls(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, List[str]]:
    """Get the URLs from the sitemap with detailed debugging."""
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
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as response:
                print(f"📊 Response status: {response.status}")
                sitemap_content = await response.text()
                print(f"📊 Content length: {len(sitemap_content)} characters")
        
        # Parse XML
        try:
            root = ET.fromstring(sitemap_content)
            print(f"🔍 XML root tag: {root.tag}")
            
            # Try different XML patterns
            patterns_to_try = [
                ("{http://www.sitemaps.org/schemas/sitemap/0.9}url", "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"),
                ("url", "loc"),
                ("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap", "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"),
                ("sitemap", "loc"),
            ]
            
            urls_found = []
            
            for url_tag, loc_tag in patterns_to_try:
                elements = root.findall(url_tag)
                for element in elements:  
                    loc_element = element.find(loc_tag)
                    if loc_element is not None and loc_element.text:
                        urls_found.append(loc_element.text)
                
                if urls_found:
                    break
            
            # Fallback to regex if no URLs found
            if not urls_found:
                url_pattern = r'https?://[^\s<>"]+langchain[^\s<>"]*'
                found_urls = re.findall(url_pattern, sitemap_content)
                if found_urls:
                    urls_found = found_urls[:100]  # Limit URLs
                
        except ET.ParseError as e:
            print(f"❌ XML parsing error: {e}")
            return {"urls_to_index": []}
        
        print(f"🎯 Final result: {len(urls_found)} URLs to index")
        return {"urls_to_index": urls_found}
        
    except Exception as e:
        print(f"❌ Error fetching sitemap: {e}")
        return {"urls_to_index": []}


async def check_url_freshness(url: str, index: Index, max_age_days: int = 7) -> tuple[bool, Optional[str]]:
    """
    Check if URL content needs re-indexing based on last indexed timestamp.
    
    Returns:
        (needs_reindex, last_indexed_date)
    """
    try:
        # Query for existing chunks with this URL prefix - using asyncio.to_thread
        query_response = await asyncio.to_thread(
            index.query,
            vector=[0.1] * 1536,  # Dummy vector for metadata query
            top_k=1,
            filter={"source_url": url},
            namespace="",  # Check in default namespace
            include_metadata=True
        )
        
        if not query_response.matches:
            print(f"🆕 New URL: {url}")
            return True, None
        
        # Check the timestamp of the most recent chunk
        last_indexed_str = query_response.matches[0].metadata.get("last_indexed_at")
        if not last_indexed_str:
            print(f"🔄 Missing timestamp for {url}, re-indexing")
            return True, None
        
        try:
            last_indexed = datetime.fromisoformat(last_indexed_str.replace('Z', '+00:00'))
            age_days = (datetime.now(last_indexed.tzinfo) - last_indexed).days
            
            if age_days > max_age_days:
                print(f"🕒 Stale content ({age_days} days old): {url}")
                return True, last_indexed_str
            else:
                print(f"✅ Fresh content ({age_days} days old): {url}")
                return False, last_indexed_str
                
        except ValueError as e:
            print(f"🔄 Invalid timestamp format for {url}: {e}")
            return True, last_indexed_str
            
    except Exception as e:
        print(f"⚠️ Error checking freshness for {url}: {e}")
        return True, None  # Re-index on error to be safe


async def delete_old_chunks(url: str, index: Index) -> int:
    """
    Delete all existing chunks for a given URL from both namespaces.
    
    Returns:
        Number of chunks deleted
    """
    deleted_count = 0
    
    # Delete from default namespace (text)
    try:
        existing_ids = await asyncio.to_thread(list, index.list(prefix=f"{url}--text"))
        if existing_ids:
            await asyncio.to_thread(index.delete, ids=existing_ids, namespace="")
            deleted_count += len(existing_ids)
            print(f"🗑️ Deleted {len(existing_ids)} text chunks for {url}")
    except Exception as e:
        print(f"⚠️ Error deleting text chunks for {url}: {e}")
    
    # Delete from images namespace
    try:
        existing_image_ids = await asyncio.to_thread(list, index.list(prefix=f"{url}--"))
        if existing_image_ids:
            # Filter for image IDs (they contain --figure-- or --simple--)
            image_ids = [id for id in existing_image_ids if ("--figure--" in id or "--simple--" in id)]
            if image_ids:
                await asyncio.to_thread(index.delete, ids=image_ids, namespace="images")
                deleted_count += len(image_ids)
                print(f"🗑️ Deleted {len(image_ids)} image chunks for {url}")
    except Exception as e:
        print(f"⚠️ Error deleting image chunks for {url}: {e}")
    
    return deleted_count


async def index_url(url: str, config: IndexConfiguration, index: Index, retry: int = 1) -> List[Document]:
    """
    Delete old chunks and re-index content from a given URL with deduplication.
    
    Args:
        url: URL to index
        config: Configuration object
        index: Pinecone index instance
        retry: Number of retries left
    
    Returns:
        List of indexed documents
    """
    try:
        print(f"🔍 Processing: {url}")
        
        # Step 1: Check if URL needs re-indexing
        needs_reindex, last_indexed = await check_url_freshness(url, index, max_age_days=7)
        
        if not needs_reindex:
            print(f"⏭️ Skipping {url} (fresh content)")
            return []
        
        # Step 2: Load and process content
        print(f"📥 Loading content from {url}")
        
        def load_web_content(web_url):
            loader = WebBaseLoader(web_paths=(web_url,))
            return loader.load()
        
        docs = await asyncio.to_thread(load_web_content, url)
        
        if not docs:
            print(f"⚠️ No content loaded for {url}")
            return []
        
        # Get HTML content for image processing
        html_content = docs[0].page_content if docs else ""
        
        # Step 3: Split text documents
        def split_documents(documents):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
        
        text_docs = await asyncio.to_thread(split_documents, docs)
        
        # Step 4: Delete old chunks before adding new ones
        deleted_count = await delete_old_chunks(url, index)
        if deleted_count > 0:
            print(f"🧹 Cleaned up {deleted_count} old chunks for {url}")
        
        # Step 5: Index text content
        if text_docs:
            print(f"📝 Indexing {len(text_docs)} text chunks from {url}")
            
            now_str = datetime.utcnow().isoformat()
            
            # Prepare metadata with proper tracking
            for i, doc in enumerate(text_docs):
                doc.metadata.update({
                    "source_url": url,
                    "last_indexed_at": now_str,
                    "source_type": "text",
                    "namespace": "",
                    "chunk_index": i
                })
            
            # Prepare data for indexing
            texts = [doc.page_content for doc in text_docs]
            metadatas = [doc.metadata for doc in text_docs]
            chunk_ids = [f"{url}--text--{i}" for i in range(len(texts))]
            
            # Index text content in default namespace
            embedding_model = retrieval.make_text_encoder(config.embedding_model)
            
            async with retrieval.make_text_indexer(config, embedding_model) as text_vectorstore:
                if hasattr(text_vectorstore, "aadd_texts"):
                    await text_vectorstore.aadd_texts(
                        texts=texts,
                        metadatas=metadatas,
                        ids=chunk_ids
                    )
                else:
                    # Fallback for sync operations
                    await asyncio.to_thread(
                        text_vectorstore.add_texts,
                        texts=texts,
                        metadatas=metadatas,
                        ids=chunk_ids
                    )
            
            print(f"✅ {len(text_docs)} text chunks indexed for {url}")
        
        # Step 6: Index images
        print(f"🖼️ Processing images for {url}")
        
        try:
            # Use existing image indexer function
            await index_ocr_from_images(url, html_content, config)
            print(f"✅ Images processed for {url}")
        except Exception as e:
            print(f"⚠️ Warning: Could not index images for {url}: {e}")
        
        # Step 7: Verify indexing success
        await asyncio.sleep(1)  # Allow time for indexing to complete
        
        # Quick verification query - using asyncio.to_thread
        try:
            verification_query = await asyncio.to_thread(
                index.query,
                vector=[0.1] * 1536,
                top_k=1,
                filter={"source_url": url},
                namespace="",
                include_metadata=True
            )
            
            if verification_query.matches:
                print(f"✅ Verification successful for {url}")
            else:
                print(f"⚠️ Verification failed for {url}")
                
        except Exception as e:
            print(f"⚠️ Could not verify indexing for {url}: {e}")
        
        print(f"🎉 Successfully processed {url}")
        return text_docs
        
    except Exception as e:
        if retry > 0:
            print(f"⚠️ Retry {url} after error: {e}")
            await asyncio.sleep(2)  # Wait before retry
            return await index_url(url, config, index, retry=retry - 1)
        else:
            print(f"❌ Final failure for {url}: {e}")
            logging.error(f"Final failure for {url}: {e}")
            return []


async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """
    Asynchronously index documents with deduplication and progress tracking.
    """
    configuration = IndexConfiguration.from_runnable_config(config)
    
    print(f"🚀 Starting indexation of {len(state.urls_to_index)} URLs")
    
    # Connect to Pinecone index - using asyncio.to_thread for blocking operations
    def get_pinecone_index():
        from pinecone import Pinecone
        from pinecone import ServerlessSpec
        
        # Get the resolved index name
        index_name = get_pinecone_index_name()
        print(f"📌 Connecting to Pinecone index: {index_name}")
        
        try:
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            
            # Check if index exists
            existing_indexes = pc.list_indexes().names()
            print(f"📋 Available indexes: {existing_indexes}")
            
            if index_name not in existing_indexes:
                print(f"🔄 Index '{index_name}' doesn't exist, creating it...")
                
                # Create the index first
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws", 
                        region="us-east-1"
                    )
                )
                
                print(f"✅ Index '{index_name}' created successfully")
                
                # Wait for index to be ready
                import time
                print("⏳ Waiting for index to be ready...")
                time.sleep(10)  # Give it time to initialize
                
            else:
                print(f"✅ Index '{index_name}' already exists")
            
            # Now connect to the index (existing or newly created)
            return pc.Index(index_name)
            
        except Exception as e:
            print(f"❌ Error with Pinecone index operations: {e}")
            raise
    
    index = await asyncio.to_thread(get_pinecone_index)
    
    # Process URLs with controlled concurrency
    semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
    
    async def process_url_with_semaphore(url):
        async with semaphore:
            return await index_url(url, configuration, index)
    
    # Process all URLs
    tasks = [process_url_with_semaphore(url) for url in state.urls_to_index]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successful vs failed indexing
    successful = sum(1 for result in results if not isinstance(result, Exception) and result)
    failed = len(results) - successful
    
    print(f"📊 Indexing complete: {successful} successful, {failed} failed")
    
    return {
        "status": "completed",
        "indexed_count": successful,
        "failed_count": failed
    }


# LangGraph Definition
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

# Compile the graph
graph = builder.compile()
graph.name = "IndexGraph"

print("🎯 IndexGraph compiled successfully with deduplication and namespace management")