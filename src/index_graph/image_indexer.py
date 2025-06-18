import asyncio
import logging
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from os.path import basename
from datetime import datetime
import aiohttp
import aiofiles

from index_graph.configuration import IndexConfiguration
from shared.retrieval import make_image_indexer


async def index_ocr_from_images(
    url: str, html_content: str, config: IndexConfiguration
) -> None:
    """
    Async version - Extract images from HTML and index them in Pinecone.
    
    For a given HTML page:
    - Extracts <figure> tags containing <img> and <figcaption>
    - Downloads images locally using async HTTP
    - Indexes captions in Pinecone 'images' namespace
    """
    print(f"🖼️ === DÉBUT INDEXATION IMAGES POUR {url} ===")
    
    try:
        async with make_image_indexer(config) as vectorstore:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Find all image elements
            all_images = soup.find_all("img")
            figures = soup.find_all("figure")
            
            print(f"🔍 Images totales trouvées: {len(all_images)}")
            print(f"🔍 Figures trouvées: {len(figures)}")
            
            if not all_images:
                print(f"⚠️ Aucune balise <img> trouvée dans {url}")
                return
            
            to_index = []
            now_str = datetime.utcnow().isoformat()
            
            # Process figures first
            if figures:
                print(f"📋 Traitement de {len(figures)} figures...")
                await process_figures_async(figures, url, to_index, now_str)
            
            # Process simple images as fallback
            print(f"📋 Traitement de {len(all_images)} images simples...")
            await process_simple_images_async(all_images, url, to_index, now_str)

            # Batch indexing
            if to_index:
                print(f"📤 Indexation de {len(to_index)} images...")
                
                try:
                    if hasattr(vectorstore, "aadd_texts"):
                        await vectorstore.aadd_texts(
                            texts=[e["text"] for e in to_index],
                            metadatas=[e["metadata"] for e in to_index],
                            ids=[e["id"] for e in to_index],
                        )
                    else:
                        # Fallback to sync in thread
                        await asyncio.to_thread(
                            vectorstore.add_texts,
                            texts=[e["text"] for e in to_index],
                            metadatas=[e["metadata"] for e in to_index],
                            ids=[e["id"] for e in to_index],
                        )

                    print(f"✅ Indexed {len(to_index)} image captions for {url}")
                    
                    # Verification
                    print(f"🧪 Test de vérification immédiate...")
                    test_results = await asyncio.to_thread(
                        vectorstore.similarity_search, 
                        "architecture", 
                        k=1
                    )
                    print(f"   Test résultats: {len(test_results)} trouvés")
                    
                except Exception as e:
                    print(f"❌ Erreur lors de l'indexation: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ Aucune image à indexer pour {url}")

    except Exception as e:
        print(f"🔥 Failed to process images for {url}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"🖼️ === FIN INDEXATION IMAGES POUR {url} ===\n")


async def process_figures_async(figures: list, url: str, to_index: list, now_str: str) -> None:
    """Process figure tags with images and captions - async version."""
    for i, fig in enumerate(figures):
        print(f"   🔍 Figure {i+1}/{len(figures)}")
        
        img_tag = fig.find("img")
        caption_tag = fig.find("figcaption")

        if not img_tag or not img_tag.get("src"):
            print(f"      ❌ Pas d'image valide")
            continue

        image_url = urljoin(url, img_tag["src"])
        caption = caption_tag.get_text(separator=" ", strip=True) if caption_tag else ""
        
        # Fallback: Use alt text if no caption
        if not caption:
            caption = img_tag.get("alt", "")
        
        if not caption or len(caption.strip()) < 5:
            print(f"      ⚠️ No useful caption/alt for image: {image_url} — skipped.")
            continue

        print(f"      ✅ Caption: {caption[:100]}...")
        print(f"      🔗 URL: {image_url}")

        await add_image_to_index_async(url, image_url, caption, to_index, now_str, "figure")


async def process_simple_images_async(all_images: list, url: str, to_index: list, now_str: str) -> None:
    """Process simple images with their alt text - async version."""
    useful_images = []
    
    for img in all_images:
        if not img.get("src"):
            continue
            
        image_url = urljoin(url, img["src"])
        alt_text = img.get("alt", "").strip()
        
        # Filter useful images (avoid logos, badges, etc.)
        if should_index_image(image_url, alt_text):
            useful_images.append((img, image_url, alt_text))
    
    print(f"   📊 Images utiles trouvées: {len(useful_images)}")
    
    # Process images concurrently
    tasks = []
    for i, (img, image_url, alt_text) in enumerate(useful_images):
        print(f"   🔍 Image {i+1}/{len(useful_images)}")
        print(f"      ✅ Alt: {alt_text[:100]}...")
        print(f"      🔗 URL: {image_url}")
        
        task = add_image_to_index_async(url, image_url, alt_text, to_index, now_str, "simple")
        tasks.append(task)
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def should_index_image(image_url: str, alt_text: str) -> bool:
    """Determine if an image should be indexed."""
    # Skip logos and badges
    skip_patterns = [
        'wordmark', 'logo', 'brand', 'badge', 'shield',
        'colab-badge', 'github', 'pypi', 'downloads'
    ]
    
    # Check URL
    for pattern in skip_patterns:
        if pattern.lower() in image_url.lower():
            return False
    
    # Check alt text
    for pattern in skip_patterns:
        if pattern.lower() in alt_text.lower():
            return False
    
    # Alt text should be descriptive
    if len(alt_text.strip()) < 10:
        return False
    
    # Positive keywords for diagrams/schemas
    useful_keywords = [
        'diagram', 'architecture', 'flow', 'chart', 'graph', 
        'workflow', 'process', 'structure', 'overview', 'concept',
        'schéma', 'diagramme', 'architecture', 'flux'
    ]
    
    for keyword in useful_keywords:
        if keyword.lower() in alt_text.lower():
            return True
    
    # If alt text is long and descriptive, probably useful
    return len(alt_text.strip()) > 20


async def download_image_async(image_url: str, local_path: Path) -> bool:
    """
    Download an image asynchronously using aiohttp.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                image_url, 
                timeout=aiohttp.ClientTimeout(total=10),
                ssl=False  # Disable SSL verification for problematic sites
            ) as response:
                if response.status == 200:
                    content = await response.read()
                    async with aiofiles.open(local_path, 'wb') as f:
                        await f.write(content)
                    print(f"      💾 Downloaded to {local_path}")
                    return True
                else:
                    print(f"      ⚠️ HTTP {response.status} for {image_url}")
                    return False
                    
    except Exception as e:
        print(f"      ⚠️ Could not download {image_url}: {e}")
        return False


async def add_image_to_index_async(
    url: str, 
    image_url: str, 
    caption: str, 
    to_index: list, 
    now_str: str, 
    source_type: str
) -> None:
    """Add an image to the indexing list - async version with image download."""
    # Generate local image name
    url_hash = hashlib.md5(url.encode()).hexdigest()
    image_name = f"{url_hash}_{basename(urlparse(image_url).path)}"
    
    # Download image asynchronously
    output_dir = Path("saved_images")
    local_path = output_dir / image_name
    
    download_success = await download_image_async(image_url, local_path)
    
    vector_id = f"{url}--{source_type}--{image_name}"

    # Store in indexing list
    image_metadata = {
        "caption": caption,
        "image_url": image_url,
        "image_path": str(local_path) if download_success else "",
        "source_url": url,
        "last_indexed_at": now_str,
        "type": "image",
        "source_type": "image",
        "namespace": "images",
        "image_source": source_type  # figure or simple
    }
    
    to_index.append({
        "text": caption,
        "metadata": image_metadata,
        "id": vector_id
    })