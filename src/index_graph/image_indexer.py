from os.path import basename
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
from index_graph.configuration import IndexConfiguration
from shared.retrieval import make_image_indexer
from datetime import datetime


async def index_ocr_from_images(
    url: str, html_content: str, config: IndexConfiguration
) -> None:
    """
    Pour une page HTML donnée :
    - extrait les balises <figure> contenant des <img> et <figcaption>
    - télécharge les images localement
    - indexe les captions (si elles existent) dans Pinecone
    VERSION AVEC DEBUG ÉTENDU
    """
    print(f"🖼️ === DÉBUT INDEXATION IMAGES POUR {url} ===")
    
    try:
        async with make_image_indexer(config) as vectorstore:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # ✅ DEBUG: Chercher tous les types d'images
            all_images = soup.find_all("img")
            figures = soup.find_all("figure")
            
            print(f"🔍 Images totales trouvées: {len(all_images)}")
            print(f"🔍 Figures trouvées: {len(figures)}")
            
            if not all_images:
                print(f"⚠️ Aucune balise <img> trouvée dans {url}")
                return
            
            if not all_images:
                print(f"⚠️ Aucune balise <img> trouvée dans {url}")
                return

            # ✅ NOUVELLE APPROCHE: Indexer TOUTES les images, pas seulement les figures
            if figures:
                print(f"📋 Traitement de {len(figures)} figures...")
                await process_figures(figures, url, to_index, now_str)
            
            # ✅ FALLBACK AUTOMATIQUE: Indexer les images simples
            print(f"📋 Traitement de {len(all_images)} images simples...")
            await process_simple_images(all_images, url, to_index, now_str)

            # ⏳ Insertion en lot, avec await si possible
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
                        vectorstore.add_texts(
                            texts=[e["text"] for e in to_index],
                            metadatas=[e["metadata"] for e in to_index],
                            ids=[e["id"] for e in to_index],
                        )

                    print(f"✅ Indexed {len(to_index)} image captions for {url}")
                    
                    # ✅ DEBUG: Vérification immédiate
                    print(f"🧪 Test de vérification immédiate...")
                    test_results = vectorstore.similarity_search("architecture", k=1)
                    print(f"   Test résultats: {len(test_results)} trouvés")
                    
                except Exception as e:
                    print(f"❌ Erreur lors de l'indexation: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ Aucune image à indexer pour {url}")

    except Exception as e:
        print(f"🔥 Failed to process figures for {url}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"🖼️ === FIN INDEXATION IMAGES POUR {url} ===\n")


async def process_figures(figures: list, url: str, to_index: list, now_str: str) -> None:
    """Traiter les balises figure avec images et captions"""
    for i, fig in enumerate(figures):
        print(f"   🔍 Figure {i+1}/{len(figures)}")
        
        img_tag = fig.find("img")
        caption_tag = fig.find("figcaption")

        if not img_tag or not img_tag.get("src"):
            print(f"      ❌ Pas d'image valide")
            continue

        image_url = urljoin(url, img_tag["src"])
        caption = caption_tag.get_text(separator=" ", strip=True) if caption_tag else ""
        
        # Fallback: Utiliser alt text si pas de caption
        if not caption:
            caption = img_tag.get("alt", "")
        
        if not caption or len(caption.strip()) < 5:
            print(f"      ⚠️ No useful caption/alt for image: {image_url} — skipped.")
            continue

        print(f"      ✅ Caption: {caption[:100]}...")
        print(f"      🔗 URL: {image_url}")

        add_image_to_index(url, image_url, caption, to_index, now_str, "figure")


async def process_simple_images(all_images: list, url: str, to_index: list, now_str: str) -> None:
    """Traiter les images simples avec leur alt text"""
    useful_images = []
    
    for img in all_images:
        if not img.get("src"):
            continue
            
        image_url = urljoin(url, img["src"])
        alt_text = img.get("alt", "").strip()
        
        # Filtrer les images utiles (éviter logos, badges, etc.)
        if should_index_image(image_url, alt_text):
            useful_images.append((img, image_url, alt_text))
    
    print(f"   📊 Images utiles trouvées: {len(useful_images)}")
    
    for i, (img, image_url, alt_text) in enumerate(useful_images):
        print(f"   🔍 Image {i+1}/{len(useful_images)}")
        print(f"      ✅ Alt: {alt_text[:100]}...")
        print(f"      🔗 URL: {image_url}")
        
        add_image_to_index(url, image_url, alt_text, to_index, now_str, "simple")


def should_index_image(image_url: str, alt_text: str) -> bool:
    """Détermine si une image doit être indexée"""
    # Ignorer les logos et badges
    skip_patterns = [
        'wordmark', 'logo', 'brand', 'badge', 'shield',
        'colab-badge', 'github', 'pypi', 'downloads'
    ]
    
    # Vérifier l'URL
    for pattern in skip_patterns:
        if pattern.lower() in image_url.lower():
            return False
    
    # Vérifier l'alt text
    for pattern in skip_patterns:
        if pattern.lower() in alt_text.lower():
            return False
    
    # L'alt text doit être descriptif
    if len(alt_text.strip()) < 10:
        return False
    
    # Mots-clés positifs pour les diagrammes/schémas
    useful_keywords = [
        'diagram', 'architecture', 'flow', 'chart', 'graph', 
        'workflow', 'process', 'structure', 'overview', 'concept',
        'schéma', 'diagramme', 'architecture', 'flux'
    ]
    
    for keyword in useful_keywords:
        if keyword.lower() in alt_text.lower():
            return True
    
    # Si l'alt text est long et descriptif, probablement utile
    return len(alt_text.strip()) > 20


def add_image_to_index(url: str, image_url: str, caption: str, to_index: list, now_str: str, source_type: str) -> None:
    """Ajouter une image à la liste d'indexation"""
    # Nom local d'image
    url_hash = hashlib.md5(url.encode()).hexdigest()
    image_name = f"{url_hash}_{basename(urlparse(image_url).path)}"
    
    # Télécharger l'image (optionnel)
    local_path = None
    try:
        image_data = requests.get(image_url, timeout=10, verify=False).content
        output_dir = Path("saved_images")
        output_dir.mkdir(exist_ok=True)
        local_path = output_dir / image_name
        local_path.write_bytes(image_data)
        print(f"      💾 Downloaded to {local_path}")
    except Exception as e:
        print(f"      ⚠️ Could not download {image_url}: {e}")
    
    vector_id = f"{url}--{source_type}--{image_name}"

    # Stocker dans la liste
    to_index.append({
        "text": caption,
        "metadata": {
            "caption": caption,
            "image_url": image_url,
            "image_path": str(local_path) if local_path else "",
            "source_url": url,
            "last_indexed_at": now_str,
            "type": "image",
            "source_type": "image",
            "namespace": "images",
            "image_source": source_type  # figure ou simple
        },
        "id": vector_id
    })