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
    """
    try:
        async with make_image_indexer(config) as vectorstore:

            soup = BeautifulSoup(html_content, "html.parser")
            figures = soup.find_all("figure")

            if not figures:
                logging.info(f"🔍 No <figure> tags found in {url}")
                return

            output_dir = Path("saved_images")
            output_dir.mkdir(exist_ok=True)

            now_str = datetime.utcnow().isoformat()
            to_index = []

            for i, fig in enumerate(figures):
                img_tag = fig.find("img")
                caption_tag = fig.find("figcaption")

                if not img_tag or not img_tag.get("src"):
                    continue

                image_url = urljoin(url, img_tag["src"])
                caption = caption_tag.get_text(separator=" ", strip=True) if caption_tag else ""

                if not caption:
                    logging.info(f"⚠️ No caption for image: {image_url} — skipped.")
                    continue

                # Nom local d’image
                url_hash = hashlib.md5(url.encode()).hexdigest()
                image_name = f"{url_hash}_{basename(urlparse(image_url).path)}"
                local_path = Path("saved_images") / image_name

                try:
                    image_data = requests.get(image_url).content
                    local_path.write_bytes(image_data)
                    logging.info(f"✅ Downloaded image {image_url} to {local_path}")
                except Exception as e:
                    logging.warning(f"❌ Failed to download image {image_url}: {e}")
                    continue

                vector_id = f"{url}--img--{image_name}"

                # Stocker dans une liste
                to_index.append({
                    "text": caption,
                    "metadata": {
                        "caption": caption,
                        "image_url": image_url,
                        "image_path": str(local_path),
                        "source_url": url,
                        "last_indexed_at": now_str,
                        "type": "image"
                    },
                    "id": vector_id
                })

            # ⏳ Insertion en lot, avec await si possible
            if to_index:
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

                logging.info(f"✅ Indexed {len(to_index)} image captions for {url}")

    except Exception as e:
        logging.error(f"🔥 Failed to process figures for {url}: {e}")
