from contextlib import contextmanager
from typing import Generator, List, Dict, Any, Optional
import logging
from langchain_core.runnables import RunnableConfig
from shared.configuration import BaseConfiguration
from shared.retrieval import make_text_encoder


@contextmanager
def make_image_retriever(
    config: RunnableConfig,
) -> Generator[Any, None, None]:
    """Create an image retriever for Pinecone with image-specific index."""
    from langchain_pinecone import PineconeVectorStore
    import os
    
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    
    # Utiliser un index séparé pour les images ou un namespace
    image_index_name = os.environ.get("PINECONE_IMAGE_INDEX_NAME", os.environ["PINECONE_INDEX_NAME"])
    
    vstore = PineconeVectorStore.from_existing_index(
        image_index_name, 
        embedding=embedding_model,
        namespace="images"  # Utiliser un namespace pour séparer les images
    )
    yield vstore.as_retriever(search_kwargs={"k": 10})  # Récupérer plus d'images


async def retrieve_relevant_images(query: str, config: RunnableConfig, k: int = 5) -> List[Dict[str, Any]]:
    """
    Récupère les images pertinentes depuis Pinecone basées sur la requête
    """
    try:
        with make_image_retriever(config) as image_retriever:
            # Rechercher les images basées sur la requête
            image_docs = await image_retriever.ainvoke(query)
            
            images = []
            for doc in image_docs:
                metadata = doc.metadata
                
                # Vérifier que c'est bien un document d'image
                if metadata.get("type") == "image":
                    images.append({
                        "caption": metadata.get("caption", ""),
                        "image_url": metadata.get("image_url", ""),
                        "image_path": metadata.get("image_path", ""),
                        "source_url": metadata.get("source_url", ""),
                        "last_indexed_at": metadata.get("last_indexed_at", ""),
                        "relevance_score": getattr(doc, 'score', None),
                        "text_content": doc.page_content
                    })
            
            logging.info(f"✅ Retrieved {len(images)} relevant images for query: {query}")
            return images[:k]  # Limiter le nombre d'images retournées
            
    except Exception as e:
        logging.error(f"❌ Failed to retrieve images: {e}")
        return []


def is_image_related_query(question: str) -> bool:
    """Détermine si la question est liée aux images"""
    image_keywords = [
        'image', 'images', 'photo', 'photos', 'picture', 'pictures',
        'diagram', 'diagrams', 'figure', 'figures', 'chart', 'charts',
        'graph', 'graphs', 'screenshot', 'screenshots', 'visual', 'visuals',
        'illustration', 'illustrations', 'schéma', 'schémas', 'graphique',
        'graphiques', 'capture', 'captures', 'visualisation', 'visualisations',
        'montrer', 'afficher', 'voir', 'regarder', 'show', 'display', 'view'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in image_keywords)


def should_include_images(query: str, image_count: int) -> bool:
    """
    Détermine si on doit inclure des images dans la réponse
    basé sur la requête et le nombre d'images disponibles
    """
    if image_count == 0:
        return False
    
    # Toujours inclure si la requête mentionne explicitement les images
    if is_image_related_query(query):
        return True
    
    # Inclure si on a peu d'images très pertinentes
    if image_count <= 3:
        return True
    
    return False


def format_images_for_response(images: List[Dict[str, Any]], max_images: int = 5) -> str:
    """
    Formate les images pour inclusion dans la réponse
    """
    if not images:
        return "Aucune image pertinente trouvée."
    
    # Limiter le nombre d'images affichées
    images_to_show = images[:max_images]
    
    formatted_images = []
    for i, img in enumerate(images_to_show, 1):
        caption = img.get('caption', 'Sans description')
        image_url = img.get('image_url', '')
        source_url = img.get('source_url', '')
        
        image_info = f"""
**Image {i}**: {caption}
- 🔗 URL de l'image: {image_url}
- 📄 Source: {source_url}
"""
        formatted_images.append(image_info.strip())
    
    result = "\n\n".join(formatted_images)
    
    if len(images) > max_images:
        result += f"\n\n*... et {len(images) - max_images} autres images disponibles*"
    
    return result