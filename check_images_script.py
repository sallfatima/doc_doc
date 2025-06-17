#!/usr/bin/env python3
"""
Script pour vérifier les images indexées dans Pinecone
"""

import os
import asyncio
from pathlib import Path
import sys

# Ajouter le src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shared.retrieval import SmartRetriever, make_text_encoder
from shared.configuration import BaseConfiguration


async def check_indexed_images():
    """Vérifier les images indexées dans Pinecone"""
    print("🔍 VÉRIFICATION DES IMAGES INDEXÉES")
    print("=" * 50)
    
    try:
        config = BaseConfiguration()
        embedding_model = make_text_encoder("openai/text-embedding-3-small")
        
        retriever = SmartRetriever(
            embedding_model=embedding_model,
            index_name=os.environ["PINECONE_INDEX_NAME"],
            search_kwargs={"k": 10}
        )
        
        # Tests avec différentes requêtes
        test_queries = [
            "architecture",
            "diagram", 
            "langchain",
            "graph",
            "workflow",
            "agent",
            "image",
            "figure"
        ]
        
        total_images_found = 0
        
        for query in test_queries:
            print(f"\n🔍 Test query: '{query}'")
            
            # Rechercher seulement les images
            results = await retriever._async_search_images(query, 5)
            
            if results:
                print(f"✅ {len(results)} images trouvées:")
                for i, doc in enumerate(results, 1):
                    caption = doc.metadata.get('caption', doc.page_content)[:100]
                    image_url = doc.metadata.get('image_url', 'N/A')
                    print(f"   {i}. {caption}...")
                    print(f"      URL: {image_url}")
                total_images_found += len(results)
            else:
                print("❌ Aucune image trouvée")
        
        print(f"\n📊 RÉSUMÉ:")
        print(f"Total d'images trouvées: {total_images_found}")
        
        if total_images_found == 0:
            print("\n⚠️ PROBLÈME: Aucune image indexée détectée!")
            print("Solutions possibles:")
            print("1. Vérifier que l'indexation a bien fonctionné")
            print("2. Relancer l'indexation avec le sitemap")
            print("3. Vérifier les namespaces Pinecone")
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(check_indexed_images())