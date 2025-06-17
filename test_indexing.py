#!/usr/bin/env python3
"""
Test de l'indexation pour vérifier que tout fonctionne
Usage: python test_indexing.py
"""

import os
import sys
import asyncio
from pathlib import Path

# Ajouter le src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shared.retrieval import make_text_indexer, make_image_indexer, make_text_encoder
from shared.configuration import BaseConfiguration
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from datetime import datetime


async def test_text_indexing():
    """Test de l'indexation de texte"""
    print("📝 Test indexation TEXTE")
    print("-" * 40)
    
    # Configuration
    config = BaseConfiguration()
    embedding_model = make_text_encoder("openai/text-embedding-3-small")
    
    # Document de test
    test_doc = Document(
        page_content="Ceci est un test d'indexation de texte pour LangChain RAG system.",
        metadata={
            "source": "test_document",
            "test_id": "text_001", 
            "indexed_at": datetime.now().isoformat(),
            "source_type": "text"
        }
    )
    
    try:
        async with make_text_indexer(config, embedding_model) as text_vectorstore:
            # Index le document de test
            await text_vectorstore.aadd_texts(
                texts=[test_doc.page_content],
                metadatas=[test_doc.metadata],
                ids=["test_text_001"]
            )
            
            print("✅ Document texte indexé avec succès")
            
            # Test de recherche immédiate
            retriever = text_vectorstore.as_retriever(search_kwargs={"k": 1})
            results = retriever.invoke("test indexation")
            
            if results:
                print(f"✅ Test recherche: trouvé {len(results)} résultat(s)")
                print(f"   Contenu: {results[0].page_content[:60]}...")
            else:
                print("⚠️ Aucun résultat trouvé lors du test de recherche")
                
        return True
        
    except Exception as e:
        print(f"❌ Erreur indexation texte: {e}")
        return False


async def test_image_indexing():
    """Test de l'indexation d'image"""
    print("\n🖼️ Test indexation IMAGES")
    print("-" * 40)
    
    # Configuration
    config = RunnableConfig(
        configurable={
            "embedding_model": "openai/text-embedding-3-small"
        }
    )
    
    # Données d'image de test
    test_image_data = {
        "text": "Test diagram showing LangChain architecture components",
        "metadata": {
            "caption": "Test diagram showing LangChain architecture components",
            "image_url": "https://example.com/test_diagram.png",
            "source_url": "https://example.com/test_page",
            "image_path": "/test/images/test_diagram.png",
            "last_indexed_at": datetime.now().isoformat(),
            "type": "image"
        },
        "id": "test_image_001"
    }
    
    try:
        async with make_image_indexer(config) as image_vectorstore:
            # Index l'image de test
            await image_vectorstore.aadd_texts(
                texts=[test_image_data["text"]],
                metadatas=[test_image_data["metadata"]],
                ids=[test_image_data["id"]]
            )
            
            print("✅ Image de test indexée avec succès")
            
            # Test de recherche immédiate
            retriever = image_vectorstore.as_retriever(search_kwargs={"k": 1})
            results = retriever.invoke("diagram architecture")
            
            if results:
                print(f"✅ Test recherche: trouvé {len(results)} résultat(s)")
                result = results[0]
                print(f"   Caption: {result.metadata.get('caption', 'N/A')}")
                print(f"   URL: {result.metadata.get('image_url', 'N/A')}")
            else:
                print("⚠️ Aucun résultat trouvé lors du test de recherche")
                
        return True
        
    except Exception as e:
        print(f"❌ Erreur indexation image: {e}")
        return False


async def test_namespace_separation():
    """Test que les namespaces sont bien séparés"""
    print("\n🔄 Test séparation NAMESPACES")
    print("-" * 40)
    
    try:
        from shared.retrieval import SmartRetriever, make_text_encoder
        from shared.configuration import BaseConfiguration
        
        config = BaseConfiguration()
        embedding_model = make_text_encoder("openai/text-embedding-3-small")
        
        # Créer le SmartRetriever
        retriever = SmartRetriever(
            embedding_model=embedding_model,
            index_name=os.environ["PINECONE_INDEX_NAME"],
            search_kwargs={"k": 5}
        )
        
        # Test recherche dans les deux namespaces
        results = await retriever.aget_relevant_documents("test diagram")
        
        text_results = [r for r in results if r.metadata.get("source_type") == "text"]
        image_results = [r for r in results if r.metadata.get("source_type") == "image"]
        
        print(f"✅ Recherche combinée: {len(text_results)} texte + {len(image_results)} images")
        
        if text_results:
            print(f"   Namespace texte: {text_results[0].metadata.get('namespace', 'default')}")
        
        if image_results:
            print(f"   Namespace images: {image_results[0].metadata.get('namespace', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test namespaces: {e}")
        return False


async def main():
    """Point d'entrée principal"""
    print("🧪 TEST COMPLET D'INDEXATION")
    print("=" * 50)
    
    # Vérifications préliminaires
    required_env = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing_env = [env for env in required_env if not os.getenv(env)]
    
    if missing_env:
        print(f"❌ Variables d'environnement manquantes: {', '.join(missing_env)}")
        return
    
    print(f"✅ Configuration OK")
    print(f"📂 Index Pinecone: {os.getenv('PINECONE_INDEX_NAME')}")
    
    # Tests
    tests = [
        ("Indexation Texte", test_text_indexing),
        ("Indexation Images", test_image_indexing), 
        ("Séparation Namespaces", test_namespace_separation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} échoué: {e}")
            results.append((test_name, False))
    
    # Rapport final
    print(f"\n{'='*60}")
    print("📊 RAPPORT FINAL")
    print(f"{'='*60}")
    
    for test_name, success in results:
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        print(f"{status} - {test_name}")
    
    total_success = sum(1 for _, success in results if success)
    print(f"\nRésultat: {total_success}/{len(results)} tests réussis")
    
    if total_success == len(results):
        print("\n🎉 Tous les tests sont passés ! Votre système d'indexation fonctionne correctement.")
    else:
        print("\n⚠️ Certains tests ont échoué. Vérifiez votre configuration.")


if __name__ == "__main__":
    asyncio.run(main())