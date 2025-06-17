#!/usr/bin/env python3
"""
Script pour diagnostiquer l'état de votre index Pinecone
"""

import os
from pinecone import Pinecone

def check_pinecone_index():
    """Vérifier l'état de l'index Pinecone"""
    print("🔍 DIAGNOSTIC PINECONE")
    print("=" * 50)
    
    try:
        # Connecter à Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = os.environ["PINECONE_INDEX_NAME"]
        
        print(f"📂 Index: {index_name}")
        
        # Vérifier que l'index existe
        indexes = pc.list_indexes().names()
        print(f"📋 Index disponibles: {indexes}")
        
        if index_name not in indexes:
            print(f"❌ L'index '{index_name}' n'existe pas!")
            return
        
        # Connecter à l'index
        index = pc.Index(index_name)
        
        # Obtenir les stats générales
        stats = index.describe_index_stats()
        print(f"\n📊 STATISTIQUES GÉNÉRALES:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Dimension: {stats.dimension}")
        
        # Vérifier les namespaces
        print(f"\n📁 NAMESPACES:")
        if hasattr(stats, 'namespaces') and stats.namespaces:
            for namespace, info in stats.namespaces.items():
                namespace_name = namespace if namespace else "(default)"
                print(f"   - {namespace_name}: {info.vector_count} vectors")
        else:
            print("   Aucun namespace trouvé")
        
        # Test de requête simple dans chaque namespace
        print(f"\n🔍 TEST DE REQUÊTE:")
        
        # Test namespace par défaut (texte)
        try:
            query_vector = [0.1] * stats.dimension  # Vecteur de test
            results_default = index.query(
                vector=query_vector,
                top_k=1,
                namespace="",
                include_metadata=True
            )
            print(f"   Namespace '' (texte): {len(results_default.matches)} résultats")
            if results_default.matches:
                print(f"      Exemple: {results_default.matches[0].metadata}")
        except Exception as e:
            print(f"   ❌ Erreur namespace '': {e}")
        
        # Test namespace images
        try:
            results_images = index.query(
                vector=query_vector,
                top_k=1,
                namespace="images",
                include_metadata=True
            )
            print(f"   Namespace 'images': {len(results_images.matches)} résultats")
            if results_images.matches:
                print(f"      Exemple: {results_images.matches[0].metadata}")
        except Exception as e:
            print(f"   ❌ Erreur namespace 'images': {e}")
            
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pinecone_index()