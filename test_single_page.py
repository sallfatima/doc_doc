#!/usr/bin/env python3
"""
Test d'indexation d'une page spécifique avec des images
"""

import asyncio
import requests
from bs4 import BeautifulSoup

def test_page_images(url: str):
    """Test si une page contient des images"""
    print(f"🔍 Test de la page: {url}")
    
    try:
        # Désactiver SSL pour éviter les erreurs
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        response = requests.get(url, verify=False, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Chercher tous les types d'images
        all_images = soup.find_all("img")
        figures = soup.find_all("figure")
        
        print(f"📊 Images totales: {len(all_images)}")
        print(f"📊 Figures: {len(figures)}")
        
        if all_images:
            print("🖼️ Exemples d'images trouvées:")
            for i, img in enumerate(all_images[:5], 1):
                src = img.get('src', 'N/A')
                alt = img.get('alt', 'N/A')[:50]
                print(f"   {i}. SRC: {src}")
                print(f"      ALT: {alt}")
        
        if figures:
            print("🖼️ Exemples de figures trouvées:")
            for i, fig in enumerate(figures[:3], 1):
                img = fig.find('img')
                caption = fig.find('figcaption')
                if img:
                    src = img.get('src', 'N/A')
                    caption_text = caption.get_text(strip=True)[:100] if caption else 'N/A'
                    print(f"   {i}. IMG: {src}")
                    print(f"      CAPTION: {caption_text}")
        
        return len(all_images) > 0
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Test plusieurs pages pour trouver celles avec des images"""
    test_urls = [
        # Pages conceptuelles (plus susceptibles d'avoir des images)
        "https://python.langchain.com/docs/concepts/",
        "https://python.langchain.com/docs/tutorials/rag/",
        "https://python.langchain.com/docs/introduction/",
        
        # Pages LangGraph (beaucoup d'images)
        "https://langchain-ai.github.io/langgraph/concepts/",
        "https://langchain-ai.github.io/langgraph/concepts/low_level/",
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
        
        # Page d'intégration (pour comparaison)
        "https://python.langchain.com/docs/integrations/chat/openai/",
    ]
    
    print("🧪 TEST DE PAGES AVEC IMAGES")
    print("=" * 50)
    
    pages_with_images = []
    
    for url in test_urls:
        print(f"\n{'='*60}")
        has_images = test_page_images(url)
        if has_images:
            pages_with_images.append(url)
            print("✅ Cette page contient des images !")
        else:
            print("❌ Cette page ne contient pas d'images")
    
    print(f"\n📋 RÉSUMÉ:")
    print(f"Pages avec images: {len(pages_with_images)}/{len(test_urls)}")
    
    if pages_with_images:
        print("\n✅ Pages recommandées pour l'indexation:")
        for url in pages_with_images:
            print(f"   - {url}")
    else:
        print("\n⚠️ Aucune page avec images trouvée - vérifiez les URLs")

if __name__ == "__main__":
    main()