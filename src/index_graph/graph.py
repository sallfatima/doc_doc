"""Simple RAG avec gestion d'images"""

from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from typing import List, Dict, Any
import re
import json

from shared import retrieval
from shared.utils import load_chat_model
from simple_rag.configuration import RagConfiguration
from simple_rag.state import GraphState, InputState


def extract_images_from_documents(documents) -> List[Dict[str, Any]]:
    """Extrait les images des documents récupérés"""
    images = []
    
    for doc in documents:
        # Vérifier si le document contient des références d'images
        content = doc.page_content
        metadata = doc.metadata or {}
        
        # Chercher des patterns d'images dans le contenu
        image_patterns = [
            r'!\[.*?\]\((.*?)\)',  # Markdown images: ![alt](url)
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',  # HTML img tags
            r'https?://[^\s]+\.(jpg|jpeg|png|gif|webp|svg)',  # URLs d'images directes
        ]
        
        for pattern in image_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    url = match[0] if match[0] else match[1]
                else:
                    url = match
                
                if url and url not in [img['url'] for img in images]:
                    images.append({
                        'url': url,
                        'source': metadata.get('source', 'Unknown'),
                        'context': content[:200] + '...' if len(content) > 200 else content,
                        'alt_text': extract_alt_text(content, url)
                    })
    
    return images


def extract_alt_text(content: str, image_url: str) -> str:
    """Extrait le texte alternatif d'une image"""
    # Chercher le texte alt dans markdown
    markdown_pattern = rf'!\[([^\]]*)\]\([^)]*{re.escape(image_url)}[^)]*\)'
    match = re.search(markdown_pattern, content)
    if match:
        return match.group(1)
    
    # Chercher le texte alt dans HTML
    html_pattern = rf'<img[^>]+src=["\'][^"\']*{re.escape(image_url)}[^"\']*["\'][^>]+alt=["\']([^"\']+)["\'][^>]*>'
    match = re.search(html_pattern, content, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return ""


def is_image_related_query(question: str) -> bool:
    """Détermine si la question est liée aux images"""
    image_keywords = [
        'image', 'images', 'photo', 'photos', 'picture', 'pictures',
        'diagram', 'diagrams', 'figure', 'figures', 'chart', 'charts',
        'graph', 'graphs', 'screenshot', 'screenshots', 'visual', 'visuals',
        'illustration', 'illustrations', 'schéma', 'schémas', 'graphique',
        'graphiques', 'capture', 'captures', 'visualisation', 'visualisations'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in image_keywords)


def retrieve(state: GraphState, *, config: RagConfiguration) -> Dict[str, Any]: 
    """Retrieve documents and images"""
    print("---RETRIEVE---")
    
    # Extract human messages and concatenate them
    question = " ".join(msg.content for msg in state.messages if isinstance(msg, HumanMessage))
    
    # Retrieval
    with retrieval.make_retriever(config) as retriever:
        documents = retriever.invoke(question)
        
        # Extraire les images des documents
        images = extract_images_from_documents(documents)
        
        # Vérifier si la question est liée aux images
        is_image_query = is_image_related_query(question)
        
        return {
            "documents": documents, 
            "messages": state.messages,
            "images": images,
            "is_image_query": is_image_query
        }


async def generate(state: GraphState, *, config: RagConfiguration):
    """Generate answer with image support"""
    print("---GENERATE---")
    
    messages = state.messages
    documents = state.documents
    images = getattr(state, 'images', [])
    is_image_query = getattr(state, 'is_image_query', False)

    # Utiliser un prompt personnalisé qui gère les images
    if is_image_query and images:
        prompt_template = """Tu es un assistant expert en documentation technique. 
        Réponds à la question de l'utilisateur en utilisant les documents fournis et les images disponibles.

        Documents disponibles:
        {context}

        Images disponibles:
        {images_info}

        Instructions:
        1. Réponds d'abord à la question basée sur les documents
        2. Si des images sont pertinentes, mentionne-les avec leurs URLs
        3. Décris brièvement le contenu des images quand c'est pertinent
        4. Formate ta réponse de manière claire et structurée

        Question: {question}

        Réponse:"""
        
        # Préparer les informations sur les images
        images_info = ""
        if images:
            images_info = "\n".join([
                f"- Image: {img['url']}\n  Source: {img['source']}\n  Description: {img['alt_text']}\n  Contexte: {img['context']}"
                for img in images
            ])
        
        # Construire le contexte
        context = "\n\n".join([doc.page_content for doc in documents])
        question = " ".join(msg.content for msg in messages if isinstance(msg, HumanMessage))
        
        # Créer le prompt final
        formatted_prompt = prompt_template.format(
            context=context,
            images_info=images_info,
            question=question
        )
        
        configuration = RagConfiguration.from_runnable_config(config)
        model = load_chat_model(configuration.model)
        
        response = await model.ainvoke([HumanMessage(content=formatted_prompt)])
        
    else:
        # Utiliser le prompt standard si pas de query image ou pas d'images
        prompt = hub.pull("langchaindoc/simple-rag")
        
        configuration = RagConfiguration.from_runnable_config(config)
        model = load_chat_model(configuration.model)

        # Chain
        rag_chain = prompt | model
        response = await rag_chain.ainvoke({"context": documents, "question": messages})
    
    return {"messages": [response], "documents": documents, "images": images}


# Définir le workflow
workflow = StateGraph(GraphState, input=InputState, config_schema=RagConfiguration)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
graph.name = "SimpleRagWithImages"