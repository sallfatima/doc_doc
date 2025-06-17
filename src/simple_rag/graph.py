"""Simple RAG optimisé avec SmartRetriever"""

from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from typing import List, Dict, Any
import logging

from shared import retrieval
from shared.utils import load_chat_model
from simple_rag.configuration import RagConfiguration
from simple_rag.state import GraphState, InputState


def categorize_documents(documents) -> Dict[str, List]:
    """Sépare les documents textuels des images basé sur source_type metadata."""
    text_docs = []
    image_docs = []
    
    for doc in documents:
        source_type = doc.metadata.get("source_type", "text")
        if source_type == "image":
            image_docs.append(doc)
        else:
            text_docs.append(doc)
    
    return {
        "text_documents": text_docs,
        "image_documents": image_docs
    }


def is_image_related_query(question: str) -> bool:
    """Détermine si la question est liée aux images"""
    image_keywords = [
        'image', 'images', 'photo', 'photos', 'picture', 'pictures',
        'diagram', 'diagrams', 'figure', 'figures', 'chart', 'charts',
        'graph', 'graphs', 'screenshot', 'screenshots', 'visual', 'visuals',
        'illustration', 'illustrations', 'schéma', 'schémas', 'graphique',
        'montrer', 'afficher', 'voir', 'regarder', 'show', 'display', 'view'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in image_keywords)


def format_images_for_response(image_docs: List, max_images: int = 5) -> str:
    """Formate les documents d'images pour inclusion dans la réponse"""
    if not image_docs:
        return "Aucune image pertinente trouvée."
    
    images_to_show = image_docs[:max_images]
    formatted_images = []
    
    for i, doc in enumerate(images_to_show, 1):
        metadata = doc.metadata
        caption = metadata.get('caption', doc.page_content)
        image_url = metadata.get('image_url', '')
        source_url = metadata.get('source_url', '')
        
        image_info = f"""**Image {i}**: {caption}
- 🔗 URL: {image_url}
- 📄 Source: {source_url}"""
        
        formatted_images.append(image_info)
    
    result = "\n\n".join(formatted_images)
    
    if len(image_docs) > max_images:
        result += f"\n\n*... et {len(image_docs) - max_images} autres images disponibles*"
    
    return result


async def retrieve(state: GraphState, *, config: RagConfiguration) -> Dict[str, Any]: 
    """Retrieve documents using optimized SmartRetriever - FULLY ASYNC"""
    print("---SMART RETRIEVE---")
    
    # Extract human messages and concatenate them
    question = " ".join(msg.content for msg in state.messages if isinstance(msg, HumanMessage))
    if not question:
        raise ValueError("Empty question: did you pass a HumanMessage?")

    # Use SmartRetriever with proper async handling
    async with retrieval.make_retriever(config) as smart_retriever:
        # Use the async method instead of sync
        all_documents = await smart_retriever.aget_relevant_documents(question)
        
        # Categorize documents
        categorized = categorize_documents(all_documents)
        text_documents = categorized["text_documents"]
        image_documents = categorized["image_documents"]
        
        # Determine if image-focused
        is_image_query = is_image_related_query(question)
        
        logging.info(f"Retrieved {len(text_documents)} text docs, {len(image_documents)} image docs")
        
        return {
            "documents": text_documents,
            "messages": state.messages,
            "images": image_documents,
            "is_image_query": is_image_query
        }
        
async def generate(state: GraphState, *, config: RagConfiguration):
    """Generate answer with integrated content"""
    print("---GENERATE---")
    
    messages = state.messages
    text_documents = state.documents
    image_documents = getattr(state, 'images', [])
    is_image_query = getattr(state, 'is_image_query', False)

    # Extract question
    question = " ".join(msg.content for msg in messages if isinstance(msg, HumanMessage))

    # Decide whether to include images
    include_images = is_image_query or len(image_documents) > 0

    if include_images and image_documents:
        # Custom prompt with images
        if is_image_query:
            # Image-focused prompt
            prompt_template = """Tu es un assistant spécialisé dans l'analyse de contenu visuel et textuel.

IMAGES DISPONIBLES:
{images_context}

CONTEXTE TEXTUEL:
{text_context}

INSTRUCTIONS (requête axée images):
1. Présente d'abord les images les plus pertinentes
2. Décris leur contenu et utilité  
3. Enrichis avec le contexte textuel
4. Fournis les URLs des images
5. Focus sur l'aspect visuel

Question: {question}

Réponse centrée sur les éléments visuels:"""
        else:
            # Text-focused with visual support
            prompt_template = """Tu es un assistant expert qui combine information textuelle et support visuel.

INFORMATION PRINCIPALE:
{text_context}

ÉLÉMENTS VISUELS COMPLÉMENTAIRES:
{images_context}

INSTRUCTIONS (requête textuelle avec support visuel):
1. Réponds avec l'information textuelle principale
2. Enrichis avec les éléments visuels pertinents
3. Mentionne les images qui illustrent ta réponse
4. Intègre naturellement texte et images

Question: {question}

Réponse complète:"""

        # Prepare contexts
        text_context = "\n\n".join([doc.page_content for doc in text_documents])
        
        images_context_parts = []
        for doc in image_documents:
            caption = doc.metadata.get('caption', doc.page_content)
            image_url = doc.metadata.get('image_url', '')
            source = doc.metadata.get('source_url', '')
            
            img_context = f"• {caption}"
            if image_url:
                img_context += f" (URL: {image_url})"
            if source:
                img_context += f" [Source: {source}]"
            
            images_context_parts.append(img_context)
        
        images_context = "\n".join(images_context_parts)

        # Create final prompt
        formatted_prompt = prompt_template.format(
            text_context=text_context,
            images_context=images_context,
            question=question
        )

        configuration = RagConfiguration.from_runnable_config(config)
        model = load_chat_model(configuration.model)

        response = await model.ainvoke([HumanMessage(content=formatted_prompt)])

        # Add detailed images section if image-focused
        if is_image_query and image_documents:
            response_content = response.content
            if isinstance(response_content, str):
                images_section = format_images_for_response(image_documents)
                response_content += f"\n\n## 🖼️ Images détaillées:\n\n{images_section}"
                response.content = response_content

    else:
       
        from langchain_core.prompts import ChatPromptTemplate
        
        # Prompt local simple pour éviter les erreurs LangSmith
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un assistant expert en LangChain et LangGraph. 
            Utilise le contexte fourni pour répondre à la question de manière précise et utile.
            
            Contexte:
            {context}"""),
            ("human", "{question}")
        ])
        
        configuration = RagConfiguration.from_runnable_config(config)
        model = load_chat_model(configuration.model)

        # Standard chain
        rag_chain = prompt | model
        response = await rag_chain.ainvoke({
            "context": text_documents, 
            "question": question
        })

    return {
        "messages": [response], 
        "documents": text_documents, 
        "images": image_documents,
        "is_image_query": is_image_query
    }


# Workflow
workflow = StateGraph(GraphState, input=InputState, config_schema=RagConfiguration)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
graph.name = "SimpleRAG"