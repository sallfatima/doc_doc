"""State management for Simple RAG with SmartRetriever support"""

from dataclasses import dataclass, field
from typing import Annotated, List, Dict, Any
from langchain_core.messages import AnyMessage
from langchain_core.documents import Document
from langgraph.graph import add_messages


@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent."""

    messages: Annotated[list[AnyMessage], add_messages]


@dataclass(kw_only=True)
class GraphState(InputState):
    """Represents the state of our graph with SmartRetriever support.

    Attributes:
        messages: List of conversation messages
        documents: List of text documents from retrieval
        images: List of image documents from retrieval  
        is_image_query: Boolean indicating if the query is image-related
    """

    documents: List[Document] = field(default_factory=list)
    """List of text documents retrieved from default namespace"""
    
    images: List[Document] = field(default_factory=list)
    """List of image documents retrieved from 'images' namespace.
    Each document contains:
    - page_content: Image caption/description
    - metadata: {
        'caption': str,
        'image_url': str, 
        'source_url': str,
        'source_type': 'image',
        'namespace': 'images',
        'type': 'image'
      }
    """
    
    is_image_query: bool = field(default=False)
    """Boolean flag indicating if the user query is related to images"""