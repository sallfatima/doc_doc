"""State management for the simple RAG graph with image support."""

from dataclasses import dataclass, field
from typing import Annotated, List, Dict, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""


@dataclass(kw_only=True)
class GraphState(InputState):
    """Represents the state of our graph with image support.

    Attributes:
        messages: List of conversation messages
        documents: List of retrieved documents
        images: List of extracted images with metadata
        is_image_query: Boolean indicating if the query is image-related
    """

    documents: List[str] = field(default_factory=list)
    """List of retrieved documents"""
    
    images: List[Dict[str, Any]] = field(default_factory=list)
    """List of extracted images with metadata including:
    - url: Image URL
    - source: Source document
    - context: Surrounding text context
    - alt_text: Alternative text description
    """
    
    is_image_query: bool = field(default=False)
    """Boolean flag indicating if the user query is related to images"""