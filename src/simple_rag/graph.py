### Nodes

from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from shared import retrieval
from shared.utils import load_chat_model
from simple_rag.configuration import RagConfiguration
from simple_rag.state import GraphState, InputState
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate


async def retrieve(state: GraphState, *, config: RagConfiguration) -> dict[str, list[str] | str]: 
    """Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    # Extract human messages and concatenate them
    question = " ".join(msg.content for msg in state.messages if isinstance(msg, HumanMessage))
    if not question:
        raise ValueError("Empty question: did you pass a HumanMessage?")


    # Retrieval
    async with retrieval.make_retriever(config) as retriever: 
        documents = await retriever.ainvoke(question)
        if not documents:
            raise ValueError("No documents retrieved from retriever.")
        return {"documents": documents, "message": state.messages}
 

async def generate(state: GraphState, *, config: RagConfiguration):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    messages = state.messages
    documents = state.documents


    question = " ".join(msg.content for msg in messages if isinstance(msg, HumanMessage))


    # Construire le prompt structuré type Chat
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant. Only use the context to answer."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    configuration = RagConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

 
    # Construire la chaîne RAG : prompt | modèle
    rag_chain = prompt | model

    # Lancer la génération
    response = await rag_chain.ainvoke({
        "context": "\n".join(documents),
        "question": question
    })
    return {
        "messages": [response],  
        "documents": documents
    }

  


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
graph.name = "SimpleRag"
