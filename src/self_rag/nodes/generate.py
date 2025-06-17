### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        Tu es un assistant intelligent. Utilise uniquement les documents suivants pour répondre à la question.
        {context}

        Question :
        {question}

        Réponse :
        """,
        )

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Post-processing
def format_docs(docs):
    """Format the list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()
