"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""
import os
from typing import Optional, List, Dict

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pinecone import Index, Pinecone, ServerlessSpec
from bs4 import BeautifulSoup
from urllib.parse import urljoin



def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""

def load_pinecone_index(index_name: str):
    """
    Charge un index Pinecone existant, ou le crée automatiquement s’il n’existe pas.
    """
    pinecone_client = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )

    indexes = pinecone_client.list_indexes().names()
    print("🔎 Index disponibles :", indexes)

    if index_name not in indexes:
        print(f"⚠️ L'index '{index_name}' n'existe pas. Création...")
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec= ServerlessSpec(
                cloud="aws",  # or "gcp"
                region="us-east-1"
            )
        )
        print(f"✅ Index '{index_name}' créé.")

    return pinecone_client.Index(index_name)

def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)



def extract_figures_from_html(html: str, base_url: str) -> List[Dict]:
    """
    Extrait toutes les figures <figure> d’un HTML, avec :
    - l’URL complète de l’image
    - le texte nettoyé (caption_text)
    - le HTML enrichi (caption_html)

    :param html: contenu HTML brut
    :param base_url: URL de base pour construire les URLs absolues
    :return: liste de dictionnaires (image_url, caption_text)
    """
    soup = BeautifulSoup(html, "html.parser")
    figures_data = []

    for figure in soup.find_all("figure"):
        img_tag = figure.find("img")
        caption_tag = figure.find("figcaption")

        if img_tag and img_tag.get("src"):
            image_url = urljoin(base_url, img_tag["src"])
            caption_text = caption_tag.get_text(separator=" ", strip=True) if caption_tag else ""
           

            figures_data.append({
                "image_url": image_url,
                "caption_text": caption_text
               
            })

    return figures_data
