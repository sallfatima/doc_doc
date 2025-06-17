"""State management for the index graph.

This module defines the state structures used in the index graph for
document indexing operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the index graph.
    
    This class defines the structure of the input state for indexing operations,
    which includes the sitemap URL to process.
    """
    
    url_site_map: str
    """URL of the sitemap to process for document indexing."""


@dataclass(kw_only=True) 
class IndexState(InputState):
    """State of the index graph for document indexing operations.
    
    This state tracks the progress of indexing operations including
    URLs discovered from sitemaps and indexing status.
    """
    
    urls_to_index: List[str] = field(default_factory=list)
    """List of URLs discovered from the sitemap that need to be indexed."""
    
    indexed_count: int = field(default=0)
    """Number of URLs that have been successfully indexed."""
    
    failed_urls: List[str] = field(default_factory=list)
    """List of URLs that failed to index."""
    
    status: str = field(default="pending")
    """Current status of the indexing operation: pending, processing, completed, failed."""