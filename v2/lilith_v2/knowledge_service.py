"""
Knowledge Augmentation Service for Lilith v2.

This module provides external knowledge lookup capabilities to "fill gaps"
in the system's internal knowledge graph. It connects the cognitive stage
to the outside world (Wikipedia, Dictionaries, etc.).

Design:
- Pluggable providers (Wiki, WordNet, Custom).
- Returns structured 'KnowledgeFragments' that can be encoded and learned.
- Robust against missing dependencies (fails soft).
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeFragment:
    """A unit of external knowledge returned by a provider."""
    source: str
    content: str  # The actual fact/definition/summary
    identifier: str  # Title, word, or ID
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class KnowledgeProvider(ABC):
    @abstractmethod
    def lookup(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[KnowledgeFragment]:
        pass

class WikipediaProvider(KnowledgeProvider):
    """Fetches summaries from Wikipedia."""
    def __init__(self, timeout: float = 3.0):
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.user_agent = "Lilith/2.0 (Cognitive Architecture)"
        self.timeout = timeout

    def lookup(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[KnowledgeFragment]:
        # Simple query cleaning - v2 relies less on regex magic, more on precise intent
        # But we still need a string for the API.
        if not query:
            return []
            
        # Basic heuristic: Use the full query or extract the likely subject if provided in context
        # For now, simplistic approach similar to v1 but cleaner
        search_term = self._clean_query(query)
        if not search_term:
            return []

        try:
            url = self.base_url + quote(search_term)
            headers = {'User-Agent': self.user_agent}
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            
            if resp.status_code == 200:
                data = resp.json()
                if 'extract' in data and data.get('type') != 'disambiguation':
                    return [KnowledgeFragment(
                        source='wikipedia',
                        content=data['extract'],
                        identifier=data.get('title', search_term),
                        confidence=0.75,
                        metadata={'url': data.get('content_urls', {}).get('desktop', {}).get('page')}
                    )]
        except Exception as e:
            logger.debug(f"Wikipedia lookup failed: {e}")
        
        return []

    def _clean_query(self, query: str) -> str:
        # Strip common question prefixes (Could be replaced by an LLM call or trained model later)
        # Keeping it simple for the architectural skeleton
        q = query.lower()
        starts = ["what is ", "who is ", "tell me about ", "define "]
        for s in starts:
            if q.startswith(s):
                return query[len(s):].strip("?.! ")
        return query.strip("?.! ")

class DictionaryProvider(KnowledgeProvider):
    """Fetches definitions from Free Dictionary API."""
    def __init__(self, timeout: float = 3.0):
        self.base_url = "https://api.dictionaryapi.dev/api/v2/entries/en/"
        self.timeout = timeout

    def lookup(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[KnowledgeFragment]:
        term = self._extract_word(query)
        if not term:
            return []
            
        try:
            url = self.base_url + quote(term)
            resp = requests.get(url, timeout=self.timeout)
            
            fragments = []
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    word = entry.get('word', term)
                    for meaning in entry.get('meanings', [])[:2]:
                        part_of_speech = meaning.get('partOfSpeech', 'unknown')
                        for definition in meaning.get('definitions', [])[:1]:
                            def_text = definition.get('definition')
                            if def_text:
                                fragments.append(KnowledgeFragment(
                                    source='dictionary',
                                    content=f"{word} ({part_of_speech}): {def_text}",
                                    identifier=word,
                                    confidence=0.85,
                                    metadata={'part_of_speech': part_of_speech}
                                ))
            return fragments
        except Exception as e:
            logger.debug(f"Dictionary lookup failed: {e}")
            return []

    def _extract_word(self, query: str) -> Optional[str]:
        # Very single-word biased
        words = query.strip().split()
        if len(words) == 1:
            return words[0]
        if query.lower().startswith("define "):
            return query[7:].strip("?.! ")
        return None

class KnowledgeService:
    """
    Orchestrates knowledge providers to find info when internal confidence is low.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.providers: List[KnowledgeProvider] = [
            DictionaryProvider(),
            WikipediaProvider()
        ]

    def search(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[KnowledgeFragment]:
        if not self.enabled or not query:
            return []

        results = []
        for provider in self.providers:
            # If we already have high quality results, maybe skip others?
            # For now, gather all.
            try:
                fragments = provider.lookup(query, context)
                results.extend(fragments)
            except Exception as e:
                logger.error(f"Provider {type(provider).__name__} error: {e}")
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
