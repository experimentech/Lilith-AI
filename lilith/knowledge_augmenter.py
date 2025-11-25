"""
Knowledge Augmentation Module

Provides external knowledge lookup capabilities for filling gaps in the pattern store.
When pattern retrieval has low confidence, queries external sources (Wikipedia, etc.)
to augment the system's knowledge base.

This implements the multi-modal architecture vision: external data sources as
input modalities that feed into the same semantic processing pipeline.
"""

import re
from typing import Optional, Tuple, Dict, Any
import requests
from urllib.parse import quote


class WikipediaLookup:
    """
    Wikipedia API interface for knowledge extraction.
    
    Uses Wikipedia's REST API to fetch article summaries without authentication.
    Extracts clean, factual information suitable for pattern learning.
    """
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.user_agent = "Lilith/1.0 (Educational neuro-symbolic AI)"
        
    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Look up a query on Wikipedia and return structured information.
        
        Args:
            query: Search query (will be used as article title)
            
        Returns:
            Dict with keys: 'title', 'extract', 'url', 'confidence'
            or None if not found
        """
        # Clean query for Wikipedia article title format
        # Convert "What is machine learning?" -> "Machine learning"
        cleaned_query = self._clean_query(query)
        
        if not cleaned_query:
            return None
            
        # Try exact title match first
        result = self._fetch_article(cleaned_query)
        
        if result:
            return result
            
        # If no exact match, try search API
        return self._search_and_fetch(cleaned_query)
    
    def _clean_query(self, query: str) -> str:
        """
        Clean user query to Wikipedia article title format.
        
        Examples:
            "What is machine learning?" -> "machine learning"
            "Tell me about Python" -> "Python"
            "Who is Ada Lovelace?" -> "Ada Lovelace"
            "Do you know what an apple is?" -> "apple"
        """
        # Remove question words and conversational phrases
        question_words = [
            'what', 'who', 'where', 'when', 'why', 'how', 'which', 
            'is', 'are', 'was', 'were', 'am',
            'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'tell', 'me', 'about', 'know', 'you', 'your',
            'a', 'an', 'the',
        ]
        
        words = query.lower().strip('?!.').split()
        cleaned = [w for w in words if w not in question_words]
        
        if not cleaned:
            return ""
            
        # Capitalize first letter of each significant word (title case)
        result = ' '.join(cleaned)
        return result.title()
    
    def _fetch_article(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Fetch Wikipedia article summary by exact title.
        """
        try:
            url = self.base_url + quote(title)
            headers = {'User-Agent': self.user_agent}
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract clean summary (first 2-3 sentences)
                extract = data.get('extract', '')
                clean_extract = self._extract_summary(extract)
                
                if clean_extract:
                    return {
                        'title': data.get('title', title),
                        'extract': clean_extract,
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        'confidence': 0.75,  # Wikipedia is generally reliable
                        'source': 'wikipedia'
                    }
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ Wikipedia lookup failed: {e}")
            return None
    
    def _search_and_fetch(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search Wikipedia and fetch the top result.
        """
        try:
            # Use Wikipedia search API
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': 1,
                'format': 'json'
            }
            headers = {'User-Agent': self.user_agent}
            
            response = requests.get(search_url, params=params, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if len(data) >= 2 and data[1]:
                    # First result title
                    title = data[1][0]
                    return self._fetch_article(title)
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ Wikipedia search failed: {e}")
            return None
    
    def _extract_summary(self, text: str, max_sentences: int = 2) -> str:
        """
        Extract first N sentences from Wikipedia summary.
        
        Keeps it concise for pattern learning - we want factual nuggets,
        not full articles.
        """
        if not text:
            return ""
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Take first N sentences
        summary = '. '.join(sentences[:max_sentences])
        
        # Ensure it ends with period
        if summary and not summary.endswith('.'):
            summary += '.'
        
        # Limit to reasonable length (pattern learning works best with 20-50 words)
        words = summary.split()
        if len(words) > 50:
            summary = ' '.join(words[:50]) + '...'
        
        return summary


class KnowledgeAugmenter:
    """
    Main knowledge augmentation interface.
    
    Coordinates external knowledge lookups and prepares information
    for integration into the pattern learning system.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: Whether external lookups are enabled (can be toggled)
        """
        self.enabled = enabled
        self.wikipedia = WikipediaLookup()
        self.lookup_count = 0
        self.success_count = 0
        
    def lookup(self, query: str, min_confidence: float = 0.6) -> Optional[Tuple[str, float, str]]:
        """
        Look up external knowledge for a query.
        
        Args:
            query: User's question or statement
            min_confidence: Minimum confidence for accepting results
            
        Returns:
            (response_text, confidence, source) or None if not found
        """
        if not self.enabled:
            return None
        
        self.lookup_count += 1
        
        # Try Wikipedia first (can add other sources later)
        try:
            wiki_result = self.wikipedia.lookup(query)
        except Exception as e:
            # Silently fail on network errors
            return None
        
        if wiki_result and wiki_result['confidence'] >= min_confidence:
            self.success_count += 1
            
            # Format response naturally
            response = self._format_response(wiki_result)
            
            print(f"  🌐 External knowledge found: {wiki_result['title']}")
            print(f"     Source: {wiki_result['source']}, Confidence: {wiki_result['confidence']}")
            
            return (response, wiki_result['confidence'], wiki_result['source'])
        
        return None
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """
        Format external knowledge into natural response.
        
        We want it to sound like a factual statement that can be learned,
        not like a Wikipedia article.
        """
        extract = result['extract']
        
        # If extract is clean and concise, use it directly
        # The teaching mechanism will extract the pattern
        return extract
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about external lookups.
        """
        success_rate = (self.success_count / self.lookup_count * 100) if self.lookup_count > 0 else 0
        
        return {
            'lookups': self.lookup_count,
            'successes': self.success_count,
            'success_rate': f"{success_rate:.1f}%",
            'enabled': self.enabled
        }
