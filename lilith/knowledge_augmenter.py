"""
Knowledge Augmentation Module

Provides external knowledge lookup capabilities for filling gaps in the pattern store.
When pattern retrieval has low confidence, queries external sources (Wikipedia, etc.)
to augment the system's knowledge base.

This implements the multi-modal architecture vision: external data sources as
input modalities that feed into the same semantic processing pipeline.
"""

import re
from typing import Optional, Tuple, Dict, Any, List
import requests
from urllib.parse import quote

# Try to import NLTK WordNet (optional)
try:
    import nltk
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
    
    # Ensure WordNet data is downloaded
    try:
        wn.synsets('test')  # Test if data is available
    except LookupError:
        print("📥 Downloading WordNet data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet
except ImportError:
    WORDNET_AVAILABLE = False
    wn = None


class WikipediaLookup:
    """
    Wikipedia API interface for knowledge extraction.
    
    Uses Wikipedia's REST API to fetch article summaries without authentication.
    Extracts clean, factual information suitable for pattern learning.
    
    Supports BNN-based topic extraction via TopicExtractor when available.
    """
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.user_agent = "Lilith/1.0 (Educational neuro-symbolic AI)"
        self.topic_extractor = None  # Set by KnowledgeAugmenter.set_topic_extractor()
        
    def lookup(self, query: str, conversation_history: str = "") -> Optional[Dict[str, Any]]:
        """
        Look up a query on Wikipedia and return structured information.
        
        Args:
            query: Search query (will be used as article title)
            conversation_history: Recent conversation context for disambiguation
            
        Returns:
            Dict with keys: 'title', 'extract', 'url', 'confidence'
            or None if not found
        """
        # Clean query for Wikipedia article title format
        # Uses BNN-based TopicExtractor if available, else falls back to regex
        cleaned_query = self._clean_query(query)
        
        if not cleaned_query:
            return None
        
        # Build full context from query + conversation history
        full_context = query
        if conversation_history:
            full_context = conversation_history + " " + query
            
        # Try exact title match first (with context for disambiguation)
        result = self._fetch_article(cleaned_query, context=full_context)
        
        if result:
            # Report success to topic extractor for learning
            if self.topic_extractor:
                self.topic_extractor.update_success(cleaned_query, True)
            return result
            
        # If no exact match, try search API
        result = self._search_and_fetch(cleaned_query, context=full_context)
        
        # Report success/failure to topic extractor
        if self.topic_extractor:
            self.topic_extractor.update_success(cleaned_query, result is not None)
        
        return result
    
    def _clean_query(self, query: str) -> str:
        """
        Clean user query to Wikipedia article title format.
        
        If TopicExtractor is available, uses BNN-based semantic matching.
        Otherwise falls back to regex patterns.
        
        Examples:
            "What is machine learning?" -> "machine learning"
            "Tell me about Python" -> "Python"
            "Who is Ada Lovelace?" -> "Ada Lovelace"
            "Do you know about dogs?" -> "Dogs" (via BNN or regex)
        """
        # Try BNN-based topic extraction first
        if self.topic_extractor:
            topic, score = self.topic_extractor.extract_topic(query)
            if topic:
                print(f"  🧠 BNN extracted topic: '{topic}' (score: {score:.3f})")
                return topic
        
        # Fallback to regex-based cleaning
        return self._regex_clean_query(query)
    
    def _regex_clean_query(self, query: str) -> str:
        """
        Fallback regex-based query cleaning.
        
        Used when TopicExtractor is not available or finds no match.
        """
        import re
        
        query_lower = query.lower().strip('?!.')
        
        # Handle polar questions: "is X a Y?" or "are X Y?"
        # We want to extract X (the subject being asked about)
        polar_patterns = [
            r'^(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+?)\s+(?:a|an)\s+.+$',  # is X a Y?
            r'^(?:is|are|was|were)\s+(.+?)\s+(?:a|an)\s+.+$',  # is X a Y? (no article before X)
            r'^(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+?)\s+\w+$',  # is X adjective?
            r'^(?:do|does|did)\s+(?:a|an|the)?\s*(.+?)\s+(?:have|eat|live|fly|swim|run|walk|talk|speak)\b.+$',  # does X verb?
        ]
        
        for pattern in polar_patterns:
            match = re.match(pattern, query_lower)
            if match:
                subject = match.group(1).strip()
                # Make sure we got something meaningful (not just articles or pronouns)
                if subject and subject not in ('a', 'an', 'the', 'it', 'this', 'that', 'you', 'i', 'we', 'they'):
                    return subject.title()
        
        # Handle "do you know about X?" pattern specifically
        know_about_match = re.match(r'^(?:do\s+you\s+know\s+(?:about|of)|tell\s+me\s+about|what\s+(?:is|are))\s+(.+)$', query_lower)
        if know_about_match:
            subject = know_about_match.group(1).strip()
            # Remove trailing question words
            subject = re.sub(r'\s*\?$', '', subject)
            if subject and subject not in ('a', 'an', 'the', 'it', 'this', 'that'):
                return subject.title()
        
        # Remove question words and conversational phrases
        question_words = [
            'what', 'who', 'where', 'when', 'why', 'how', 'which', 
            'is', 'are', 'was', 'were', 'am',
            'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'tell', 'me', 'about', 'know', 'you', 'your',
            'a', 'an', 'the',
            'like', 'love', 'hate', 'want', 'need',  # Preference/desire verbs
        ]
        
        words = query_lower.split()
        cleaned = [w for w in words if w not in question_words]
        
        if not cleaned:
            return ""
            
        # Capitalize first letter of each significant word (title case)
        result = ' '.join(cleaned)
        return result.title()
    
    def _fetch_article(self, title: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Fetch Wikipedia article summary by exact title.
        
        Args:
            title: Article title to fetch
            context: Original query context for disambiguation
        """
        try:
            url = self.base_url + quote(title)
            headers = {'User-Agent': self.user_agent}
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract clean summary (first 2-3 sentences)
                extract = data.get('extract', '')
                
                # Check if this is a disambiguation page
                if self._is_disambiguation_page(extract):
                    print(f"  🔀 Disambiguation page detected for '{title}'")
                    # Try to resolve using context
                    resolved = self._resolve_disambiguation(title, context, extract)
                    if resolved:
                        return resolved
                    else:
                        print(f"  ⚠️  Could not resolve disambiguation - returning None")
                        # Don't return disambiguation page - it's not useful info
                        return None
                
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
    
    def _is_disambiguation_page(self, text: str) -> bool:
        """
        Detect if text is from a Wikipedia disambiguation page.
        
        Args:
            text: Article extract text
            
        Returns:
            True if disambiguation page
        """
        if not text:
            return False
        
        # Common disambiguation indicators
        disambig_patterns = [
            'may refer to:',
            'may also refer to:',
            'can refer to:',
            'is the name of:',
            'disambiguation',
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in disambig_patterns)
    
    def _resolve_disambiguation(
        self, 
        title: str, 
        context: str, 
        disambig_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve disambiguation by using context from query.
        
        Strategy:
        1. Extract disambiguation options from text
        2. Score each option against context words
        3. Fetch the highest-scoring option
        
        Args:
            title: Original title that was disambiguous
            context: Original query for context clues
            disambig_text: Disambiguation page text
            
        Returns:
            Article dict for best match or None
        """
        # Extract context words from original query
        context_words = set(context.lower().split())
        
        # Common words to ignore
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'tell', 'me', 'about', 'you', 'your'}
        context_words = context_words - stop_words
        
        # Parse disambiguation options
        options = self._parse_disambiguation_options(disambig_text)
        
        if not options:
            return None
        
        # Initialize stemmer for better word matching (bird = birds)
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            use_stemming = True
        except ImportError:
            stemmer = None
            use_stemming = False
        
        # Score each option based on context overlap
        scored_options = []
        for option_title, option_desc in options:
            # Count how many context words appear in the option description
            desc_lower = (option_title + " " + option_desc).lower()
            desc_tokens = desc_lower.split()
            
            score = 0
            for word in context_words:
                # Try stem matching first (bird = birds)
                if use_stemming:
                    word_stem = stemmer.stem(word)
                    for token in desc_tokens:
                        token_stem = stemmer.stem(token)
                        if token_stem == word_stem:
                            score += 3  # Highest weight for stem match
                            break
                    else:
                        # No stem match, try exact/partial
                        if word in desc_tokens:
                            score += 2  # Exact word match
                        elif word in desc_lower or any(word in token or token in word for token in desc_tokens):
                            score += 1  # Partial match
                else:
                    # No stemming available - use original logic
                    if word in desc_tokens:
                        score += 2  # Exact word match
                    elif word in desc_lower or any(word in token or token in word for token in desc_tokens):
                        score += 1  # Partial match
            
            scored_options.append((score, option_title, option_desc))
        
        # Sort by score (highest first)
        scored_options.sort(reverse=True, key=lambda x: x[0])
        
        # Debug: Show top scoring options
        if scored_options:
            print(f"  📊 Disambiguation scores:")
            for score, title, desc in scored_options[:3]:
                print(f"     {score} pts: {title} - {desc[:50]}...")
        
        # If top score is > 0, try to fetch that article
        if scored_options and scored_options[0][0] > 0:
            best_title = scored_options[0][1]
            print(f"  ✨ Resolved to: '{best_title}' (score: {scored_options[0][0]})")
            
            # Fetch the resolved article (without context to avoid recursion)
            return self._fetch_article(best_title, context="")
        
        return None
    
    def _parse_disambiguation_options(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse disambiguation page to extract options.
        
        Wikipedia disambiguation pages typically have bullet points like:
        - Python (programming language), a high-level programming language
        - Python (mythology), a serpent in Greek mythology
        
        Args:
            text: Disambiguation page text
            
        Returns:
            List of (title, description) tuples
        """
        options = []
        
        # Split into lines and look for list items
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for patterns like "Title, description" or "Title - description"
            # Common format: "Python (programming), a language"
            match = re.match(r'^[•\-\*]?\s*(.+?)\s*[,\-]\s*(.+)$', line)
            if match:
                title_part = match.group(1).strip()
                desc_part = match.group(2).strip()
                
                # Clean up title (remove extra markers)
                title_part = re.sub(r'^\[?|\]?$', '', title_part)
                
                if title_part and desc_part:
                    options.append((title_part, desc_part))
        
        return options[:10]  # Limit to first 10 options
    
    def _search_and_fetch(self, query: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Search Wikipedia and fetch the top result.
        
        Args:
            query: Search query
            context: Original query for disambiguation resolution
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
                    return self._fetch_article(title, context=context)
            
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


class WiktionaryLookup:
    """
    Wiktionary API interface for word definitions and etymology.
    
    Better than Wikipedia for vocabulary, word meanings, and language questions.
    """
    
    def __init__(self):
        self.base_url = "https://en.wiktionary.org/api/rest_v1/page/definition/"
        self.user_agent = "Lilith/1.0 (Educational neuro-symbolic AI)"
    
    def lookup(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Look up a word definition on Wiktionary.
        
        Args:
            word: The word to define
            
        Returns:
            Dict with keys: 'word', 'definitions', 'part_of_speech', 'source'
            or None if not found
        """
        try:
            # Clean the word (remove "what is", "define", etc.)
            cleaned_word = self._clean_word(word)
            
            if not cleaned_word:
                return None
            
            url = self.base_url + quote(cleaned_word)
            headers = {'User-Agent': self.user_agent}
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Wiktionary returns definitions grouped by language
                if 'en' in data:
                    definitions = self._extract_definitions(data['en'])
                    
                    if definitions:
                        return {
                            'word': cleaned_word,
                            'definitions': definitions,
                            'confidence': 0.85,  # Wiktionary is authoritative for words
                            'source': 'wiktionary'
                        }
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ Wiktionary lookup failed: {e}")
            return None
    
    def _clean_word(self, query: str) -> str:
        """
        Extract the target word from a query.
        
        Examples:
            "What does ephemeral mean?" -> "ephemeral"
            "Define recalcitrant" -> "recalcitrant"
            "ephemeral" -> "ephemeral"
        """
        query = query.lower().strip('?!.')
        
        # Remove common question patterns
        patterns = [
            r'what (?:does|is|are) (.+?) mean',
            r'what\'?s the meaning of (.+)',
            r'define (.+)',
            r'definition of (.+)',
            r'meaning of (.+)',
            r'what (?:does|do) (.+?) mean',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        
        # If no pattern matched, assume the query is just the word
        # Remove articles and common words
        words = query.split()
        if len(words) == 1:
            return words[0]
        
        # Remove "a", "an", "the" from start
        if words and words[0] in ('a', 'an', 'the'):
            words = words[1:]
        
        return ' '.join(words) if words else ""
    
    def _extract_definitions(self, lang_data: List[Dict]) -> List[Dict[str, str]]:
        """
        Extract clean definitions from Wiktionary response.
        
        Returns list of {part_of_speech, definition} dicts.
        """
        definitions = []
        
        for entry in lang_data[:3]:  # Top 3 parts of speech
            pos = entry.get('partOfSpeech', 'unknown')
            
            for defn in entry.get('definitions', [])[:2]:  # Top 2 definitions per POS
                definition_text = defn.get('definition', '')
                
                # Clean HTML tags if any
                definition_text = re.sub(r'<[^>]+>', '', definition_text)
                
                if definition_text:
                    definitions.append({
                        'part_of_speech': pos,
                        'definition': definition_text
                    })
        
        return definitions


class WordNetLookup:
    """
    WordNet interface for synonyms, antonyms, and word relationships.
    
    Uses NLTK's WordNet corpus (offline, fast, no API calls).
    """
    
    def __init__(self):
        self.available = WORDNET_AVAILABLE
    
    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Look up word relationships in WordNet.
        
        Args:
            query: Word or synonym query
            
        Returns:
            Dict with keys: 'word', 'synonyms', 'definition', 'source'
            or None if not found
        """
        if not self.available:
            return None
        
        try:
            # Extract the target word
            word = self._clean_query(query)
            
            if not word:
                return None
            
            # Get synsets for this word
            synsets = wn.synsets(word)
            
            if not synsets:
                return None
            
            # Get the primary synset (most common meaning)
            primary = synsets[0]
            
            # Extract synonyms (lemmas from same synset)
            synonyms = set()
            for synset in synsets[:3]:  # Top 3 senses
                for lemma in synset.lemmas()[:5]:  # Top 5 synonyms per sense
                    syn_word = lemma.name().replace('_', ' ')
                    if syn_word.lower() != word.lower():
                        synonyms.add(syn_word)
            
            # Get antonyms if available
            antonyms = set()
            for synset in synsets[:2]:
                for lemma in synset.lemmas():
                    for ant in lemma.antonyms()[:3]:
                        antonyms.add(ant.name().replace('_', ' '))
            
            # Get definition
            definition = primary.definition()
            
            # Get examples if available
            examples = primary.examples()[:2]
            
            return {
                'word': word,
                'definition': definition,
                'synonyms': list(synonyms)[:8],  # Top 8 synonyms
                'antonyms': list(antonyms)[:5] if antonyms else None,
                'examples': examples if examples else None,
                'confidence': 0.80,  # WordNet is reliable
                'source': 'wordnet'
            }
            
        except Exception as e:
            print(f"  ⚠️ WordNet lookup failed: {e}")
            return None
    
    def _clean_query(self, query: str) -> str:
        """
        Extract target word from query.
        
        Examples:
            "synonym for happy" -> "happy"
            "what's another word for sad" -> "sad"
            "antonym of good" -> "good"
        """
        query = query.lower().strip('?!.')
        
        # Patterns for synonym/antonym queries
        patterns = [
            r'synonym (?:for|of) (.+)',
            r'another word for (.+)',
            r'antonym (?:for|of) (.+)',
            r'opposite of (.+)',
            r'similar (?:to|as) (.+)',
            r'like (.+) but',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        
        # If no pattern, assume it's just the word
        words = query.split()
        if len(words) <= 2:
            return query.replace(' ', '_')  # WordNet uses underscores
        
        return ""


class FreeDictionaryLookup:
    """
    Free Dictionary API interface for definitions with pronunciation and examples.
    
    Good balance of features: definitions, phonetics, usage examples.
    """
    
    def __init__(self):
        self.base_url = "https://api.dictionaryapi.dev/api/v2/entries/en/"
    
    def lookup(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Look up a word in Free Dictionary.
        
        Args:
            word: The word to define
            
        Returns:
            Dict with keys: 'word', 'phonetic', 'definitions', 'examples', 'source'
            or None if not found
        """
        try:
            # Clean the word
            cleaned_word = self._clean_word(word)
            
            if not cleaned_word:
                return None
            
            url = self.base_url + quote(cleaned_word)
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    entry = data[0]
                    
                    # Extract phonetic
                    phonetic = entry.get('phonetic', '')
                    
                    # Extract definitions and examples
                    definitions = []
                    examples = []
                    
                    for meaning in entry.get('meanings', [])[:3]:  # Top 3 parts of speech
                        pos = meaning.get('partOfSpeech', '')
                        
                        for defn in meaning.get('definitions', [])[:2]:  # Top 2 per POS
                            def_text = defn.get('definition', '')
                            example = defn.get('example', '')
                            
                            if def_text:
                                definitions.append({
                                    'part_of_speech': pos,
                                    'definition': def_text
                                })
                            
                            if example:
                                examples.append(example)
                    
                    if definitions:
                        return {
                            'word': cleaned_word,
                            'phonetic': phonetic,
                            'definitions': definitions,
                            'examples': examples[:3] if examples else None,
                            'confidence': 0.82,
                            'source': 'free_dictionary'
                        }
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ Free Dictionary lookup failed: {e}")
            return None
    
    def _clean_word(self, query: str) -> str:
        """Extract the target word from query."""
        query = query.lower().strip('?!.')
        
        # Remove question patterns
        patterns = [
            r'what (?:does|is) (.+?) mean',
            r'define (.+)',
            r'meaning of (.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        
        # Just the word
        words = query.split()
        if words and words[0] in ('a', 'an', 'the'):
            words = words[1:]
        
        return words[0] if words and len(words) == 1 else query


class KnowledgeAugmenter:
    """
    Main knowledge augmentation interface.
    
    Coordinates external knowledge lookups and prepares information
    for integration into the pattern learning system.
    
    Tries multiple sources in order:
    1. WordNet (offline, fast) - for synonyms, word relationships
    2. Wiktionary - for word definitions
    3. Free Dictionary - for definitions with examples
    4. Wikipedia - for general knowledge, concepts
    
    Topic Extraction:
    When a TopicExtractor is provided, uses BNN-based semantic matching
    to extract topics from queries instead of regex patterns.
    """
    
    def __init__(self, enabled: bool = True, topic_extractor=None):
        """
        Args:
            enabled: Whether external lookups are enabled (can be toggled)
            topic_extractor: Optional TopicExtractor for BNN-based topic extraction
        """
        self.enabled = enabled
        self.topic_extractor = topic_extractor
        
        # Initialize all lookup sources
        self.wordnet = WordNetLookup()
        self.wiktionary = WiktionaryLookup()
        self.free_dictionary = FreeDictionaryLookup()
        self.wikipedia = WikipediaLookup()
        
        # Statistics
        self.lookup_count = 0
        self.success_count = 0
        self.source_stats = {
            'wordnet': 0,
            'wiktionary': 0,
            'free_dictionary': 0,
            'wikipedia': 0
        }
    
    def set_topic_extractor(self, topic_extractor) -> None:
        """
        Set the topic extractor for BNN-based query cleaning.
        
        This allows late binding when the encoder isn't available at init time.
        """
        self.topic_extractor = topic_extractor
        if topic_extractor:
            # Also give it to Wikipedia lookup
            self.wikipedia.topic_extractor = topic_extractor

    
    def lookup(self, query: str, conversation_history: str = "", min_confidence: float = 0.6) -> Optional[Tuple[str, float, str]]:
        """
        Look up external knowledge for a query.
        
        Tries sources in order of specificity:
        - WordNet for synonym/antonym queries
        - Wiktionary/Free Dictionary for word definitions
        - Wikipedia for general knowledge
        
        Args:
            query: User's question or statement
            conversation_history: Recent conversation context for disambiguation
            min_confidence: Minimum confidence for accepting results
            
        Returns:
            (response_text, confidence, source) or None if not found
        """
        if not self.enabled:
            return None
        
        self.lookup_count += 1
        query_lower = query.lower()
        
        # Determine query type and try appropriate sources
        
        # 1. Synonym/antonym queries -> WordNet first
        if any(word in query_lower for word in ['synonym', 'antonym', 'another word', 'opposite', 'similar to']):
            result = self._try_wordnet(query, min_confidence)
            if result:
                return result
        
        # 2. Word definition queries -> Try Wiktionary, then Free Dictionary
        if any(word in query_lower for word in ['mean', 'define', 'definition', 'meaning']):
            # Try Wiktionary first (more comprehensive)
            result = self._try_wiktionary(query, min_confidence)
            if result:
                return result
            
            # Fallback to Free Dictionary
            result = self._try_free_dictionary(query, min_confidence)
            if result:
                return result
            
            # Also try WordNet for basic definition
            result = self._try_wordnet(query, min_confidence)
            if result:
                return result
        
        # 3. General knowledge -> Wikipedia (with conversation context)
        result = self._try_wikipedia(query, conversation_history, min_confidence)
        if result:
            return result
        
        # 4. Last resort: try all word sources for any single-word query
        words = query_lower.strip('?!.').split()
        if len(words) <= 2:
            # Try WordNet
            result = self._try_wordnet(query, min_confidence)
            if result:
                return result
            
            # Try dictionaries
            result = self._try_wiktionary(query, min_confidence)
            if result:
                return result
            
            result = self._try_free_dictionary(query, min_confidence)
            if result:
                return result
        
        return None
    
    def _try_wordnet(self, query: str, min_confidence: float) -> Optional[Tuple[str, float, str]]:
        """Try WordNet lookup."""
        try:
            result = self.wordnet.lookup(query)
            
            if result and result['confidence'] >= min_confidence:
                self.success_count += 1
                self.source_stats['wordnet'] += 1
                
                response = self._format_wordnet_response(result)
                
                print(f"  📖 WordNet found: {result['word']}")
                print(f"     Synonyms: {len(result.get('synonyms', []))}, Confidence: {result['confidence']}")
                
                return (response, result['confidence'], result['source'])
        except Exception:
            pass
        
        return None
    
    def _try_wiktionary(self, query: str, min_confidence: float) -> Optional[Tuple[str, float, str]]:
        """Try Wiktionary lookup."""
        try:
            result = self.wiktionary.lookup(query)
            
            if result and result['confidence'] >= min_confidence:
                self.success_count += 1
                self.source_stats['wiktionary'] += 1
                
                response = self._format_wiktionary_response(result)
                
                print(f"  📘 Wiktionary found: {result['word']}")
                print(f"     Definitions: {len(result['definitions'])}, Confidence: {result['confidence']}")
                
                return (response, result['confidence'], result['source'])
        except Exception:
            pass
        
        return None
    
    def _try_free_dictionary(self, query: str, min_confidence: float) -> Optional[Tuple[str, float, str]]:
        """Try Free Dictionary lookup."""
        try:
            result = self.free_dictionary.lookup(query)
            
            if result and result['confidence'] >= min_confidence:
                self.success_count += 1
                self.source_stats['free_dictionary'] += 1
                
                response = self._format_free_dictionary_response(result)
                
                print(f"  📕 Free Dictionary found: {result['word']}")
                print(f"     Definitions: {len(result['definitions'])}, Confidence: {result['confidence']}")
                
                return (response, result['confidence'], result['source'])
        except Exception:
            pass
        
        return None
    
    def _try_wikipedia(self, query: str, conversation_history: str, min_confidence: float) -> Optional[Tuple[str, float, str]]:
        """Try Wikipedia lookup with conversation context."""
        try:
            wiki_result = self.wikipedia.lookup(query, conversation_history=conversation_history)
            
            if wiki_result and wiki_result['confidence'] >= min_confidence:
                self.success_count += 1
                self.source_stats['wikipedia'] += 1
                
                response = self._format_response(wiki_result)
                
                print(f"  🌐 Wikipedia found: {wiki_result['title']}")
                print(f"     Source: {wiki_result['source']}, Confidence: {wiki_result['confidence']}")
                
                return (response, wiki_result['confidence'], wiki_result['source'])
        except Exception:
            pass
        
        return None
    
    def _format_wordnet_response(self, result: Dict[str, Any]) -> str:
        """Format WordNet result into natural response."""
        word = result['word']
        definition = result.get('definition', '')
        synonyms = result.get('synonyms', [])
        antonyms = result.get('antonyms', [])
        examples = result.get('examples', [])
        
        # Build response
        parts = []
        
        # Definition
        if definition:
            parts.append(f"{word.capitalize()}: {definition}")
        
        # Synonyms
        if synonyms:
            syn_list = ', '.join(synonyms[:5])
            parts.append(f"Synonyms: {syn_list}")
        
        # Antonyms
        if antonyms:
            ant_list = ', '.join(antonyms[:3])
            parts.append(f"Antonyms: {ant_list}")
        
        # Example
        if examples:
            parts.append(f"Example: \"{examples[0]}\"")
        
        return '. '.join(parts) + '.' if parts else definition
    
    def _format_wiktionary_response(self, result: Dict[str, Any]) -> str:
        """Format Wiktionary result into natural response."""
        word = result['word']
        definitions = result.get('definitions', [])
        
        if not definitions:
            return f"{word.capitalize()} is a word."
        
        # Use first definition
        first_def = definitions[0]
        pos = first_def.get('part_of_speech', '').lower()
        definition = first_def.get('definition', '')
        
        # Format naturally
        if pos:
            response = f"{word.capitalize()} ({pos}): {definition}"
        else:
            response = f"{word.capitalize()}: {definition}"
        
        # Add second definition if different part of speech
        if len(definitions) > 1:
            second_def = definitions[1]
            if second_def.get('part_of_speech') != first_def.get('part_of_speech'):
                response += f" It can also be a {second_def.get('part_of_speech', 'word')}: {second_def.get('definition', '')}"
        
        return response
    
    def _format_free_dictionary_response(self, result: Dict[str, Any]) -> str:
        """Format Free Dictionary result into natural response."""
        word = result['word']
        phonetic = result.get('phonetic', '')
        definitions = result.get('definitions', [])
        examples = result.get('examples', [])
        
        if not definitions:
            return f"{word.capitalize()} is a word."
        
        # Use first definition
        first_def = definitions[0]
        pos = first_def.get('part_of_speech', '').lower()
        definition = first_def.get('definition', '')
        
        # Format with phonetic if available
        parts = []
        
        if phonetic:
            parts.append(f"{word.capitalize()} {phonetic} ({pos}): {definition}")
        elif pos:
            parts.append(f"{word.capitalize()} ({pos}): {definition}")
        else:
            parts.append(f"{word.capitalize()}: {definition}")
        
        # Add example if available
        if examples:
            parts.append(f"Example: \"{examples[0]}\"")
        
        return '. '.join(parts) + '.'
    
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
            'enabled': self.enabled,
            'sources': self.source_stats,
            'wordnet_available': self.wordnet.available if self.wordnet else False
        }
