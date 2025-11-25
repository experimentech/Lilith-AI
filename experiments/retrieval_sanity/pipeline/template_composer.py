"""
TemplateComposer - Syntax-Guided Response Composition

Generates responses by filling templates with concept properties.
Uses pattern matching for intent detection (simple heuristics).

This separates syntax (templates) from semantics (concepts).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import re


@dataclass
class ResponseTemplate:
    """Template for generating responses"""
    template_id: str
    intent: str                    # "definition_query", "how_query", etc.
    patterns: List[str]            # Trigger patterns to match
    template: str                  # "{concept} is {definition}."
    slots: List[str]               # Required slots
    examples: List[str]            # Example responses
    success_score: float = 0.75


class TemplateComposer:
    """
    Composes responses using templates + semantic concepts.
    
    Key insight: Separate structure (templates) from content (concepts).
    """
    
    def __init__(self):
        """Initialize with basic templates"""
        self.templates: Dict[str, ResponseTemplate] = {}
        self._bootstrap_templates()
    
    def _bootstrap_templates(self):
        """Initialize with common templates"""
        
        # Definition query template
        self.templates["definition_query"] = ResponseTemplate(
            template_id="def_001",
            intent="definition_query",
            patterns=["what is", "what are", "define", "definition of"],
            template="{concept} is {definition}.",
            slots=["concept", "definition"],
            examples=[
                "Machine learning is a branch of AI that learns from data.",
                "Neural networks are computing systems inspired by biological neurons."
            ]
        )
        
        # How/mechanism query template
        self.templates["how_query"] = ResponseTemplate(
            template_id="how_001",
            intent="how_query",
            patterns=["how does", "how do", "how is", "how are", "explain how"],
            template="{concept} works by {mechanism}.",
            slots=["concept", "mechanism"],
            examples=[
                "Supervised learning works by training on labeled examples.",
                "Neural networks work by processing information through layers."
            ]
        )
        
        # Multi-property elaboration template
        self.templates["elaboration"] = ResponseTemplate(
            template_id="elab_001",
            intent="elaboration",
            patterns=["tell me more", "elaborate", "explain", "tell me about"],
            template="{concept} {property1}. {property2}",
            slots=["concept", "property1", "property2"],
            examples=[
                "Machine learning enables computers to learn from data. It is used in many applications."
            ]
        )
    
    def match_intent(self, query: str) -> Optional[ResponseTemplate]:
        """
        Match query to appropriate template.
        
        Simple pattern matching. In production with more data, would use
        BNN intent classifier.
        
        Args:
            query: User query
            
        Returns:
            Best matching template or None
        """
        query_lower = query.lower()
        
        # Try each template in priority order
        for template_name in ["definition_query", "how_query", "elaboration"]:
            template = self.templates[template_name]
            
            # Check if any pattern matches
            for pattern in template.patterns:
                if pattern in query_lower:
                    return template
        
        return None
    
    def extract_concept_from_query(self, query: str, template: ResponseTemplate) -> Optional[str]:
        """
        Extract the concept being asked about.
        
        Simple heuristic extraction.
        
        Args:
            query: User query
            template: Matched template
            
        Returns:
            Extracted concept or None
        """
        query_lower = query.lower()
        
        # Remove question patterns to isolate concept
        concept = query_lower
        for pattern in template.patterns:
            concept = concept.replace(pattern, "").strip()
        
        # Remove question marks and extra whitespace
        concept = concept.replace("?", "").strip()
        
        # Handle common question words
        question_words = ["what", "how", "why", "when", "where", "who"]
        words = concept.split()
        concept = " ".join(w for w in words if w not in question_words)
        
        return concept.strip() if concept else None
    
    def fill_template(
        self, 
        template: ResponseTemplate,
        concept_term: str,
        properties: List[str]
    ) -> Optional[str]:
        """
        Fill template slots with concept data.
        
        Args:
            template: Template to fill
            concept_term: The concept name
            properties: List of concept properties
            
        Returns:
            Composed response or None if can't fill
        """
        if not properties:
            return None
        
        # Different filling strategies based on intent
        if template.intent == "definition_query":
            # Use first property as definition
            definition = properties[0]
            
            # Capitalize concept name
            concept_cap = concept_term.capitalize()
            
            return template.template.format(
                concept=concept_cap,
                definition=definition
            )
        
        elif template.intent == "how_query":
            # Find property that describes mechanism/process
            mechanism = self._find_mechanism_property(properties)
            if not mechanism:
                mechanism = properties[0]  # Fallback to first property
            
            concept_cap = concept_term.capitalize()
            
            return template.template.format(
                concept=concept_cap,
                mechanism=mechanism
            )
        
        elif template.intent == "elaboration":
            # Use multiple properties
            if len(properties) < 2:
                # Fall back to single property
                return f"{concept_term.capitalize()} {properties[0]}."
            
            # Format properties as sentences
            prop1 = self._format_as_sentence(properties[0], concept_term)
            prop2 = self._format_as_sentence(properties[1], concept_term)
            
            return template.template.format(
                concept=concept_term.capitalize(),
                property1=prop1,
                property2=prop2
            )
        
        return None
    
    def compose_response(
        self,
        query: str,
        concept_term: str,
        properties: List[str],
        confidence: float = 0.85
    ) -> Optional[Dict[str, any]]:
        """
        Main composition pipeline.
        
        Args:
            query: User query
            concept_term: Matched concept name
            properties: Concept properties to use
            confidence: Confidence in concept match
            
        Returns:
            Dict with response and metadata, or None
        """
        # Match intent
        template = self.match_intent(query)
        if not template:
            return None
        
        # Fill template
        response_text = self.fill_template(template, concept_term, properties)
        if not response_text:
            return None
        
        return {
            "text": response_text,
            "template_id": template.template_id,
            "intent": template.intent,
            "confidence": confidence * template.success_score,
            "is_compositional": True
        }
    
    def _find_mechanism_property(self, properties: List[str]) -> Optional[str]:
        """Find property that describes how something works"""
        # Look for properties with mechanism keywords
        mechanism_keywords = ["works by", "uses", "applies", "processes", "trains", "learns"]
        
        for prop in properties:
            prop_lower = prop.lower()
            if any(kw in prop_lower for kw in mechanism_keywords):
                return prop
        
        return None
    
    def _format_as_sentence(self, property_text: str, concept: str) -> str:
        """
        Format a property as a grammatical sentence.
        
        Simple heuristics.
        """
        prop = property_text.strip()
        
        # If property already mentions the concept, use as-is
        if concept.lower() in prop.lower():
            # Ensure it starts with capital
            return prop[0].upper() + prop[1:] if prop else prop
        
        # Otherwise, add subject
        # Check if property starts with verb
        if prop.split()[0].lower() in ["is", "are", "has", "have", "uses", "works", "learns"]:
            return f"It {prop}"
        else:
            return prop
