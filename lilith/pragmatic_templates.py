"""
Pragmatic Templates Database - Conversational Patterns for Layer 4

This is the SMALL, FIXED-SIZE database that handles conversational flow.
Stores dialogue act templates (~50 total), NOT factual knowledge.

Key distinction:
- pragmatic_templates.db: Conversational patterns (grows slowly, linguistic)
- concept_store.db: Factual knowledge (grows with learning, semantic)

Layer 4 does COMPOSITION, not STORAGE!
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from pathlib import Path


@dataclass
class PragmaticTemplate:
    """A conversational template for dialogue acts"""
    template_id: str
    category: str                  # "greeting", "acknowledgment", "definition", etc.
    intent: str                    # Specific intent within category
    template: str                  # "{greeting}! {topic_acknowledgment}"
    slots: List[str]               # Required slots to fill
    priority: int = 5              # Higher = more preferred (1-10)
    examples: List[str] = None     # Example outputs
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class PragmaticTemplateStore:
    """
    Store of conversational templates for dialogue acts.
    
    This is Layer 4's "linguistic knowledge" - how to structure conversation.
    Factual knowledge lives in ConceptStore instead.
    
    Size: ~50 templates (small, fixed, grows like grammar rules)
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize pragmatic template store.
        
        Args:
            storage_path: JSON file path (optional, uses defaults if None)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.templates: Dict[str, PragmaticTemplate] = {}
        
        # Load or bootstrap
        if self.storage_path and self.storage_path.exists():
            self._load_templates()
        else:
            self._bootstrap_default_templates()
            if self.storage_path:
                self._save_templates()
    
    def _bootstrap_default_templates(self):
        """Bootstrap with default conversational templates"""
        
        # ============================================================
        # GREETING TEMPLATES (5)
        # ============================================================
        
        self.templates["greeting_simple"] = PragmaticTemplate(
            template_id="greeting_simple",
            category="greeting",
            intent="simple_greeting",
            template="Hello! {offer_help}",
            slots=["offer_help"],
            priority=7,
            examples=["Hello! How can I help you?"]
        )
        
        self.templates["greeting_continue_topic"] = PragmaticTemplate(
            template_id="greeting_continue_topic",
            category="greeting",
            intent="greeting_with_context",
            template="Hi! {continue_previous_topic}",
            slots=["continue_previous_topic"],
            priority=8,
            examples=["Hi! Want to continue talking about Python?"]
        )
        
        self.templates["greeting_time_aware"] = PragmaticTemplate(
            template_id="greeting_time_aware",
            category="greeting",
            intent="time_aware_greeting",
            template="{time_greeting}! {offer_help}",
            slots=["time_greeting", "offer_help"],
            priority=6,
            examples=["Good morning! What can I help you with?"]
        )
        
        # ============================================================
        # ACKNOWLEDGMENT TEMPLATES (10)
        # ============================================================
        
        self.templates["ack_simple"] = PragmaticTemplate(
            template_id="ack_simple",
            category="acknowledgment",
            intent="simple_acknowledgment",
            template="I see. {elaboration}",
            slots=["elaboration"],
            priority=7,
            examples=["I see. That's an interesting topic."]
        )
        
        self.templates["ack_understanding"] = PragmaticTemplate(
            template_id="ack_understanding",
            category="acknowledgment",
            intent="show_understanding",
            template="That makes sense. {related_concept}",
            slots=["related_concept"],
            priority=8,
            examples=["That makes sense. It's related to object-oriented programming."]
        )
        
        self.templates["ack_interest"] = PragmaticTemplate(
            template_id="ack_interest",
            category="acknowledgment",
            intent="express_interest",
            template="Interesting! {follow_up_question}",
            slots=["follow_up_question"],
            priority=6,
            examples=["Interesting! What aspect are you most curious about?"]
        )
        
        self.templates["ack_summary"] = PragmaticTemplate(
            template_id="ack_summary",
            category="acknowledgment",
            intent="summarize_understanding",
            template="Got it. {summary}",
            slots=["summary"],
            priority=7,
            examples=["Got it. So we're talking about neural networks."]
        )
        
        self.templates["ack_agreement"] = PragmaticTemplate(
            template_id="ack_agreement",
            category="acknowledgment",
            intent="agree",
            template="Exactly. {elaboration}",
            slots=["elaboration"],
            priority=6,
            examples=["Exactly. That's a key concept in machine learning."]
        )
        
        # ============================================================
        # DEFINITION TEMPLATES (15)
        # ============================================================
        
        self.templates["def_simple"] = PragmaticTemplate(
            template_id="def_simple",
            category="definition",
            intent="simple_definition",
            template="{concept} is {primary_property}.",
            slots=["concept", "primary_property"],
            priority=8,
            examples=["Machine learning is a branch of artificial intelligence."]
        )
        
        self.templates["def_with_elaboration"] = PragmaticTemplate(
            template_id="def_with_elaboration",
            category="definition",
            intent="definition_with_detail",
            template="{concept} is {primary_property}. {elaboration}",
            slots=["concept", "primary_property", "elaboration"],
            priority=9,
            examples=["Python is a high-level programming language. It's known for its readability."]
        )
        
        self.templates["def_with_example"] = PragmaticTemplate(
            template_id="def_with_example",
            category="definition",
            intent="definition_with_example",
            template="{concept} refers to {properties}. For example, {example}",
            slots=["concept", "properties", "example"],
            priority=7,
            examples=["Supervised learning refers to training with labeled data. For example, image classification."]
        )
        
        self.templates["def_essence"] = PragmaticTemplate(
            template_id="def_essence",
            category="definition",
            intent="essential_definition",
            template="In essence, {concept} {description}.",
            slots=["concept", "description"],
            priority=6,
            examples=["In essence, neural networks mimic biological brain structures."]
        )
        
        self.templates["def_functional"] = PragmaticTemplate(
            template_id="def_functional",
            category="definition",
            intent="functional_definition",
            template="{concept} is used to {purpose}. {mechanism}",
            slots=["concept", "purpose", "mechanism"],
            priority=7,
            examples=["Gradient descent is used to optimize neural networks. It iteratively adjusts weights."]
        )
        
        self.templates["def_comparative"] = PragmaticTemplate(
            template_id="def_comparative",
            category="definition",
            intent="comparative_definition",
            template="{concept} is similar to {comparison}, but {distinction}.",
            slots=["concept", "comparison", "distinction"],
            priority=6,
            examples=["Deep learning is similar to machine learning, but uses multi-layer networks."]
        )
        
        # ============================================================
        # CONTINUATION TEMPLATES (10)
        # ============================================================
        
        self.templates["cont_building_on"] = PragmaticTemplate(
            template_id="cont_building_on",
            category="continuation",
            intent="build_on_previous",
            template="Building on {previous_topic}, {new_info}",
            slots=["previous_topic", "new_info"],
            priority=9,
            examples=["Building on what you said about Python, it's also great for data science."]
        )
        
        self.templates["cont_as_mentioned"] = PragmaticTemplate(
            template_id="cont_as_mentioned",
            category="continuation",
            intent="reference_previous",
            template="As you mentioned about {topic}, {related_fact}",
            slots=["topic", "related_fact"],
            priority=8,
            examples=["As you mentioned about machine learning, it requires large datasets."]
        )
        
        self.templates["cont_relates_to"] = PragmaticTemplate(
            template_id="cont_relates_to",
            category="continuation",
            intent="show_connection",
            template="That relates to {concept} because {connection}.",
            slots=["concept", "connection"],
            priority=7,
            examples=["That relates to supervised learning because it uses labeled data."]
        )
        
        self.templates["cont_following_up"] = PragmaticTemplate(
            template_id="cont_following_up",
            category="continuation",
            intent="follow_up",
            template="Following up on {previous_topic}, {additional_info}",
            slots=["previous_topic", "additional_info"],
            priority=7,
            examples=["Following up on neural networks, they can have millions of parameters."]
        )
        
        self.templates["cont_in_context"] = PragmaticTemplate(
            template_id="cont_in_context",
            category="continuation",
            intent="contextualize",
            template="In the context of {context}, {statement}",
            slots=["context", "statement"],
            priority=6,
            examples=["In the context of deep learning, activation functions are crucial."]
        )
        
        # ============================================================
        # ELABORATION TEMPLATES (10)
        # ============================================================
        
        self.templates["elab_example"] = PragmaticTemplate(
            template_id="elab_example",
            category="elaboration",
            intent="provide_example",
            template="For example, {examples}",
            slots=["examples"],
            priority=8,
            examples=["For example, Python is used in web development and data science."]
        )
        
        self.templates["elab_application"] = PragmaticTemplate(
            template_id="elab_application",
            category="elaboration",
            intent="show_application",
            template="This is useful for {applications}.",
            slots=["applications"],
            priority=7,
            examples=["This is useful for image recognition and natural language processing."]
        )
        
        self.templates["elab_related"] = PragmaticTemplate(
            template_id="elab_related",
            category="elaboration",
            intent="show_relations",
            template="It's related to {related_concepts}.",
            slots=["related_concepts"],
            priority=7,
            examples=["It's related to optimization algorithms and backpropagation."]
        )
        
        self.templates["elab_properties"] = PragmaticTemplate(
            template_id="elab_properties",
            category="elaboration",
            intent="list_properties",
            template="{concept} has several key features: {properties}",
            slots=["concept", "properties"],
            priority=6,
            examples=["Python has several key features: readability, extensive libraries, and dynamic typing."]
        )
        
        self.templates["elab_mechanism"] = PragmaticTemplate(
            template_id="elab_mechanism",
            category="elaboration",
            intent="explain_mechanism",
            template="It works by {mechanism}. {details}",
            slots=["mechanism", "details"],
            priority=7,
            examples=["It works by adjusting weights iteratively. This minimizes the error function."]
        )
        
        # ============================================================
        # OPINION TEMPLATES (6)
        # ============================================================
        
        self.templates["opinion_positive"] = PragmaticTemplate(
            template_id="opinion_positive",
            category="opinion",
            intent="express_interest",
            template="I find {topic} fascinating! {elaboration}",
            slots=["topic", "elaboration"],
            priority=9,
            examples=["I find birds fascinating! Their ability to fly and adapt to diverse environments is remarkable."]
        )
        
        self.templates["opinion_neutral"] = PragmaticTemplate(
            template_id="opinion_neutral",
            category="opinion",
            intent="express_balanced_view",
            template="{topic} is an interesting subject. {elaboration}",
            slots=["topic", "elaboration"],
            priority=8,
            examples=["Machine learning is an interesting subject. It has both powerful applications and important ethical considerations."]
        )
        
        self.templates["opinion_analytical"] = PragmaticTemplate(
            template_id="opinion_analytical",
            category="opinion",
            intent="express_analysis",
            template="What I find particularly interesting about {topic} is {aspect}. {elaboration}",
            slots=["topic", "aspect", "elaboration"],
            priority=9,
            examples=["What I find particularly interesting about quantum computing is its potential to solve currently intractable problems."]
        )
        
        self.templates["opinion_comparative"] = PragmaticTemplate(
            template_id="opinion_comparative",
            category="opinion",
            intent="express_preference",
            template="I appreciate both {option1} and {option2}, though they serve different purposes. {elaboration}",
            slots=["option1", "option2", "elaboration"],
            priority=8,
            examples=["I appreciate both Python and JavaScript, though they serve different purposes. Python excels in data science while JavaScript dominates web development."]
        )
        
        self.templates["opinion_thoughtful"] = PragmaticTemplate(
            template_id="opinion_thoughtful",
            category="opinion",
            intent="express_consideration",
            template="That's a thought-provoking question about {topic}. {perspective}",
            slots=["topic", "perspective"],
            priority=7,
            examples=["That's a thought-provoking question about artificial consciousness. It touches on deep philosophical questions about the nature of awareness."]
        )
        
        self.templates["opinion_curious"] = PragmaticTemplate(
            template_id="opinion_curious",
            category="opinion",
            intent="express_curiosity",
            template="The interesting thing about {topic} is {aspect}. {question}",
            slots=["topic", "aspect", "question"],
            priority=8,
            examples=["The interesting thing about black holes is their extreme gravitational effects. What aspects are you most curious about?"]
        )
        
        # ============================================================
        # CLARIFICATION TEMPLATES (2)
        # ============================================================
        
        self.templates["clarify_ambiguous"] = PragmaticTemplate(
            template_id="clarify_ambiguous",
            category="clarification",
            intent="ask_for_specifics",
            template="Could you clarify - are you asking about {option1} or {option2}?",
            slots=["option1", "option2"],
            priority=8,
            examples=["Could you clarify - are you asking about Python the language or Python the snake?"]
        )
        
        self.templates["clarify_context"] = PragmaticTemplate(
            template_id="clarify_context",
            category="clarification",
            intent="need_context",
            template="I'd like to help, but could you provide more context about {topic}?",
            slots=["topic"],
            priority=7,
            examples=["I'd like to help, but could you provide more context about what you're trying to learn?"]
        )
        
        # ============================================================
        # CONFIRMATION TEMPLATES (4) - Yes/No questions
        # ============================================================
        
        self.templates["confirm_yes"] = PragmaticTemplate(
            template_id="confirm_yes",
            category="confirmation",
            intent="affirm_relationship",
            template="Yes, {subject} is {relationship}. {elaboration}",
            slots=["subject", "relationship", "elaboration"],
            priority=9,
            examples=["Yes, a wyvern is a type of dragon. It has two legs and two wings instead of four legs."]
        )
        
        self.templates["confirm_yes_simple"] = PragmaticTemplate(
            template_id="confirm_yes_simple",
            category="confirmation",
            intent="affirm_simple",
            template="Yes, that's correct - {subject} is indeed {relationship}.",
            slots=["subject", "relationship"],
            priority=8,
            examples=["Yes, that's correct - a wyvern is indeed a type of dragon."]
        )
        
        self.templates["confirm_partial"] = PragmaticTemplate(
            template_id="confirm_partial",
            category="confirmation",
            intent="partial_affirm",
            template="In a way, yes. {subject} {relationship}, though {caveat}.",
            slots=["subject", "relationship", "caveat"],
            priority=7,
            examples=["In a way, yes. A wyvern is related to dragons, though it's typically depicted differently."]
        )
        
        self.templates["confirm_uncertain"] = PragmaticTemplate(
            template_id="confirm_uncertain",
            category="confirmation",
            intent="express_uncertainty",
            template="Based on what I know, {subject} {relationship}. {qualifier}",
            slots=["subject", "relationship", "qualifier"],
            priority=6,
            examples=["Based on what I know, a wyvern is considered a dragon subtype. Different sources may vary on this."]
        )
        
        # ============================================================
        # TEACHING/ACKNOWLEDGMENT TEMPLATES (4) - User teaching the system
        # ============================================================
        
        self.templates["teaching_acknowledge"] = PragmaticTemplate(
            template_id="teaching_acknowledge",
            category="teaching",
            intent="acknowledge_learning",
            template="I see, so {subject} is {relationship}. That's useful to know!",
            slots=["subject", "relationship"],
            priority=9,
            examples=["I see, so a wyvern is a type of dragon. That's useful to know!"]
        )
        
        self.templates["teaching_confirm_stored"] = PragmaticTemplate(
            template_id="teaching_confirm_stored",
            category="teaching",
            intent="confirm_storage",
            template="Got it - I've noted that {subject} is {relationship}. Thanks for teaching me!",
            slots=["subject", "relationship"],
            priority=8,
            examples=["Got it - I've noted that a wyvern is a type of dragon. Thanks for teaching me!"]
        )
        
        self.templates["teaching_expand"] = PragmaticTemplate(
            template_id="teaching_expand",
            category="teaching",
            intent="expand_knowledge",
            template="Thanks for clarifying that {subject} is {relationship}. Is there anything else that distinguishes it?",
            slots=["subject", "relationship"],
            priority=7,
            examples=["Thanks for clarifying that a wyvern is a type of dragon. Is there anything else that distinguishes it?"]
        )
        
        self.templates["teaching_connect"] = PragmaticTemplate(
            template_id="teaching_connect",
            category="teaching",
            intent="connect_concepts",
            template="That makes sense - {subject} being {relationship} connects with {related_concept}.",
            slots=["subject", "relationship", "related_concept"],
            priority=7,
            examples=["That makes sense - a wyvern being a type of dragon connects with what I know about mythical creatures."]
        )
        
    def get_template(self, template_id: str) -> Optional[PragmaticTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: str) -> List[PragmaticTemplate]:
        """Get all templates in a category"""
        return [t for t in self.templates.values() if t.category == category]
    
    def get_templates_by_intent(self, intent: str) -> List[PragmaticTemplate]:
        """Get all templates matching an intent"""
        return [t for t in self.templates.values() if t.intent == intent]
    
    def match_best_template(
        self, 
        category: str, 
        available_slots: Dict[str, str]
    ) -> Optional[PragmaticTemplate]:
        """
        Find best template for category with available slot data.
        
        Args:
            category: Template category ("greeting", "definition", etc.)
            available_slots: Slots we can fill {"concept": "Python", "property": "..."}
            
        Returns:
            Best matching template or None
        """
        candidates = self.get_templates_by_category(category)
        
        # Filter to templates where we can fill all required slots
        fillable = []
        for template in candidates:
            if all(slot in available_slots for slot in template.slots):
                fillable.append(template)
        
        if not fillable:
            return None
        
        # Return highest priority
        fillable.sort(key=lambda t: t.priority, reverse=True)
        return fillable[0]
    
    def fill_template(
        self,
        template: PragmaticTemplate,
        slot_values: Dict[str, str]
    ) -> str:
        """
        Fill template with slot values.
        
        Args:
            template: Template to fill
            slot_values: Values for each slot
            
        Returns:
            Filled template string
        """
        result = template.template
        
        for slot, value in slot_values.items():
            placeholder = "{" + slot + "}"
            if placeholder in result:
                result = result.replace(placeholder, value)
        
        return result
    
    def _save_templates(self):
        """Save templates to JSON"""
        if not self.storage_path:
            return
        
        data = {}
        for template_id, template in self.templates.items():
            data[template_id] = {
                "template_id": template.template_id,
                "category": template.category,
                "intent": template.intent,
                "template": template.template,
                "slots": template.slots,
                "priority": template.priority,
                "examples": template.examples
            }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_templates(self):
        """Load templates from JSON"""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        for template_id, template_data in data.items():
            self.templates[template_id] = PragmaticTemplate(**template_data)
    
    def get_stats(self) -> Dict:
        """Get statistics about templates"""
        categories = {}
        for template in self.templates.values():
            categories[template.category] = categories.get(template.category, 0) + 1
        
        return {
            "total_templates": len(self.templates),
            "by_category": categories
        }
    
    def save(self, path: str):
        """
        Save templates to JSON file.
        
        Args:
            path: Path to save file (will add .json if not present)
        """
        import json
        from pathlib import Path
        
        # Ensure .json extension
        if not path.endswith('.json'):
            path = path + '.json'
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert templates to serializable format
        data = {
            "templates": [
                {
                    "template_id": t.template_id,
                    "category": t.category,
                    "intent": t.intent,
                    "template": t.template,
                    "slots": t.slots,
                    "priority": t.priority,
                    "examples": t.examples
                }
                for t in self.templates.values()
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'PragmaticTemplateStore':
        """
        Load templates from JSON file.
        
        Args:
            path: Path to load file
            
        Returns:
            PragmaticTemplateStore instance
        """
        import json
        
        # Ensure .json extension
        if not path.endswith('.json'):
            path = path + '.json'
        
        store = cls()
        store.templates = {}  # Clear defaults
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for t_data in data["templates"]:
            template = PragmaticTemplate(
                template_id=t_data["template_id"],
                category=t_data["category"],
                intent=t_data["intent"],
                template=t_data["template"],
                slots=t_data["slots"],
                priority=t_data["priority"],
                examples=t_data["examples"]
            )
            store.templates[template.template_id] = template
        
        return store
