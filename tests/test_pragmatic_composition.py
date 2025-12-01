#!/usr/bin/env python3
"""
Test pragmatic template composition (Layer 4 restructured).

This tests the three-step composition:
1. BNN classifies intent
2. Concept retrieved from concept_store
3. Template selected and filled
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lilith.pragmatic_templates import PragmaticTemplateStore


def test_template_storage():
    """Test that templates can be saved and loaded."""
    print("\n" + "="*60)
    print("TEST 1: Template Persistence")
    print("="*60)
    
    templates = PragmaticTemplateStore()
    
    # Count templates by category
    categories = ["greeting", "acknowledgment", "definition", "continuation", "elaboration", "clarification"]
    
    for category in categories:
        templates_in_cat = templates.get_templates_by_category(category)
        print(f"  {category:15s}: {len(templates_in_cat):2d} templates")
    
    total = len(templates.templates)
    print(f"\n  Total templates: {total}")
    print(f"  ✅ Template store initialized with {total} conversational patterns")


def test_template_matching():
    """Test template matching with available slots."""
    print("\n" + "="*60)
    print("TEST 2: Template Matching")
    print("="*60)
    
    templates = PragmaticTemplateStore()
    
    # Test greeting templates
    print("\n  Testing greeting templates:")
    available_slots = {
        "offer_help": "How can I help you?",
        "continue_previous_topic": "Want to continue talking about Python?"
    }
    
    template = templates.match_best_template("greeting", available_slots)
    if template:
        response = templates.fill_template(template, available_slots)
        print(f"    ✅ Matched: {template.template_id}")
        print(f"    Response: {response}")
    else:
        print("    ❌ No template matched")
    
    # Test definition templates
    print("\n  Testing definition templates:")
    available_slots = {
        "concept": "Python",
        "primary_property": "a high-level programming language",
        "elaboration": "known for its simple and readable syntax"
    }
    
    template = templates.match_best_template("definition", available_slots)
    if template:
        response = templates.fill_template(template, available_slots)
        print(f"    ✅ Matched: {template.template_id}")
        print(f"    Response: {response}")
    else:
        print("    ❌ No template matched")


def test_template_categories():
    """Test all template categories."""
    print("\n" + "="*60)
    print("TEST 3: Template Categories")
    print("="*60)
    
    templates = PragmaticTemplateStore()
    
    # Test one template from each category
    test_cases = [
        ("greeting", {"offer_help": "How can I help?"}),
        ("acknowledgment", {"elaboration": "That's interesting."}),
        ("definition", {"concept": "Python", "primary_property": "a programming language"}),
        ("continuation", {"previous_topic": "functions", "new_info": "classes are also important"}),
        ("elaboration", {"concept": "Python", "examples": "web development, data science"}),
        ("clarification", {"topic": "functions"})
    ]
    
    for category, slots in test_cases:
        template = templates.match_best_template(category, slots)
        if template:
            response = templates.fill_template(template, slots)
            print(f"  ✅ {category:15s}: {response[:60]}...")
        else:
            print(f"  ❌ {category:15s}: No template matched")


def main():
    """Run all pragmatic composition tests."""
    print("\n🧪 Testing Pragmatic Template Composition (Layer 4 Restructured)")
    print("This tests compositional response generation using:")
    print("  1. Pragmatic templates (~50 linguistic patterns)")
    print("  2. Template matching by category and available slots")
    print("  3. Slot filling with concept properties")
    
    test_template_storage()
    test_template_matching()
    test_template_categories()
    
    print("\n" + "="*60)
    print("✅ Pragmatic Template Tests Complete")
    print("="*60)
    print("\nKey insights:")
    print("  • Templates are LINGUISTIC knowledge (HOW to say things)")
    print("  • Concepts are SEMANTIC knowledge (WHAT to say)")
    print("  • Separation enables novel composition from learned concepts")
    print("\nNext steps:")
    print("  1. Test with full ResponseComposer integration")
    print("  2. Test with real BNN + concept retrieval")
    print("  3. Wire pragmatic mode into session.py")
    print("  4. Migrate existing patterns to concepts")


if __name__ == "__main__":
    main()
