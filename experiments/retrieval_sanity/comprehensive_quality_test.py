#!/usr/bin/env python3
"""
Comprehensive Quality Test Suite
Tests the conversation system with diverse inputs to identify quality gaps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conversation_loop import ConversationLoop


def print_test_result(test_num, user_input, response, expected_intent, issues):
    """Print formatted test result"""
    print(f"\n{'='*80}")
    print(f"Test {test_num}")
    print(f"{'='*80}")
    print(f"User: {user_input}")
    print(f"Bot: {response}")
    print(f"Expected Intent: {expected_intent}")
    if issues:
        print(f"⚠️  Issues: {', '.join(issues)}")
    else:
        print("✅ No obvious issues")


def analyze_response(user_input, response, expected_intent, expected_keywords=None):
    """Analyze response quality and identify issues"""
    issues = []
    
    # Check for repetition
    words = response.lower().split()
    if len(words) != len(set(words)) and len([w for w in words if words.count(w) > 2]) > 0:
        issues.append("excessive repetition")
    
    # Check for very short responses (might be incomplete)
    if len(words) < 5 and expected_intent not in ["greeting", "farewell", "agreement"]:
        issues.append("too short for context")
    
    # Check for generic responses when specific expected
    generic_phrases = ["yes", "no", "okay", "sure", "i see", "alright"]
    if expected_intent in ["technical_explain", "explain"] and response.lower().strip() in generic_phrases:
        issues.append("too generic for technical question")
    
    # Check for missing expected keywords
    if expected_keywords:
        response_lower = response.lower()
        missing = [kw for kw in expected_keywords if kw not in response_lower]
        if len(missing) == len(expected_keywords):  # None of the keywords present
            issues.append(f"missing expected context ({', '.join(missing[:2])})")
    
    # Check for off-topic responses
    user_lower = user_input.lower()
    response_lower = response.lower()
    
    # Extract key nouns from user input
    user_words = set(user_lower.replace('?', '').replace('.', '').split())
    response_words = set(response_lower.replace('?', '').replace('.', '').split())
    
    # Remove common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where',
                 'who', 'which', 'this', 'that', 'these', 'those', 'i', 'you', 'they',
                 'we', 'it', 'them', 'us', 'my', 'your', 'their', 'our', 'its'}
    
    user_content = user_words - stopwords
    response_content = response_words - stopwords
    
    # Check for topic overlap (allow for explanations that introduce new terms)
    if expected_intent in ["technical_explain", "explain", "question_info"] and len(user_content) > 2:
        overlap = user_content & response_content
        if not overlap and len(user_content) > 3:
            issues.append("possible topic mismatch")
    
    return issues


def main():
    """Run comprehensive quality tests"""
    print("Comprehensive Quality Test Suite")
    print("="*80)
    
    # Initialize conversation loop
    conv = ConversationLoop()
    
    # Test cases with expected intents and keywords
    test_cases = [
        # Greetings and basic interaction
        ("Hello!", "greeting", None),
        ("How are you today?", "question_info", ["good", "well", "fine"]),
        ("Good to meet you", "greeting", None),
        
        # Technical questions - core topics
        ("What is a neural network?", "technical_explain", ["network", "learn", "neuron", "data"]),
        ("How do transformers work?", "technical_explain", ["attention", "transformer", "model"]),
        ("Can you explain backpropagation?", "technical_explain", ["gradient", "learning", "weight", "error"]),
        ("What's the difference between CNN and RNN?", "technical_explain", ["convolutional", "recurrent", "sequence"]),
        
        # Follow-up questions (context-dependent)
        ("Why is that important?", "question_info", None),
        ("Can you give an example?", "question_request", ["example", "for instance"]),
        ("How does that work in practice?", "question_info", ["practice", "real", "application"]),
        
        # Clarification requests
        ("I don't understand", "question_request", ["clarify", "explain", "help"]),
        ("Could you explain that differently?", "question_request", ["explain", "different"]),
        ("What do you mean by that?", "question_info", None),
        
        # Complex multi-part questions
        ("Are neural networks always deep, and do they require lots of data?", "question_info", ["deep", "data", "network"]),
        ("How do I choose between supervised and unsupervised learning for my problem?", "question_info", ["supervised", "unsupervised", "choose"]),
        
        # Edge cases
        ("xyzabc nonsense input", "statement", None),  # Nonsense input
        ("", "statement", None),  # Empty (shouldn't happen but test anyway)
        ("What about purple elephants in machine learning?", "question_info", None),  # Absurd question
        
        # Topic transitions
        ("Thanks, that helps. What about optimization algorithms?", "question_info", ["optimization", "algorithm"]),
        ("Interesting. Let's talk about something else - what are GANs?", "question_info", ["GAN", "generative"]),
        
        # Agreement/disagreement
        ("That makes sense", "agreement", None),
        ("I disagree with that approach", "disagreement", None),
        ("Exactly!", "agreement", None),
        
        # Farewells
        ("Thanks for the help", "farewell", None),
        ("Goodbye", "farewell", None),
        ("See you later", "farewell", None),
    ]
    
    total_tests = len(test_cases)
    issues_found = 0
    issue_types = {}
    
    for i, (user_input, expected_intent, expected_keywords) in enumerate(test_cases, 1):
        if not user_input:  # Skip empty input test
            continue
            
        response = conv.process_user_input(user_input)
        issues = analyze_response(user_input, response, expected_intent, expected_keywords)
        
        print_test_result(i, user_input, response, expected_intent, issues)
        
        if issues:
            issues_found += 1
            for issue in issues:
                issue_types[issue] = issue_types.get(issue, 0) + 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tests: {total_tests}")
    print(f"Tests with issues: {issues_found}")
    print(f"Clean tests: {total_tests - issues_found}")
    print(f"Quality score: {((total_tests - issues_found) / total_tests * 10):.1f}/10")
    
    if issue_types:
        print("\nIssue breakdown:")
        for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {issue_type}: {count}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Identify primary quality gaps
    if "missing expected context" in issue_types and issue_types["missing expected context"] > 3:
        print("🔴 HIGH PRIORITY: Responses frequently miss expected context/keywords")
        print("   → Implement semantic relevance scoring")
    
    if "possible topic mismatch" in issue_types and issue_types["possible topic mismatch"] > 2:
        print("🔴 HIGH PRIORITY: Responses sometimes off-topic")
        print("   → Improve context encoding and response selection")
    
    if "too generic for technical question" in issue_types:
        print("🟡 MEDIUM PRIORITY: Generic responses to technical questions")
        print("   → Add confidence-based fallbacks or better intent matching")
    
    if "too short for context" in issue_types:
        print("🟡 MEDIUM PRIORITY: Some responses too brief")
        print("   → Adjust validation criteria or pattern selection")


if __name__ == "__main__":
    main()
