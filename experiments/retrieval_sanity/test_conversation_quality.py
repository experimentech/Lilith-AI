"""
Conversation Quality Test - Focus on General Conversation

Tests quality on everyday conversation topics that we actually trained on.
This is what we should be optimizing for!
"""

from conversation_loop import ConversationLoop


def test_conversation_quality():
    """Test quality on general conversation scenarios."""
    
    print("=" * 80)
    print("GENERAL CONVERSATION QUALITY TEST")
    print("=" * 80)
    print()
    
    conv = ConversationLoop()
    
    # Test scenarios based on what's in Conversation.csv
    test_scenarios = [
        {
            "name": "Greetings",
            "exchanges": [
                ("Hi, how are you?", ["fine", "good", "doing", "great"]),
                ("I'm doing well, thanks!", ["good", "great", "glad"]),
            ]
        },
        {
            "name": "Weather Talk",
            "exchanges": [
                ("What do you think about the weather today?", ["weather", "warm", "cold", "nice", "sunny", "rain"]),
                ("It's pretty nice outside", ["yeah", "yes", "nice", "good", "agree"]),
            ]
        },
        {
            "name": "Movies",
            "exchanges": [
                ("Do you like movies?", ["yes", "yeah", "love", "like", "watch"]),
                ("What's your favorite movie?", ["movie", "like", "love", "favorite", "watch"]),
            ]
        },
        {
            "name": "School/Work",
            "exchanges": [
                ("How was school today?", ["good", "fine", "okay", "school", "class"]),
                ("What did you do?", ["class", "study", "homework", "learn"]),
            ]
        },
        {
            "name": "Weekend Plans",
            "exchanges": [
                ("What are you doing this weekend?", ["weekend", "plan", "going", "do", "not sure"]),
                ("I'm going to the beach", ["beach", "fun", "nice", "weather", "good"]),
            ]
        },
        {
            "name": "Gratitude/Farewell",
            "exchanges": [
                ("Thanks for chatting!", ["welcome", "sure", "problem", "glad", "anytime"]),
                ("See you later", ["bye", "later", "see you", "great day", "talk"]),
            ]
        },
    ]
    
    total_tests = 0
    issues = 0
    issue_details = []
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario['name']}")
        print('='*80)
        
        # Reset conversation for each scenario
        conv.history.turns.clear()
        
        for user_input, expected_keywords in scenario['exchanges']:
            response = conv.process_user_input(user_input)
            total_tests += 1
            
            print(f"\nUser: {user_input}")
            print(f"Bot: {response}")
            
            # Check if response contains expected keywords
            response_lower = response.lower()
            has_keyword = any(kw in response_lower for kw in expected_keywords)
            
            # Check if response is reasonable length
            is_reasonable_length = len(response.split()) >= 3
            
            # Flag issues
            has_issue = False
            issue_type = []
            
            if not has_keyword:
                has_issue = True
                issue_type.append(f"missing expected keywords: {expected_keywords}")
            
            if not is_reasonable_length:
                has_issue = True
                issue_type.append("too short")
            
            # Check for obvious mismatches
            if "southern california" in response_lower and "california" not in user_input.lower():
                has_issue = True
                issue_type.append("off-topic response")
            
            if has_issue:
                issues += 1
                issue_details.append({
                    "scenario": scenario['name'],
                    "input": user_input,
                    "response": response,
                    "issues": issue_type
                })
                print(f"⚠️  Issues: {', '.join(issue_type)}")
            else:
                print(f"✅ Good response")
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total exchanges tested: {total_tests}")
    print(f"Exchanges with issues: {issues}")
    print(f"Clean exchanges: {total_tests - issues}")
    print(f"Quality score: {((total_tests - issues) / total_tests * 10):.1f}/10")
    print()
    
    if issue_details:
        print("=" * 80)
        print("DETAILED ISSUES")
        print("=" * 80)
        for i, detail in enumerate(issue_details, 1):
            print(f"\n{i}. {detail['scenario']}")
            print(f"   Input: {detail['input']}")
            print(f"   Response: {detail['response']}")
            print(f"   Issues: {', '.join(detail['issues'])}")
    
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Calculate issue categories
    keyword_issues = sum(1 for d in issue_details if any("keyword" in i for i in d['issues']))
    length_issues = sum(1 for d in issue_details if any("short" in i for i in d['issues']))
    offtopic_issues = sum(1 for d in issue_details if any("off-topic" in i for i in d['issues']))
    
    if keyword_issues > 0:
        print(f"🔴 {keyword_issues} responses missing expected context")
        print("   → Improve semantic matching between user input and response patterns")
    
    if offtopic_issues > 0:
        print(f"🔴 {offtopic_issues} off-topic responses")
        print("   → Raise relevance threshold or improve pattern filtering")
    
    if length_issues > 0:
        print(f"🟡 {length_issues} responses too brief")
        print("   → Adjust validation criteria")
    
    print()


if __name__ == "__main__":
    test_conversation_quality()
