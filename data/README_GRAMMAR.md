# Grammar Bootstrap Dataset

This directory contains a comprehensive grammatical training dataset designed to bootstrap Lilith's grammar capabilities.

## Overview

The `grammar_bootstrap.txt` file contains **360 carefully crafted sentences** covering:

- **Simple, compound, and complex statements**
- **All major question types** (WH-questions, yes/no, choice)
- **Imperative statements** (commands/requests)
- **Conditional statements**
- **Comparative and superlative forms**
- **All major tenses** (present, past, future, perfect)
- **Modal verbs** (can, could, may, might, must, should, would)
- **Passive voice constructions**
- **Gerunds and infinitives**
- **Relative clauses**
- **Quantifiers and determiners**
- **Adverbial phrases**
- **Conversational patterns** (acknowledgments, clarifications, transitions)
- **Technical explanations**
- **Analogies and metaphors**
- **Expressing emotions, opinions, and uncertainty**

## Usage

### Quick Bootstrap

To bootstrap Lilith's grammar with the training dataset:

```bash
python bootstrap_grammar.py
```

This will:
1. Load all 360 training sentences
2. Process them through Lilith's learning pipeline
3. Enable syntax stage, reasoning stage, and neuroplasticity
4. Learn grammatical patterns and structures
5. Test the results with sample queries

### Custom Data Directory

To use a specific data directory:

```bash
python bootstrap_grammar.py --data-dir /path/to/data
```

### Custom User ID

To use a specific user ID for the bootstrap session:

```bash
python bootstrap_grammar.py --user-id my_bootstrap_user
```

## What Gets Learned

After bootstrapping, Lilith will have:

1. **Response Patterns**: 360+ grammatical sentence patterns stored
2. **Syntax Patterns**: BNN-encoded POS sequences for structure recognition
3. **Intent Clusters**: BNN-based clustering of query types
4. **Vocabulary**: All words and their contexts
5. **Concept Associations**: Semantic relationships between terms
6. **PMFlow Activations**: Trained embeddings for grammatical structures

## Benefits

### Before Bootstrap
- Limited grammatical pattern recognition
- Generic fallback responses
- Poor sentence structure in generated responses
- Weak adaptation to different question types

### After Bootstrap
- Recognizes diverse sentence structures
- Better pattern matching for various query types
- More grammatical response generation
- Stronger foundation for learning from conversations
- Improved syntax stage performance

## Integration with Lilith

The bootstrap process works with all of Lilith's learning layers:

### 1. **Intake Layer** (NoiseNormalizer)
   - Learns common sentence patterns
   - Improves query understanding

### 2. **Reasoning Layer** (ReasoningStage)
   - Better concept activation from diverse structures
   - Improved intent resolution
   - More effective deliberation

### 3. **Syntax Layer** (SyntaxStage)
   - Learns POS sequences via PMFlow BNN
   - Builds grammatical pattern library
   - Improves grammar correction capabilities

### 4. **Response Layer** (ResponseComposer)
   - More patterns for retrieval and adaptation
   - Better template matching
   - Improved compositional generation

## Dataset Structure

The training file is organized by grammatical category with clear section headers:

```
# ============================================================================
# SIMPLE STATEMENTS (Subject-Verb-Object)
# ============================================================================

I understand your question.
You asked about machine learning.
...

# ============================================================================
# WH QUESTIONS
# ============================================================================

What is machine learning?
How does this work?
...
```

Comments (lines starting with `#`) and blank lines are automatically skipped during loading.

## Maintenance

### Adding New Patterns

To add new grammatical patterns:

1. Edit `grammar_bootstrap.txt`
2. Add sentences under the appropriate category
3. Or create a new category section
4. Re-run the bootstrap script

### Testing Changes

After adding new patterns:

```bash
# Clean previous bootstrap
rm -rf data/test_grammar

# Run with test data directory
python bootstrap_grammar.py --data-dir data/test_grammar

# Check the results
```

## Performance

- **Processing Speed**: ~25-30 sentences/second
- **Total Bootstrap Time**: ~15-20 seconds for 360 sentences
- **Memory Usage**: Depends on PMFlow encoder size
- **Storage**: Patterns, embeddings, and vocabulary stored in SQLite

## Technical Details

### What Happens During Bootstrap

1. **Load Dataset**: Read and parse `grammar_bootstrap.txt`
2. **Initialize Session**: Create LilithSession with grammar enabled
3. **Process Each Sentence**: 
   - Parse through intake layer
   - Generate PMFlow embeddings
   - Store as response pattern
   - Update syntax patterns
   - Build vocabulary
   - Create concept associations
4. **Apply Plasticity**: Reinforce learned patterns
5. **Test**: Validate with sample queries

### Passive Mode Learning

The bootstrap uses `passive_mode=True` for processing, which means:
- Sentences are learned without generating responses
- Faster processing
- Focus on pattern storage rather than interaction

## Troubleshooting

### "Grammar dataset not found"
- Ensure `data/grammar_bootstrap.txt` exists
- Check file permissions

### "Failed on: [sentence]"
- Check for encoding issues
- Verify sentence format
- Look for special characters

### Low confidence in test queries
- Normal for fresh bootstrap with no prior knowledge
- System learns better with conversational interaction
- Add more domain-specific training data for your use case

## Future Enhancements

Potential improvements to the bootstrap system:

1. **Domain-Specific Datasets**: Create specialized grammar sets (technical, casual, formal)
2. **Multilingual Bootstrap**: Add datasets for other languages
3. **Progressive Loading**: Bootstrap in stages (basics first, then advanced)
4. **Validation Metrics**: Measure grammatical accuracy improvements
5. **Interactive Bootstrap**: Allow user feedback during training

## License

This grammatical training dataset is part of the Lilith AI project and follows the same license.

## Contributing

To contribute additional grammatical patterns:

1. Follow the existing category structure
2. Ensure sentences are grammatically correct
3. Cover edge cases and variations
4. Add comments explaining unusual patterns
5. Test with the bootstrap script before submitting
