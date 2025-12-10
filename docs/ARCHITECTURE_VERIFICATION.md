# Architecture Verification: BNN + Database Per Layer

## Overview

This document describes a segmented architecture in which each cognitive layer pairs a Bayesian/neural encoder (BNN) with a supporting database. The BNN component performs pattern recognition and produces representations; the database stores facts, patterns, and rules that are retrieved using the representations produced by the BNN. Each layer emits a working representation that serves as input for the subsequent layer.

## Layer 1: Intake (Character/Token Level)

Working representation: raw text -> normalized tokens

BNN component:
- Encoder that learns character-level and tokenization patterns
- Plasticity to adapt normalization rules

Database component:
- Table: intake_patterns
- Stores typo corrections and normalization rules (for example: "teh" -> "the")

Status:
- BNN present; intake database exists but is not currently utilized for lookup in the pipeline.

Operation:
1. BNN indicates pattern similarity for candidate corrections.
2. Database performs a similarity-based lookup to retrieve corrections.
3. Correction results are applied to produce normalized tokens for the semantic layer.

## Layer 2: Semantic (Concept Level)

Working representation: tokens -> concept embeddings and topic activations

BNN component:
- Encoder that maps tokens to concept embeddings (e.g., PMFlow or equivalent)
- Plasticity to refine concept representations

Database component:
- Table: semantic_taxonomy
- Stores concept definitions and inter-concept relationships (IS-A, PART-OF, attributes)

Status:
- BNN embedding generation is operating; semantic taxonomy is currently static and rarely queried during runtime.

Operation:
1. Words are embedded into a concept space by the BNN.
2. Embeddings are used to query the semantic taxonomy for related concepts and relationships.
3. The combined semantic representation is emitted to downstream layers.

## Layer 3: Syntax (Grammar Level)

Working representation: tokens -> POS sequences -> grammatical structures

BNN component:
- POS tagger and pattern encoder
- Plasticity to learn new grammatical structures

Database component:
- Table: syntax_patterns
- Stores grammatical templates and phrase structure patterns (for example: "DT JJ NN" -> NounPhrase)

Status:
- BNN produces POS and pattern encodings; syntax patterns table exists but is underutilized.

Operation:
1. BNN produces POS and pattern embeddings from tokens.
2. Pattern embeddings are used to query syntax_patterns for valid phrase or parse templates.
3. The parse structure is produced and passed to the pragmatic/response layer.

## Layer 4: Pragmatic/Response (Dialogue Level)

Working representation: context -> intent embeddings -> response selection

BNN component:
- Semantic/context encoder for intent recognition
- Learner that updates response-selection parameters based on feedback

Database component:
- Table: conversation_patterns (patterns, trigger contexts, success metrics)

Status:
- Both BNN and patterns database are in use; current pattern retrieval primarily relies on keyword matching rather than embedding-based similarity.

Operation:
1. Context is encoded into an intent embedding by the BNN.
2. The patterns database is queried using a hybrid strategy combining keywords and embedding similarity to select candidate responses.
3. The selected response is returned and its success metrics are updated based on feedback.

## Observed Integration Issues

The architecture is conceptually aligned: each layer should pair a BNN with a database and use BNN-generated representations to drive database queries. In practice, several layers generate embeddings without using those representations to perform similarity-based database lookups. This results in unused representational capacity and a keyword-dominant retrieval strategy in the pragmatic layer.

## Recommended Implementation Changes

1. Ensure BNN embeddings are used as primary or hybrid keys for database queries across all layers. Implement similarity search (for example, approximate nearest neighbors) on stored database vectors.

2. Convert static resources into queryable tables that can be incrementally updated. For example, allow the semantic_taxonomy and intake_patterns tables to be extended by learning procedures when new facts or corrections are validated.

3. Standardize the working representations between layers to guarantee clean input/output contracts (for example: normalized tokens -> concept embeddings -> parse trees -> response context).

4. Adopt hybrid retrieval strategies in the pragmatic layer that weight keyword matches and embedding similarity according to empirical performance (configurable weights).

5. Instrument data paths so that the provenance of retrieved facts is recorded, enabling interpretability and straightforward manual correction.

## Example Implementation Sketches

The following sketches illustrate the intended connection pattern between BNN outputs and database queries (pseudocode):

Intake layer:

```python
class IntakeLearner(GeneralPurposeLearner):
    def process(self, text):
        embeddings = self.bnn_encode(text)
        corrections = self.db.query_similar(embeddings, threshold=0.8)
        normalized = apply_corrections(text, corrections)
        return normalized
```

Semantic layer:

```python
class SemanticLearner(GeneralPurposeLearner):
    def process(self, tokens):
        word_embeddings = self.bnn_encode(tokens)
        concepts = []
        for word, embedding in zip(tokens, word_embeddings):
            related = self.db.query_concepts(word, embedding)
            concepts.extend(related)
        return concepts, word_embeddings
```

Syntax layer:

```python
class SyntaxLearner(GeneralPurposeLearner):
    def process(self, tokens):
        pos_sequence = self.bnn_tag_pos(tokens)
        pattern_embedding = self.bnn_encode_pattern(pos_sequence)
        structures = self.db.query_syntax_patterns(pattern_embedding)
        return parse_tree
```

Pragmatic layer:

```python
class PragmaticLearner(GeneralPurposeLearner):
    def compose_response(self, context):
        intent_embedding = self.bnn_encode(context)
        patterns = self.db.query_patterns(
            keywords=extract_keywords(context),
            semantic_embedding=intent_embedding,
            hybrid_weight=0.7
        )
        return selected_pattern.response_text
```

## Conclusion

The layered BNN + database architecture is sound; the principal work required is the integration of BNN-produced representations into database retrieval operations and the conversion of static resources into updatable, queryable knowledge stores. Implementing the recommended changes will improve utilization of learned representations, enhance interpretability, and allow knowledge to be updated without retraining the BNN components.
