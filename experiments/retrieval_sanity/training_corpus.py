"""Training corpus for contrastive learning.

Provides diverse examples across multiple concept categories for
training the semantic stage with contrastive plasticity.
"""

# Larger corpus with clear concept categories
TRAINING_CORPUS = [
    # Medical/Hospital cluster
    "alice visited the hospital",
    "bob went to the hospital",
    "the doctor works at the hospital",
    "the patient received treatment",
    "nurses care for patients",
    "the emergency room was busy",
    "medical staff wear scrubs",
    "the surgeon performed an operation",
    "hospital rooms have beds",
    "doctors prescribe medicine",
    "the clinic provides healthcare",
    "ambulances transport patients",
    "hospitals have operating rooms",
    "medical equipment saves lives",
    "the physician examined the patient",
    
    # Park/Outdoor cluster
    "alice met bob at the park",
    "the park has many trees",
    "children play in the park",
    "the outdoor garden is beautiful",
    "we walked through the park",
    "flowers bloom in the garden",
    "people jog in the park",
    "the park has a playground",
    "families picnic outdoors",
    "the garden has colorful flowers",
    "trees provide shade in the park",
    "the park fountain is beautiful",
    "birds sing in the trees",
    "the meadow is green and peaceful",
    "outdoor activities are healthy",
    
    # Library/Learning cluster
    "the library has books",
    "students study in the library",
    "libraries have quiet reading rooms",
    "librarians help find books",
    "the library has many shelves",
    "people read books quietly",
    "the library offers free books",
    "students borrow books from the library",
    "the library has a catalog system",
    "reading rooms are peaceful",
    "libraries preserve knowledge",
    "the library has computers",
    "students research in the library",
    "libraries organize books by category",
    "the library collection is extensive",
    
    # Classroom/Education cluster
    "students study in the classroom",
    "the teacher explains the lesson",
    "classrooms have desks and chairs",
    "students take notes in class",
    "the professor lectures to students",
    "classrooms have whiteboards",
    "students raise their hands to answer",
    "teachers assign homework",
    "the classroom has educational posters",
    "students learn from teachers",
    "exams test student knowledge",
    "the classroom is full of students",
    "teachers grade assignments",
    "students collaborate on projects",
    "the school has many classrooms",
    
    # Restaurant/Food cluster
    "the restaurant serves delicious food",
    "waiters take customer orders",
    "chefs cook in the kitchen",
    "the menu has many options",
    "restaurants have tables and chairs",
    "customers enjoy their meals",
    "the chef prepares fresh ingredients",
    "restaurants serve breakfast lunch dinner",
    "waiters bring food to tables",
    "the kitchen is clean and organized",
    "customers pay for their meals",
    "restaurants have busy hours",
    "chefs create new recipes",
    "the restaurant has good reviews",
    "food is served hot and fresh",
    
    # Office/Work cluster
    "employees work in the office",
    "the office has many cubicles",
    "meetings are held in conference rooms",
    "workers use computers daily",
    "the office has a reception desk",
    "employees collaborate on projects",
    "managers lead their teams",
    "the office has modern equipment",
    "workers take lunch breaks",
    "offices have filing cabinets",
    "employees answer phones and emails",
    "the office building is tall",
    "workers commute to the office",
    "offices have printers and copiers",
    "employees attend training sessions",
]

# Concept labels for evaluation
CONCEPT_LABELS = {
    # Medical (0-14)
    **{i: "medical" for i in range(15)},
    # Park/Outdoor (15-29)
    **{i: "outdoor" for i in range(15, 30)},
    # Library (30-44)
    **{i: "library" for i in range(30, 45)},
    # Classroom (45-59)
    **{i: "classroom" for i in range(45, 60)},
    # Restaurant (60-74)
    **{i: "restaurant" for i in range(60, 75)},
    # Office (75-89)
    **{i: "office" for i in range(75, 90)},
}

# Test queries for each category
TEST_QUERIES = [
    ("hospital visit", "medical", [0, 1, 2, 3, 4, 5]),
    ("doctor appointment", "medical", [2, 3, 9, 14]),
    ("emergency treatment", "medical", [5, 7, 11]),
    
    ("outdoor recreation", "outdoor", [15, 16, 17, 18, 24]),
    ("nature walk", "outdoor", [15, 16, 19, 25, 28]),
    ("park activities", "outdoor", [15, 16, 17, 22, 24]),
    
    ("reading books", "library", [30, 31, 32, 33, 35]),
    ("research materials", "library", [30, 31, 42, 43]),
    ("quiet study", "library", [30, 31, 32, 39]),
    
    ("student learning", "classroom", [45, 46, 48, 49, 50]),
    ("teacher instruction", "classroom", [46, 47, 51, 52]),
    ("educational environment", "classroom", [45, 46, 48, 53]),
    
    ("dining experience", "restaurant", [60, 61, 65, 68]),
    ("food preparation", "restaurant", [62, 63, 66, 72]),
    ("customer service", "restaurant", [61, 65, 68, 70]),
    
    ("workplace environment", "office", [75, 76, 78, 82]),
    ("professional collaboration", "office", [76, 80, 81, 84]),
    ("business operations", "office", [75, 77, 79, 83]),
]


def get_corpus():
    """Get the full training corpus."""
    return TRAINING_CORPUS


def get_concept_labels():
    """Get concept labels for each document."""
    return CONCEPT_LABELS


def get_test_queries():
    """Get test queries with expected relevant documents."""
    return TEST_QUERIES


def corpus_statistics():
    """Print corpus statistics."""
    print("=" * 80)
    print("TRAINING CORPUS STATISTICS")
    print("=" * 80)
    print(f"Total documents: {len(TRAINING_CORPUS)}")
    print(f"\nDocuments per category:")
    
    categories = {}
    for idx, label in CONCEPT_LABELS.items():
        categories[label] = categories.get(label, 0) + 1
    
    for category, count in sorted(categories.items()):
        print(f"  {category:12} {count:3} docs")
    
    print(f"\nTest queries: {len(TEST_QUERIES)}")
    queries_per_cat = {}
    for _, cat, _ in TEST_QUERIES:
        queries_per_cat[cat] = queries_per_cat.get(cat, 0) + 1
    
    for category, count in sorted(queries_per_cat.items()):
        print(f"  {category:12} {count:3} queries")
    
    print("=" * 80)


if __name__ == "__main__":
    corpus_statistics()
    
    print("\nSample documents:")
    for i in [0, 15, 30, 45, 60, 75]:
        print(f"  [{i:2d}] ({CONCEPT_LABELS[i]:12}) {TRAINING_CORPUS[i]}")
