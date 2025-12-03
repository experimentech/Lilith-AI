#!/usr/bin/env python3
"""Quick test to verify opinion templates are loaded."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lilith.pragmatic_templates import PragmaticTemplateStore

store = PragmaticTemplateStore()

print("Total templates:", len(store.templates))
print("\nTemplates by category:")
categories = {}
for template in store.templates.values():
    if template.category not in categories:
        categories[template.category] = []
    categories[template.category].append(template.template_id)

for cat, temps in sorted(categories.items()):
    print(f"  {cat}: {len(temps)} templates")
    for t in temps:
        print(f"    - {t}")

print("\nOpinion templates:")
opinion_templates = store.get_templates_by_category("opinion")
for t in opinion_templates:
    print(f"  {t.template_id}: slots={t.slots}, priority={t.priority}")
    print(f"    Template: {t.template}")
