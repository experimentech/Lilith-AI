
import unittest
import shutil
import os
from pathlib import Path
from v2.lilith_v2.multi_tenant_store import MultiTenantGraphManager

class TestMultiTenant(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("./tmp_mt_test")
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.base = self.test_root / "base"
        self.users = self.test_root / "users"
        self.manager = MultiTenantGraphManager(str(self.base), str(self.users))

    def tearDown(self):
        self.manager.close()
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_overlay(self):
        # Teacher writes to Base
        self.manager.add_node("concept_a", "concept", "Concept A", tenant_id="teacher")
        
        # User 1 sees it
        node = self.manager.get_node("concept_a", tenant_id="user1")
        self.assertIsNotNone(node)
        self.assertEqual(node["term"], "Concept A")
        
        # User 1 Overrides it (Mutable User Store)
        self.manager.add_node("concept_a", "concept", "Concept A (My Version)", tenant_id="user1")
        
        # User 1 sees override
        node = self.manager.get_node("concept_a", tenant_id="user1")
        self.assertEqual(node["term"], "Concept A (My Version)")
        
        # User 2 sees Original (Base)
        node = self.manager.get_node("concept_a", tenant_id="user2")
        self.assertEqual(node["term"], "Concept A")
        
        # Teacher sees Original
        node = self.manager.get_node("concept_a", tenant_id="teacher")
        self.assertEqual(node["term"], "Concept A")

if __name__ == "__main__":
    unittest.main()
