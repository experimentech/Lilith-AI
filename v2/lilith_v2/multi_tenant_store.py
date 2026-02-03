from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Union
import logging
import sqlite3
import weakref

from .pmflow_state import PMFlowStateStore
from .pmflow_sqlite import SQLitePMFlowStateStore
from .relational_graph_store import RelationalGraphStore

logger = logging.getLogger(__name__)


class MultiTenantPMFlowManager(PMFlowStateStore):
    """
    Manages tenant-specific PMFlow stores overlaid on base stores.
    
    Layer order (highest to lowest priority):
    1. Tenant-specific (data/users/{tenant}/)
    2. Production (data/production/) - large corpus, gitignored
    3. Base (data/base/) - seed data, tracked in git
    """
    def __init__(
        self, 
        base_root: str, 
        tenant_root: str,
        production_root: Optional[str] = None,
    ):
        self.base_root = Path(base_root)
        self.tenant_root = Path(tenant_root)
        self.base_root.mkdir(parents=True, exist_ok=True)
        self.tenant_root.mkdir(parents=True, exist_ok=True)
        
        # Base store is always active
        self._base_store = SQLitePMFlowStateStore(str(self.base_root / "pmflow.sqlite"))
        
        # Production store (optional, gitignored, for large corpus)
        self._production_store: Optional[SQLitePMFlowStateStore] = None
        if production_root:
            self.production_root = Path(production_root)
            if self.production_root.exists():
                prod_db = self.production_root / "pmflow.sqlite"
                if prod_db.exists():
                    self._production_store = SQLitePMFlowStateStore(str(prod_db))
                    logger.info(f"Production PMFlow store loaded: {prod_db}")
        
        # Cache for tenant stores (weak refs ideally, or LRU)
        self._stores: Dict[str, SQLitePMFlowStateStore] = {}

    def _get_store(self, tenant_id: Optional[str]) -> Tuple[SQLitePMFlowStateStore, bool]:
        """Returns (store, is_read_only_base)"""
        if not tenant_id or tenant_id == "teacher" or tenant_id == "base":
            return self._base_store, False # Write allowed to base if explicitly 'teacher'
        
        if tenant_id in self._stores:
            return self._stores[tenant_id], False
            
        # Create/Load tenant store
        t_path = self.tenant_root / tenant_id
        t_path.mkdir(parents=True, exist_ok=True)
        store = SQLitePMFlowStateStore(str(t_path / "pmflow.sqlite"))
        self._stores[tenant_id] = store
        return store, False

    def load_state(self, branch_id: str, tenant_id: str = None) -> Dict[str, Any]:
        """
        Load state. Priority: tenant → production → base.
        """
        # 1. Try Tenant
        if tenant_id and tenant_id != "teacher":
            store, _ = self._get_store(tenant_id)
            state = store.load_state(branch_id)
            if state: 
                return state
        
        # 2. Try Production (if available)
        if self._production_store:
            state = self._production_store.load_state(branch_id)
            if state:
                return state
        
        # 3. Fallback to Base
        return self._base_store.load_state(branch_id)

    def save_state(self, branch_id: str, state: Dict[str, Any], version: int, tenant_id: str = None) -> None:
        store, _ = self._get_store(tenant_id)
        store.save_state(branch_id, state, version)

    def bump_version(self, branch_id: str, tenant_id: str = None) -> int:
        store, _ = self._get_store(tenant_id)
        return store.bump_version(branch_id)

    def latent_dims(self, branch_id: str, tenant_id: str = None) -> Tuple[int, ...]:
        # Check tenant → production → base
        store, _ = self._get_store(tenant_id)
        dims = store.latent_dims(branch_id)
        if dims:
            return dims
        
        if self._production_store:
            dims = self._production_store.latent_dims(branch_id)
            if dims:
                return dims
                
        return self._base_store.latent_dims(branch_id)

    def compact(self, tenant_id: str = None) -> None:
        if tenant_id:
            s, _ = self._get_store(tenant_id)
            s.compact()
        else:
            self._base_store.compact()

    def close(self) -> None:
        self._base_store.close()
        if self._production_store:
            self._production_store.close()
        for s in self._stores.values():
            s.close()


class MultiTenantGraphManager(RelationalGraphStore):
    """
    Manages tenant-specific Knowledge Graphs overlaid on base stores.
    
    Layer order (highest to lowest priority):
    1. Tenant-specific (data/users/{tenant}/)
    2. Production (data/production/) - large corpus, gitignored
    3. Base (data/base/) - seed data, tracked in git
    """
    def __init__(
        self, 
        base_root: str, 
        tenant_root: str,
        production_root: Optional[str] = None,
    ):
        self.base_root = Path(base_root)
        self.tenant_root = Path(tenant_root)
        self.base_root.mkdir(parents=True, exist_ok=True)
        self.tenant_root.mkdir(parents=True, exist_ok=True)
        
        # We don't call super().__init__ because we don't hold a single connection
        self._base_store = RelationalGraphStore(str(self.base_root / "knowledge.sqlite"))
        
        # Production store (optional, gitignored, for large corpus)
        self._production_store: Optional[RelationalGraphStore] = None
        if production_root:
            self.production_root = Path(production_root)
            if self.production_root.exists():
                prod_db = self.production_root / "knowledge.sqlite"
                if prod_db.exists():
                    self._production_store = RelationalGraphStore(str(prod_db))
                    logger.info(f"Production graph store loaded: {prod_db}")
        
        self._stores: Dict[str, RelationalGraphStore] = {}
    
    def _init_schema(self):
        pass # Handle by sub-stores

    def _get_store(self, tenant_id: Optional[str]) -> RelationalGraphStore:
        if not tenant_id or tenant_id == "teacher" or tenant_id == "base":
            return self._base_store
        
        if tenant_id in self._stores:
            return self._stores[tenant_id]
            
        t_path = self.tenant_root / tenant_id
        t_path.mkdir(parents=True, exist_ok=True)
        store = RelationalGraphStore(str(t_path / "knowledge.sqlite"))
        self._stores[tenant_id] = store
        return store

    # --- Write Operations (Tenant Specific) ---

    def add_node(self, node_id: str, node_type: str, term: str, confidence: float = 1.0, data: Dict[str, Any] = None, tenant_id: str = None) -> None:
        store = self._get_store(tenant_id)
        store.add_node(node_id, node_type, term, confidence, data)

    def add_edge(self, source: str, target: str, edge_type: str, confidence: float = 1.0, tenant_id: str = None) -> None:
        store = self._get_store(tenant_id)
        store.add_edge(source, target, edge_type, confidence)

    # --- Read Operations (Overlay: User > Base) ---

    def get_node(self, node_id: str, tenant_id: str = None) -> Optional[Dict[str, Any]]:
        # Priority: tenant → production → base
        if tenant_id and tenant_id != "teacher":
            store = self._get_store(tenant_id)
            node = store.get_node(node_id)
            if node:
                return node
        
        # Try production
        if self._production_store:
            node = self._production_store.get_node(node_id)
            if node:
                return node
        
        # Fallback to base
        return self._base_store.get_node(node_id)

    def get_related(self, source_id: str, edge_type: Optional[str] = None, tenant_id: str = None) -> List[Dict[str, Any]]:
        """
        Merge neighbors from Base, Production, and Tenant.
        If same target exists in multiple layers, higher priority wins.
        """
        results = {} # target_id -> relation_dict

        # Helper to merge
        def merge_rows(rows):
            for r in rows:
                tid = r["target"]["id"]
                results[tid] = r

        # 1. Base (lowest priority)
        base_rows = self._base_store.get_related(source_id, edge_type)
        merge_rows(base_rows)

        # 2. Production (medium priority)
        if self._production_store:
            prod_rows = self._production_store.get_related(source_id, edge_type)
            merge_rows(prod_rows)

        # 3. Tenant (highest priority)
        if tenant_id and tenant_id != "teacher":
            store = self._get_store(tenant_id)
            tenant_rows = store.get_related(source_id, edge_type)
            merge_rows(tenant_rows)
            
        return list(results.values())

    def traverse_bfs(self, start_id: str, max_depth: int = 3, edge_types: Optional[List[str]] = None, tenant_id: str = None) -> List[Dict[str, Any]]:
        """
        BFS across the overlaid graph.
        We cannot use the optimized SQL traversal of one store. We must implement BFS logic here and use get_related().
        """
        # (current_id, path_list, min_confidence)
        queue = [(start_id, [start_id], 1.0)]
        visited = {start_id}
        valid_paths = []
        
        while queue:
            curr, path, conf = queue.pop(0)
            
            if len(path) > 1:
                valid_paths.append({"path_ids": path, "confidence": conf})
            
            if len(path) >= max_depth:
                continue

            # Get neighbors from combined view
            neighbors = self.get_related(curr, edge_type=None, tenant_id=tenant_id)
            
            for nb in neighbors:
                tgt_node = nb["target"]
                tgt_id = tgt_node["id"]
                etype = nb["relation"]
                econf = nb["confidence"]
                
                if edge_types and etype not in edge_types:
                    continue
                
                if tgt_id not in visited:
                    visited.add(tgt_id)
                    new_conf = min(conf, econf)
                    queue.append((tgt_id, path + [tgt_id], new_conf))
                    
        return sorted(valid_paths, key=lambda x: x['confidence'], reverse=True)

    def find_nodes_by_term(self, term_fragment: str, tenant_id: str = None) -> List[Dict[str, Any]]:
        # Search all layers and merge by id
        res_map = {}
        
        def merge(rows):
            for r in rows:
                res_map[r['id']] = r

        # Base (lowest priority)
        merge(self._base_store.find_nodes_by_term(term_fragment))
        
        # Production
        if self._production_store:
            merge(self._production_store.find_nodes_by_term(term_fragment))

        # Tenant (highest priority)
        if tenant_id and tenant_id != "teacher":
            store = self._get_store(tenant_id)
            merge(store.find_nodes_by_term(term_fragment))
            
        return list(res_map.values())
        
    def get_all_terms(self, tenant_id: str = None) -> List[Tuple[str, str]]:
        """Return (id, term) for all concepts, merging all layers."""
        
        terms_map = {}
        
        # 1. Base Terms (lowest priority)
        for nid, term in self._base_store.get_all_terms():
            terms_map[nid] = term
        
        # 2. Production Terms
        if self._production_store:
            for nid, term in self._production_store.get_all_terms():
                terms_map[nid] = term
            
        # 3. Tenant Terms (highest priority)
        if tenant_id and tenant_id != "teacher":
            store = self._get_store(tenant_id)
            for nid, term in store.get_all_terms():
                 terms_map[nid] = term

        return list(terms_map.items())

    def close(self) -> None:
        self._base_store.close()
        if self._production_store:
            self._production_store.close()
        for s in self._stores.values():
            s.close()
