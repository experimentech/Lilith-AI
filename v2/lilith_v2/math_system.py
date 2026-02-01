from typing import Optional, List, Tuple, Any, Dict
import re
from dataclasses import dataclass
import logging

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    
logger = logging.getLogger(__name__)

@dataclass
class MathResult:
    """Result from mathematical computation"""
    query: str
    expression: str
    result: str
    steps: List[str]
    latex: Optional[str] = None
    confidence: float = 1.0

class MathSystem:
    """
    Symbolic computation backend for mathematical queries.
    Ports v1 logic to v2.
    """
    
    def __init__(self):
        if not SYMPY_AVAILABLE:
            logger.warning("SymPy not available. MathSystem will be disabled.")
            return
            
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application,)
        )
        
    def check_confidence(self, query: str) -> float:
        """Check if this backend can handle the query."""
        if not SYMPY_AVAILABLE: return 0.0
        
        query_lower = query.lower()
        
        # Strong indicators
        strong_patterns = [
            r'\d+\s*[+\-*/^×÷]\s*\d+',
            r'(sin|cos|tan|log|sqrt|exp|abs)\s*\(',
            r'[=]\s*0',
        ]
        if any(re.search(p, query) for p in strong_patterns):
            return 0.95
            
        # Keywords
        math_keywords = {'calculate', 'compute', 'solve', 'evaluate', 'derive', 'integrate'}
        if any(w in query_lower for w in math_keywords) and re.search(r'\d+', query):
            return 0.85
            
        return 0.0

    def compute(self, query: str) -> Optional[MathResult]:
        """Compute mathematical result."""
        if not SYMPY_AVAILABLE: return None
        
        try:
            intent, expression_str = self._parse_query(query)
            if not expression_str: return None
            
            if intent == 'solve' and '=' in expression_str:
                result, steps = self._solve_equation_str(expression_str)
            else:
                expr = self._parse_expression(expression_str)
                if expr is None: return None
                
                if intent == 'derivative': result, steps = self._compute_derivative(expr)
                elif intent == 'integral': result, steps = self._compute_integral(expr)
                else: result, steps = self._evaluate_expression(expr)
                
            return MathResult(
                query=query,
                expression=expression_str,
                result=str(result),
                steps=steps,
                latex=sp.latex(result) if hasattr(sp, 'latex') else None
            )
        except Exception as e:
            logger.error(f"Math computation failed: {e}")
            return None

    def _parse_query(self, query: str) -> Tuple[str, str]:
        query_lower = query.lower()
        if 'solve' in query_lower: intent = 'solve'
        elif 'deriv' in query_lower: intent = 'derivative'
        elif 'integ' in query_lower: intent = 'integral'
        else: intent = 'evaluate'
        
        return intent, self._extract_expression(query)

    def _extract_expression(self, query: str) -> str:
        # Simple extraction logic
        # Remove common "noise" words
        ignore = ['what is', 'calculate', 'solve', '?', 'the', 'of', 'please']
        out = query.lower()
        for w in ignore: out = out.replace(w, '')
        
        # Normalize operators
        replacements = {
            'plus': '+', 'minus': '-', 'times': '*', 'divided by': '/',
            'squared': '**2', 'cubed': '**3', '^': '**'
        }
        for k, v in replacements.items(): out = out.replace(k, v)
        
        return out.strip()

    def _parse_expression(self, expr_str: str):
        try:
            return parse_expr(expr_str, transformations=self.transformations)
        except:
            return None

    def _evaluate_expression(self, expr):
        if expr.is_number:
            return sp.N(expr), ["Analyzed numerical expression"]
        return sp.simplify(expr), ["Simplified expression"]
        
    def _solve_equation_str(self, eq_str):
        parts = eq_str.split('=')
        lhs = self._parse_expression(parts[0])
        rhs = self._parse_expression(parts[1])
        eq = sp.Eq(lhs, rhs)
        res = sp.solve(eq)
        return res, [f"Solved {lhs} = {rhs}"]
        
    def _compute_derivative(self, expr):
        vars_ = list(expr.free_symbols)
        if not vars_: return 0, ["Constant"]
        res = sp.diff(expr, vars_[0])
        return res, [f"Differentiated w.r.t {vars_[0]}"]

    def _compute_integral(self, expr):
        vars_ = list(expr.free_symbols)
        if not vars_: return expr, ["Constant integration"] # Simplification
        res = sp.integrate(expr, vars_[0])
        return res, [f"Integrated w.r.t {vars_[0]}"]
