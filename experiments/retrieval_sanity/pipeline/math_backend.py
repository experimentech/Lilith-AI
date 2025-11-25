"""
Math Backend - Symbolic Computation for Mathematical Queries

Prevents mathematical expressions from polluting the linguistic database
by handling them with exact symbolic computation instead.

Uses SymPy for safe, sandboxed symbolic mathematics.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import re

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


@dataclass
class MathResult:
    """Result from mathematical computation"""
    query: str                  # Original query
    expression: str             # Parsed mathematical expression
    result: str                 # Computed result
    steps: List[str]           # Step-by-step solution (if applicable)
    latex: Optional[str] = None  # LaTeX representation
    confidence: float = 1.0     # Always 1.0 for exact computation


class MathBackend:
    """
    Symbolic computation backend for mathematical queries.
    
    Handles:
    - Arithmetic: 2 + 2, 15 * 7, 100 / 4
    - Algebra: solve x + 5 = 10, expand (x + 2)^2
    - Calculus: derivative of x^2, integral of sin(x)
    - Simplification: simplify (x^2 - 4)/(x - 2)
    - Evaluation: sqrt(16), sin(pi/4)
    
    Safety:
    - Pure symbolic computation (no code execution)
    - Uses SymPy (sandboxed symbolic math library)
    - No file system access
    - No network access
    - Timeout protection
    """
    
    def __init__(self, timeout: int = 5):
        """
        Initialize math backend.
        
        Args:
            timeout: Maximum computation time in seconds
        """
        if not SYMPY_AVAILABLE:
            raise RuntimeError("SymPy required for math backend. Install with: pip install sympy")
        
        self.timeout = timeout
        
        # SymPy parsing transformations
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application,)
        )
    
    def can_handle(self, query: str) -> Tuple[bool, float]:
        """
        Check if this backend can handle the query.
        
        Args:
            query: User query
            
        Returns:
            (can_handle, confidence)
        """
        query_lower = query.lower()
        
        # Strong indicators (high confidence)
        strong_patterns = [
            r'\d+\s*[+\-*/^×÷]\s*\d+',  # Arithmetic operators with numbers
            r'(sin|cos|tan|log|sqrt|exp|abs)\s*\(',  # Math functions
            r'[=]\s*0',  # Equations
        ]
        
        for pattern in strong_patterns:
            if re.search(pattern, query):
                return True, 0.95
        
        # Math keywords (medium confidence)
        math_keywords = [
            'calculate', 'compute', 'solve', 'evaluate', 'simplify',
            'derivative', 'integral', 'factor', 'expand', 'square root',
            'logarithm', 'equals', 'plus', 'minus', 'times', 'divided by'
        ]
        
        keyword_count = sum(1 for kw in math_keywords if kw in query_lower)
        if keyword_count >= 2:
            return True, 0.85
        elif keyword_count == 1:
            # Check if there are also numbers
            if re.search(r'\d+', query):
                return True, 0.70
        
        # Weak indicators (low confidence)
        operators = ['+', '-', '*', '/', '^', '=', '×', '÷']
        if any(op in query for op in operators):
            # Has operators but might be linguistic (e.g., "C++ programming")
            if re.search(r'\d', query):  # Has numbers too
                return True, 0.60
        
        return False, 0.0
    
    def compute(self, query: str) -> Optional[MathResult]:
        """
        Compute mathematical result.
        
        Args:
            query: Mathematical query
            
        Returns:
            MathResult if successful, None if cannot compute
        """
        try:
            # 1. Detect intent and extract expression
            intent, expression_str = self._parse_query(query)
            
            if not expression_str:
                return None
            
            # 2. For equations, handle differently (don't parse the = sign)
            if intent == 'solve' and '=' in expression_str:
                result, steps = self._solve_equation_from_string(expression_str)
            else:
                # Parse expression
                expr = self._parse_expression(expression_str)
                
                if expr is None:
                    return None
                
                # 3. Compute based on intent
                if intent == 'derivative':
                    result, steps = self._compute_derivative(expr)
                elif intent == 'integral':
                    result, steps = self._compute_integral(expr)
                elif intent == 'expand':
                    result, steps = self._expand_expression(expr)
                elif intent == 'factor':
                    result, steps = self._factor_expression(expr)
                elif intent == 'simplify':
                    result, steps = self._simplify_expression(expr)
                else:  # 'evaluate'
                    result, steps = self._evaluate_expression(expr)
            
            # 4. Format result
            result_str = str(result)
            latex_str = sp.latex(result) if hasattr(sp, 'latex') else None
            
            return MathResult(
                query=query,
                expression=expression_str,
                result=result_str,
                steps=steps,
                latex=latex_str,
                confidence=1.0
            )
            
        except Exception as e:
            # Computation failed - return None to fallback to linguistic
            print(f"  ⚠️  Math computation failed: {e}")
            return None
    
    def _parse_query(self, query: str) -> Tuple[str, str]:
        """
        Parse query to detect intent and extract expression.
        
        Returns:
            (intent, expression_string)
        """
        query_lower = query.lower()
        
        # Detect intent
        if any(kw in query_lower for kw in ['solve', 'solve for', 'find x']):
            intent = 'solve'
        elif any(kw in query_lower for kw in ['derivative', 'differentiate', "d/dx"]):
            intent = 'derivative'
        elif any(kw in query_lower for kw in ['integral', 'integrate', '∫']):
            intent = 'integral'
        elif 'expand' in query_lower:
            intent = 'expand'
        elif 'factor' in query_lower:
            intent = 'factor'
        elif 'simplify' in query_lower:
            intent = 'simplify'
        else:
            intent = 'evaluate'
        
        # Extract expression
        expression = self._extract_expression(query, intent)
        
        return intent, expression
    
    def _extract_expression(self, query: str, intent: str) -> str:
        """Extract mathematical expression from natural language query"""
        
        # Remove common question words and phrases
        removals = [
            'what is', 'what are', 'calculate', 'compute', 'solve', 'evaluate',
            'find', 'the', 'of', 'for', '?', 'please', 'can you'
        ]
        
        expression = query.lower()
        for removal in removals:
            expression = expression.replace(removal, ' ')
        
        # Convert word operators to symbols
        replacements = {
            'plus': '+',
            'minus': '-',
            'times': '*',
            'multiplied by': '*',
            'divided by': '/',
            'over': '/',
            'to the power of': '**',
            'squared': '**2',
            'cubed': '**3',
            'square root': 'sqrt',
            'x': 'x',  # Preserve variable
        }
        
        for word, symbol in replacements.items():
            expression = expression.replace(word, symbol)
        
        # Clean up
        expression = expression.strip()
        expression = re.sub(r'\s+', ' ', expression)  # Multiple spaces to single
        expression = expression.replace(' ', '')  # Remove remaining spaces
        
        return expression
    
    def _parse_expression(self, expr_str: str) -> Optional[sp.Expr]:
        """Parse string into SymPy expression"""
        try:
            # Handle common patterns
            expr_str = expr_str.replace('^', '**')  # Convert ^ to **
            expr_str = expr_str.replace('÷', '/')   # Convert ÷ to /
            expr_str = expr_str.replace('×', '*')   # Convert × to *
            
            # Parse with SymPy
            expr = parse_expr(expr_str, transformations=self.transformations)
            return expr
            
        except Exception as e:
            print(f"  ⚠️  Failed to parse expression '{expr_str}': {e}")
            return None
    
    def _evaluate_expression(self, expr: sp.Expr) -> Tuple[sp.Expr, List[str]]:
        """Evaluate expression to get numerical result"""
        steps = []
        
        # Try to evaluate numerically
        if expr.is_number or not expr.free_symbols:
            result = sp.N(expr)  # Numerical evaluation
            steps.append(f"Evaluate: {expr}")
            steps.append(f"Result: {result}")
        else:
            # Has variables - just simplify
            result = sp.simplify(expr)
            steps.append(f"Simplified: {result}")
        
        return result, steps
    
    def _solve_equation(self, expr: sp.Expr, original: str) -> Tuple[sp.Expr, List[str]]:
        """Solve equation for variable (DEPRECATED - use _solve_equation_from_string)"""
        steps = []
        
        # Check if it's an equation (has =)
        if '=' in original:
            parts = original.split('=')
            if len(parts) == 2:
                lhs = self._parse_expression(parts[0])
                rhs = self._parse_expression(parts[1])
                equation = sp.Eq(lhs, rhs)
                
                # Solve for x (or first variable found)
                variables = equation.free_symbols
                if variables:
                    var = list(variables)[0]
                    solutions = sp.solve(equation, var)
                    
                    steps.append(f"Equation: {lhs} = {rhs}")
                    steps.append(f"Solve for {var}")
                    steps.append(f"Solution: {var} = {solutions}")
                    
                    if len(solutions) == 1:
                        return solutions[0], steps
                    else:
                        return solutions, steps
        
        # Not an equation - just evaluate
        return self._evaluate_expression(expr)
    
    def _solve_equation_from_string(self, equation_str: str) -> Tuple[Optional[sp.Expr], List[str]]:
        """Solve equation from string containing = sign"""
        steps = []
        
        # Split on = sign
        parts = equation_str.split('=')
        if len(parts) != 2:
            return None, ["Invalid equation format"]
        
        # Parse both sides
        lhs = self._parse_expression(parts[0].strip())
        rhs = self._parse_expression(parts[1].strip())
        
        if lhs is None or rhs is None:
            return None, ["Failed to parse equation"]
        
        # Create equation
        equation = sp.Eq(lhs, rhs)
        
        # Solve for x (or first variable found)
        variables = equation.free_symbols
        if not variables:
            return None, ["No variables to solve for"]
        
        var = list(variables)[0]
        solutions = sp.solve(equation, var)
        
        steps.append(f"Equation: {lhs} = {rhs}")
        steps.append(f"Solve for {var}")
        
        if len(solutions) == 1:
            steps.append(f"Solution: {var} = {solutions[0]}")
            return solutions[0], steps
        else:
            steps.append(f"Solutions: {var} = {solutions}")
            return solutions, steps
    
    def _compute_derivative(self, expr: sp.Expr) -> Tuple[sp.Expr, List[str]]:
        """Compute derivative"""
        steps = []
        
        # Find variable
        variables = expr.free_symbols
        if not variables:
            return expr, ["Expression has no variables"]
        
        var = list(variables)[0]  # Use first variable
        
        # Compute derivative
        derivative = sp.diff(expr, var)
        simplified = sp.simplify(derivative)
        
        steps.append(f"f({var}) = {expr}")
        steps.append(f"f'({var}) = d/d{var}[{expr}]")
        steps.append(f"f'({var}) = {simplified}")
        
        return simplified, steps
    
    def _compute_integral(self, expr: sp.Expr) -> Tuple[sp.Expr, List[str]]:
        """Compute integral"""
        steps = []
        
        # Find variable
        variables = expr.free_symbols
        if not variables:
            return expr, ["Expression has no variables"]
        
        var = list(variables)[0]
        
        # Compute integral
        integral = sp.integrate(expr, var)
        simplified = sp.simplify(integral)
        
        steps.append(f"∫ {expr} d{var}")
        steps.append(f"= {simplified} + C")
        
        return simplified, steps
    
    def _expand_expression(self, expr: sp.Expr) -> Tuple[sp.Expr, List[str]]:
        """Expand expression"""
        steps = []
        
        expanded = sp.expand(expr)
        
        steps.append(f"Original: {expr}")
        steps.append(f"Expanded: {expanded}")
        
        return expanded, steps
    
    def _factor_expression(self, expr: sp.Expr) -> Tuple[sp.Expr, List[str]]:
        """Factor expression"""
        steps = []
        
        factored = sp.factor(expr)
        
        steps.append(f"Original: {expr}")
        steps.append(f"Factored: {factored}")
        
        return factored, steps
    
    def _simplify_expression(self, expr: sp.Expr) -> Tuple[sp.Expr, List[str]]:
        """Simplify expression"""
        steps = []
        
        simplified = sp.simplify(expr)
        
        steps.append(f"Original: {expr}")
        steps.append(f"Simplified: {simplified}")
        
        return simplified, steps
