import sympy as sp
import random
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, parse_expr
import ast


class BaseQuestion:
    def __init__(self):
        self.input_hint = "Use Python-style syntax: `**` for powers, `*` for multiplication, `sqrt(x)` for square root, and `log(x)` for natural log. . Provide answer in fractions and make the multiplication explicit if applicable."

    def format_answer(self, ans):
        """Consistently format the answer in LaTeX with an 'Answer: ' prefix."""
        return "Answer: " + sp.latex(ans)
    
    def check_sym_answer(self, String: str, correct_expr):
        """Check symbolic answers by parsing and simplifying the difference."""
        try:
            parsed_expr = parse_expr(String)
            return sp.simplify(parsed_expr - correct_expr) == 0
        except Exception:
            return False
    
    def check_numeric_answer(self, String: str, correct_val, tol=1e-3):
        """Check numerical answers using float conversion and a tolerance."""
        try:
            parsed_expr = parse_expr(String)
            return abs(float(parsed_expr) - float(correct_val)) < tol
        except Exception:
            return False

# Limits
class LimitQuestion(BaseQuestion):
    symbol = sp.symbols('x')
    
    def __init__(self):
        self.input_hint = "Enter the function `f(x)` value evaluated at the limit. Use `**` for powers and `/` for division. Provide answer in fractions and make the multiplication explicit if applicable."
        expr, x0, limit_expr, _ = self.random_limit_equation()
        self.equation = expr
        self.to = x0
        self.answer = limit_expr
        self.question_text = f"Compute the limit: $\\lim_{{x \\to {self.to}}} f(x) = {sp.latex(self.equation)}$."
        self.latex_answer = self.format_answer(self.answer)
    
    def random_limit_equation(self):
        x = self.symbol
        # Generate random coefficients, ensuring 'c' is nonzero
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.choice([i for i in range(-10, 11) if i != 0])
        d = random.randint(-10, 10)
        
        expr = (a * x + b) / (c * x**2 + d)
        
        # Try 10 times to get an x0 that does not make the denominator zero
        x0 = None
        for _ in range(10):
            candidate = random.randint(-10, 10)
            if sp.N(c * candidate**2 + d) != 0:
                x0 = candidate
                break
        if x0 is None:
            x0 = 0
        
        limit_expr = sp.limit(expr, x, x0)
        limit_equation = sp.Eq(limit_expr, limit_expr)
        return expr, x0, limit_expr, limit_equation

    def check_answer(self, latexString: str):
        try:
            parsed_expr = parse_expr(latexString)
            user_limit = sp.limit(parsed_expr, self.symbol, self.to)
            return sp.simplify(user_limit - self.answer) == 0
        except Exception:
            return False

# Linear Algebra
class LinearAlgebraQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Enter vectors as [1, 2] and matrices as [[1, 2], [3, 4]] where [[top row], [bottom row]].  Use integers or fractions only."
        self.matrix1, self.matrix2 = self.generate_matrices()
        self.operation = 'multiplication'
        self.answer = self.matrix1 * self.matrix2
        self.question_text = f"Given $A = {sp.latex(self.matrix1)}$ and $B = {sp.latex(self.matrix2)}$, compute $AB$."
        self.latex_answer = self.format_answer(self.answer)
    
    # Generating the matrices
    def generate_matrices(self):
        def random_matrix():
            return sp.Matrix([[random.randint(-5, 5) for _ in range(2)] for _ in range(2)])
        return random_matrix(), random_matrix()
    
    def check_answer(self, latexString: str):
        try:
            parsed_string = f"Matrix({latexString})"
            
            parsed_matrix = parse_expr(parsed_string)
            return sp.simplify(parsed_matrix - self.answer) == sp.zeros(*self.answer.shape)
        except Exception:
            return False

# Trigonometry
class TrigonometryQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Evaluate the trigonometric function numerically. Use values like `sqrt(3)/2`, `1/2`, etc. Provide answer in fractions and make the multiplication explicit if applicable."
        self.x = sp.symbols('x')
        self.function = random.choice([sp.sin(self.x), sp.cos(self.x)])
        self.point = random.choice([0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, sp.pi])
        self.question_text = f"Evaluate ${sp.latex(self.function)}$ at $x = {sp.latex(self.point)}$."
        self.answer = self.function.subs(self.x, self.point).evalf()
        self.latex_answer = self.format_answer(self.answer)
    
    def check_answer(self, latexString: str):
        
        no_whitespace = latexString.replace(" ", "")
        str_func = str(self.function)
        str_point = str(self.point)
        to_check = str_func.replace("x", str_point)
        # Check if the user didn't just input the equation itself (like cos(pi/6))
        if no_whitespace == to_check:
            return False
        return self.check_numeric_answer(latexString, self.answer)



# Indefinite Integral 
class IndefiniteIntegralQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Omit the constant of integration `+ C` unless you are explicitly asked. Provide answer in fractions and make the multiplication explicit if applicable."
        x = sp.symbols('x')
        self.func = random.choice([x**2 + 2*x, sp.sin(x), sp.exp(x)])
        self.integral = sp.integrate(self.func, x)
        self.question_text = f"Compute the indefinite integral of ${sp.latex(self.func)}$ with respect to $x$."
        # Explicitly include the constant of integration
        self.latex_answer = self.format_answer(self.integral + sp.Symbol('C'))
        self.answer = self.integral
    
    def check_answer(self, latexString: str):
        try:
            parsed_expr = parse_expr(latexString)
            diff = sp.simplify(parsed_expr - self.answer)
            # Allow differences that are simply an arbitrary constant
            return diff.free_symbols <= {sp.Symbol('C')} or diff == 0
        except Exception:
            return False
    # Derivatives
class DerivativeQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Use sin(x), cos(x), ln(x), exp(x), and write powers like x**2. Provide answer in fractions and make the multiplication explicit if applicable."
        x = sp.symbols('x')
        self.x = x
        
        # Create a list of function types for first-year level
        function_types = [
            # Polynomials
            lambda: random.randint(-5, 5)*x**random.randint(1, 3) + random.randint(-5, 5)*x + random.randint(-5, 5),
            # Trigonometric functions
            lambda: random.choice([sp.sin(x), sp.cos(x)]),
            # Exponential/logarithmic
            lambda: sp.exp(random.randint(1, 2)*x),
            lambda: sp.log(x + random.randint(1, 3)),
            # Product rule examples
            lambda: x * sp.sin(x),
            # Quotient rule examples
            lambda: random.randint(1, 5) / (x + random.randint(1, 3)),
            # Chain rule examples
            lambda: sp.sin(x**2)
        ]
        
        # Choose a random function type
        self.func = random.choice(function_types)()
        
        # Compute the derivative
        self.derivative = sp.diff(self.func, x)
        
        # Generate question text
        self.question_text = f"Find the derivative of $f(x) = {sp.latex(self.func)}$ with respect to $x$."
        self.latex_answer = self.format_answer(self.derivative)
        self.answer = self.derivative
    
    def check_answer(self, latexString: str):
        return self.check_sym_answer(latexString, self.derivative)

# Definite Integrals
class DefiniteIntegralQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Simplify your final result. Write constants like pi and e as `pi` and `exp(x)`, and use fractions like 1/2. make the multiplication explicit if applicable. Eulers number can be written as exp(1)."
        x = sp.symbols('x')
        self.x = x
        
        # Create a list of integrable functions
        function_types = [
            # Simple polynomials
            lambda: random.randint(1, 3)*x**random.randint(1, 2) + random.randint(-3, 3),
            # Basic trigonometric functions
            lambda: random.choice([sp.sin(x), sp.cos(x)]),
            # Simple exponential
            lambda: sp.exp(x),
            # Simple rational functions
            lambda: 1/(x + random.randint(1, 2))
        ]
        
        # Choose a random function type
        self.func = random.choice(function_types)()
        
        # Choose simple integration limits
        limit_options = [(0, 1), (0, 2), (-1, 1), (0, sp.pi/2), (-sp.pi/2, sp.pi/2)]
        self.a, self.b = random.choice(limit_options)
        
        # Compute the definite integral
        self.integral = sp.integrate(self.func, (x, self.a, self.b))
        
        self.answer = self.integral

        # Generate question text
        self.question_text = f"Evaluate the definite integral: $\\int_{{{sp.latex(self.a)}}}^{{{sp.latex(self.b)}}} {sp.latex(self.func)}\\, dx$."
        self.latex_answer = self.format_answer(self.integral)
    
    def check_answer(self, latexString: str):
        # For definite integrals, we can use numerical comparison
        return self.check_numeric_answer(latexString, self.integral)
    
    
    # Systems of Linear Equations
class LinearSystemQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Enter equations like `x = 1, y = -2` with commas separating variables. Don’t skip any... Provide answer in fractions if applicable"
        # Define symbols
        x, y, z = sp.symbols('x y z')
        self.variables = [x, y, z]
        
        # Randomly choose between 2 or 3 equations
        self.num_equations = random.randint(2, 3)
        self.equations = []
        self.matrix_form = None
        self.answer = None
        
        # Generate a system with a unique answer
        self._generate_system()
        
        # Format the question
        self.question_text = self._format_question()
        self.latex_answer = self.format_answer(self.answer)
    
    def _generate_system(self):
        # Create a random matrix that will have a unique answer
        num_vars = self.num_equations
        
        # First create a random answer vector
        answer_values = [random.randint(-5, 5) for _ in range(num_vars)]
        
        # Create a matrix that's guaranteed to be non-singular by ensuring it has full rank
        while True:
            # Generate random coefficients
            A = sp.Matrix([[random.randint(-5, 5) for _ in range(num_vars)] for _ in range(num_vars)])
            
            # Check if matrix is non-singular (det ≠ 0)
            if A.det() != 0:
                break
        
        # Create the right-hand side vector b = A*x where x is our answer
        answer_vector = sp.Matrix(answer_values)
        b = A * answer_vector
        
        # Store the matrix form for reference
        self.matrix_form = (A, b)
        
        # Create the equations
        for i in range(num_vars):
            lhs = 0
            for j in range(num_vars):
                lhs += A[i, j] * self.variables[j]
            self.equations.append(sp.Eq(lhs, b[i]))
        
        # Create answer dictionary
        self.answer = {self.variables[i]: answer_values[i] for i in range(num_vars)}
    
    def _format_question(self):
        equation_strings = []
        for eq in self.equations:
            equation_strings.append(sp.latex(eq))
        
        latex_lines = r"\\ ".join(equation_strings)
        return f"Solve the following system of linear equations:\n\n$$\\begin{{cases}} {latex_lines} \\end{{cases}}$$"
        
    
    def check_answer(self, latexString: str):
        try:
            # Parse the answer which should be in the form "x = a, y = b, z = c"
            # or similar format
            pairs = latexString.split(',')
            user_answer = {}
            
            # Extract each variable and its value
            for pair in pairs:
                if '=' in pair:
                    var_part, val_part = pair.split('=')
                    
                    # Figure out which variable this is
                    for var in self.variables[:self.num_equations]:
                        if str(var) in var_part:
                            # Try to parse the value
                            try:
                                value = parse_expr(val_part.strip())
                                user_answer[var] = value
                            except:
                                pass
            
            # Check if all variables are accounted for
            if len(user_answer) != self.num_equations:
                return False
            
            # Verify each value matches our answer
            for var, val in self.answer.items():
                if var in user_answer and abs(float(user_answer[var]) - float(val)) > 1e-3:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def format_answer(self, ans):
        """Format the answer dictionary as a LaTeX string."""
        answer_parts = []
        for var in self.variables[:self.num_equations]:
            answer_parts.append(f"{sp.latex(var)} = {sp.latex(ans[var])}")
        
        return "Answer: " + ", ".join(answer_parts)
    
# Matrix and Vector Operations
class MatrixVectorOperationsQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Enter vectors as [1, 2] and matrices as [[1, 2], [3, 4]] where [[top row], [bottom row]].  Use integers or fractions only."
        # Choose operation type: matrix-matrix, matrix-vector, or vector-vector
        self.operation_type = random.choice([
            'matrix_matrix', 
            'matrix_vector', 
            'vector_vector'
        ])
        
        # Generate the question based on the operation type
        if self.operation_type == 'matrix_matrix':
            self._generate_matrix_matrix_question()
        elif self.operation_type == 'matrix_vector':
            self._generate_matrix_vector_question()
        else:  # vector_vector
            self._generate_vector_vector_question()
            
        # Format the answer
        self.latex_answer = self.format_answer(self.answer)
    
    def _generate_matrix_matrix_question(self):
        # Choose matrix dimensions ensuring multiplication is possible
        m = random.randint(2, 3)  # rows of A
        n = random.randint(2, 3)  # cols of A / rows of B
        p = random.randint(2, 3)  # cols of B
        
        # Generate matrices with small integer entries
        matrix_A = sp.Matrix([[random.randint(-5, 5) for _ in range(n)] for _ in range(m)])
        matrix_B = sp.Matrix([[random.randint(-5, 5) for _ in range(p)] for _ in range(n)])
        
        # Compute the product
        self.answer = matrix_A * matrix_B
        
        # Store the operands
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        
        # Generate question text
        self.question_text = (f"Compute the matrix product AB where:\n"
                              f"$A = {sp.latex(matrix_A)}$ and\n" 
                              f"$B = {sp.latex(matrix_B)}$")
    
    def _generate_matrix_vector_question(self):
        # Choose dimensions
        m = random.randint(2, 3)  # rows of A
        n = random.randint(2, 3)  # cols of A / size of v
        
        # Generate matrix and vector with small integer entries
        matrix_A = sp.Matrix([[random.randint(-5, 5) for _ in range(n)] for _ in range(m)])
        vector_v = sp.Matrix([random.randint(-5, 5) for _ in range(n)])
        
        # Compute the product
        self.answer = matrix_A * vector_v
        
        # Store the operands
        self.matrix_A = matrix_A
        self.vector_v = vector_v
        
        # Generate question text
        self.question_text = (f"Compute the matrix-vector product Av where:\n"
                              f"$A = {sp.latex(matrix_A)}$ and\n" 
                              f"$v = {sp.latex(vector_v)}$")
    
    def _generate_vector_vector_question(self):
        # Choose dimension for vectors
        n = random.randint(2, 4)  # length of vectors
        
        # Generate vectors with small integer entries
        vector_u = sp.Matrix([random.randint(-5, 5) for _ in range(n)])
        vector_v = sp.Matrix([random.randint(-5, 5) for _ in range(n)])
        
        # Compute the dot product
        self.answer = vector_u.dot(vector_v)
        
        # Store the operands
        self.vector_u = vector_u
        self.vector_v = vector_v
        
        # Generate question text
        self.question_text = (f"Compute the dot product u·v where:\n"
                              f"$u = {sp.latex(vector_u)}$ and\n" 
                              f"$v = {sp.latex(vector_v)}$")
    
    def check_answer(self, user_input: str) -> bool:
        """
        user_input should be:
         - a scalar string for dot product,
         - a Python list string like "[1,2,3]" for a vector,
         - or "[[1,2],[3,4]]" for a matrix.
        """
        try:
            if self.operation_type == 'vector_vector':
                # scalar answer
                parsed_expr = parse_expr(user_input)
                return sp.simplify(parsed_expr - self.answer) == 0

            
            parsed_string = f"Matrix({user_input})"
            
            parsed_matrix = parse_expr(parsed_string)
            return sp.simplify(parsed_matrix - self.answer) == sp.zeros(*self.answer.shape)
        except Exception:
            return False
    
    def format_answer(self, ans):
        """Format the answer as a LaTeX string."""
        if self.operation_type == 'vector_vector':
            # Dot product is a scalar
            return f"Answer: {sp.latex(ans)}"
        else:
            # Matrix results
            return f"Answer: {sp.latex(ans)}"


class QuadraticInterceptsQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Write the two x-intercepts, then the y-intercept after a double colon. Example: 1,-2:3 where x1_intercept,x2_intercept:y_intercept. (If there is one root, write it twice, if applicable us fractions)."
        # 1) pick a,b,c so that b^2 - 4ac >= 0
        while True:
            a = random.randint(1, 5)
            b = random.randint(-5, 5)
            c = random.randint(-5, 5)
            if b*b - 4*a*c >= 0:
                break

        self.a, self.b, self.c = a, b, c
        x = sp.symbols('x')

        # 2) compute roots and y-intercept
        roots = sp.solve(a*x**2 + b*x + c, x)
        if len(roots) == 1:
            roots = [roots[0], roots[0]]
        self.roots = roots
        self.y_int = c

        self.answer = (self.roots, self.y_int)

        # 3) question text and LaTeX answer for display
        self.question_text = (
            f"Let $$y = {a}x^2 + ({b})x + ({c})$$\n\n"
            "(i) What are the $x$‑intercepts?\n\n"
            "(ii) What is the $y$‑intercept?"
        )
        # purely for showing the solution:
        self.latex_answer = (
            r"(i)\; x = \frac{-b\pm\sqrt{b^2-4ac}}{2a} "
            r"\;\Rightarrow\; " + sp.latex(sp.Matrix(self.roots)) +
            r"\quad (ii)\; y(0)= " + str(c)
        )

    def check_answer(self, user_input: str) -> bool:
        """
        expects user_input like:
           r1,r2:y0
        where r1,r2 can be integers or fractional expressions.
        """
        # helper to parse a fractional/string expression into a Sympy Expr
        def _to_sympy(o):
            if isinstance(o, (int, float, sp.Rational)):
                return sp.Rational(o)
            return parse_expr(
                str(o),
                transformations=standard_transformations +
                                (implicit_multiplication_application,),
                evaluate=True
            )
        
        split_input = user_input.split(":")
        
        user_roots_temp = split_input[0]
        user_roots_split = user_roots_temp.split(",")
        user_y = split_input[1].strip()
        if not (isinstance(user_roots_split, (list, tuple))):
            return False
        if not (len(user_roots_split) == 2):
            user_roots_split = [user_roots_split[0], user_roots_split[0]]


        user_roots = [parse_expr(e) for e in user_roots_split]
        correct_set = {sp.simplify(r) for r in self.roots}
        user_set    = {sp.simplify(r) for r in user_roots}
        if correct_set != user_set:
            return False

        # parse y-intercept
        u_y = _to_sympy(user_y)
        if sp.simplify(u_y - self.y_int) != 0:
            return False

        return True
    

class PowerOfTenQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Write powers of 10 using `**` or `^`. For example, 100 should be written as 10**2 or 10^2."
        # pick small m,n
        self.m = random.randint(1, 11)
        self.n = random.randint(1, 11)
        
        # build the symbolic expressions
        x = sp.symbols('x')
        self.expr = (10**self.m)**self.n
        # the simplified answer is 10^(m*n)
        self.answer = 10**(self.m * self.n)
        
        # question text
        self.question_text = (
            f"Simplify the expression:\n\n"
            f"$$\\bigl(10^{{{self.m}}}\\bigr)^{{{self.n}}}$$\n\n"
        )
        
        # formatted LaTeX for display of the correct answer
        self.latex_answer = (
            r"10^{" + str(self.m * self.n) + r"}"
        )

    def check_answer(self, user_input: str) -> bool:
        """
        Expects something like "10**k" or "10^k". 
        We replace '^'→'**', parse, and compare.
        """
        try:
            # normalize caret to Python exponent
            s = user_input.strip().replace('^', '**')
            # parse with Sympy
            
            # Check if user didn't just input the equation itself
            no_whitespace = s.replace(" ", "")
            if no_whitespace == f"(10**{self.m})**{self.n}" or no_whitespace == f"(10**{self.n})**{self.m}":
                return False
            user_expr = parse_expr(
                s,
                transformations=standard_transformations +
                                (implicit_multiplication_application,),
                evaluate=True
            )
            # check equivalence
            return sp.simplify(user_expr - self.answer) == 0
        except Exception:
            return False
        
# Matrix Inversion
class MatrixInversionQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Enter vectors as [1, 2] and matrices as [[1, 2], [3, 4]] where [[top row], [bottom row]].  Use integers or fractions only."
        
        # Choose matrix size (2x2 or 3x3)
        self.size = random.choice([2, 3])
        
        # Generate an invertible matrix
        self.matrix = self._generate_invertible_matrix()
        
        # Compute the inverse
        self.answer = self.matrix.inv()
        
        # Generate question text
        self.question_text = f"Find the inverse of the matrix $A = {sp.latex(self.matrix)}$."
        
        # Format the answer
        self.latex_answer = self.format_answer(self.answer)
    
    def _generate_invertible_matrix(self):
        """Generate a random invertible matrix with integer entries."""
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            # Generate random matrix with small integer entries
            if self.size == 2:
                matrix = sp.Matrix([
                    [random.randint(-5, 5), random.randint(-5, 5)],
                    [random.randint(-5, 5), random.randint(-5, 5)]
                ])
            else:  # size == 3
                matrix = sp.Matrix([
                    [random.randint(-3, 3), random.randint(-3, 3), random.randint(-3, 3)],
                    [random.randint(-3, 3), random.randint(-3, 3), random.randint(-3, 3)],
                    [random.randint(-3, 3), random.randint(-3, 3), random.randint(-3, 3)]
                ])
            
            # Check if the matrix is invertible (determinant != 0)
            det = matrix.det()
            if det != 0:
                # Also check that the inverse has reasonably simple entries
                try:
                    inv_matrix = matrix.inv()
                    # Check that all entries in the inverse are rational with small denominators
                    all_simple = True
                    for i in range(self.size):
                        for j in range(self.size):
                            entry = inv_matrix[i, j]
                            # Convert to rational and check denominator
                            rational_entry = sp.Rational(entry)
                            if abs(rational_entry.p) > 20 or abs(rational_entry.q) > 20:
                                all_simple = False
                                break
                        if not all_simple:
                            break
                    
                    if all_simple:
                        return matrix
                except:
                    # If inverse computation fails, try again
                    pass
            
            attempts += 1
        
        # Fallback: create a simple invertible matrix
        if self.size == 2:
            return sp.Matrix([[2, 1], [1, 1]])
        else:
            return sp.Matrix([[2, 0, 1], [0, 1, 0], [1, 0, 1]])
    
    def check_answer(self, latexString: str):
        try:
            
            parsed_string = f"Matrix({latexString})"
            
            parsed_matrix = parse_expr(parsed_string)
            return sp.simplify(parsed_matrix - self.answer) == sp.zeros(*self.answer.shape)
        except Exception:
            return False
    
    def format_answer(self, ans):
        """Format the matrix answer as LaTeX."""
        return f"Answer: $A^{{-1}} = {sp.latex(ans)}$"
    
    
# Trigonometric Limits
class TrigonometricLimitQuestion(BaseQuestion):
    symbol = sp.symbols('x')
    
    def __init__(self):
        self.input_hint = "Enter the limit value. Use fractions like `1/2`, or write `0` for zero, `oo` for infinity. Provide answer in fractions if applicable and make the multiplication explicit if applicable."
        expr, x0, limit_expr = self.random_trig_limit_equation()
        self.equation = expr
        self.to = x0
        self.answer = limit_expr
        self.question_text = f"Compute the limit: $\\lim_{{x \\to {sp.latex(self.to)}}} {sp.latex(self.equation)}$."
        self.latex_answer = self.format_answer(self.answer)
    
    def random_trig_limit_equation(self):
        x = self.symbol
        
        # Choose from common trigonometric limit types
        limit_types = [
            # sin(x)/x type limits
            lambda a: (sp.sin(a*x)/(a*x), 0, 1),
            lambda a: (sp.sin(a*x)/x, 0, a),
            lambda a: (x/sp.sin(a*x), 0, sp.Rational(1,a)),
            
            # (1-cos(x))/x type limits  
            lambda a: ((1 - sp.cos(a*x))/x, 0, 0),
            lambda a: ((1 - sp.cos(a*x))/(a*x), 0, 0),
            lambda a: ((1 - sp.cos(a*x))/(a*x**2), 0, sp.Rational(a,2)),
            
            # tan(x)/x type limits
            lambda a: (sp.tan(a*x)/(a*x), 0, 1),
            lambda a: (sp.tan(a*x)/x, 0, a),
            
            # sin ratio limits
            lambda a, b: (sp.sin(a*x)/sp.sin(b*x), 0, sp.Rational(a,b)),
        ]
        
        # Pick a random limit type
        limit_type = random.choice(limit_types[:7])  # Exclude the two-parameter one for now
        
        # Generate random coefficient (small integers)
        a = random.choice([1, 2, 3, 4, 5])
        
        try:
            expr, x0, expected_limit = limit_type(a)
            
            # Verify the limit computation
            computed_limit = sp.limit(expr, x, x0)
            
            # Handle cases where SymPy might return different forms
            if sp.simplify(computed_limit - expected_limit) == 0:
                return expr, x0, expected_limit
            else:
                return expr, x0, computed_limit
                
        except Exception:
            # Fallback to a simple sin(x)/x limit
            return sp.sin(x)/x, 0, 1
    
    def check_answer(self, latexString: str):
        try:
            parsed_expr = parse_expr(latexString)
            return sp.simplify(parsed_expr - self.answer) == 0
        except Exception:
            return False


# Limits at Infinity
class InfinityLimitQuestion(BaseQuestion):
    symbol = sp.symbols('x')
    
    def __init__(self):
        self.input_hint = "Enter the limit value. Use `oo` for infinity, `-oo` for negative infinity, `zoo` for undefined, or a number/fraction."
        expr, direction, limit_expr = self.random_infinity_limit_equation()
        self.equation = expr
        self.to = direction
        self.answer = limit_expr
        self.question_text = f"Compute the limit: $\\lim_{{x \\to {sp.latex(self.to)}}} {sp.latex(self.equation)}$."
        self.latex_answer = self.format_answer(self.answer)
    
    def random_infinity_limit_equation(self):
        x = self.symbol
        
        # Choose direction: +∞ or -∞
        direction = random.choice([sp.oo, -sp.oo])
        
        # Generate different types of infinity limits
        limit_types = [
            # Rational functions (polynomial ratios)
            self._rational_function_limit,
            # Functions with square roots
            self._sqrt_function_limit,
            # Exponential functions
            self._exponential_function_limit,
            # Functions with fractions and polynomials
            self._mixed_function_limit,
        ]
        
        # Pick a random limit type
        limit_type = random.choice(limit_types)
        expr, computed_limit = limit_type(x, direction)
        
        return expr, direction, computed_limit
    
    def _rational_function_limit(self, x, direction):
        """Generate rational function limits like (ax^n + b)/(cx^m + d)"""
        # Coefficients
        a = random.choice([-3, -2, -1, 1, 2, 3])
        b = random.randint(-5, 5)
        c = random.choice([-3, -2, -1, 1, 2, 3])
        d = random.randint(-5, 5)
        
        # Powers
        n = random.randint(1, 3)
        m = random.randint(1, 3)
        
        # Create the expression
        numerator = a * x**n + b
        denominator = c * x**m + d
        expr = numerator / denominator
        
        # Compute the limit
        limit_result = sp.limit(expr, x, direction)
        
        return expr, limit_result
    
    def _sqrt_function_limit(self, x, direction):
        """Generate limits with square roots like sqrt(x^2 + ax) - x"""
        a = random.randint(1, 5)
        
        # Only use positive infinity for square root expressions to avoid complex numbers
        if direction == -sp.oo:
            direction = sp.oo
        
        expr_types = [
            sp.sqrt(x**2 + a*x) - x,
            sp.sqrt(x**2 + a) - x,
            (x + a) / sp.sqrt(x**2 + 1),
            sp.sqrt(x + a) / x,
        ]
        
        expr = random.choice(expr_types)
        limit_result = sp.limit(expr, x, direction)
        
        return expr, limit_result
    
    def _exponential_function_limit(self, x, direction):
        """Generate exponential limits"""
        a = random.randint(1, 3)
        
        expr_types = [
            sp.exp(x) / x,
            x / sp.exp(x),
            (sp.exp(x) - 1) / x,
            sp.exp(-x) * x,
        ]
        
        expr = random.choice(expr_types)
        limit_result = sp.limit(expr, x, direction)
        
        return expr, limit_result
    
    def _mixed_function_limit(self, x, direction):
        """Generate mixed function limits"""
        a = random.randint(1, 4)
        b = random.randint(1, 4)
        
        expr_types = [
            (a*x + b) / (x + 1),
            x / (x**2 + a),
            (x**2 + a*x) / (b*x**2 + 1),
            (a*x**3 + x) / (b*x**3 + x**2 + 1),
        ]
        
        expr = random.choice(expr_types)
        limit_result = sp.limit(expr, x, direction)
        
        return expr, limit_result
    
    def check_answer(self, latexString: str):
        try:
            parsed_expr = parse_expr(latexString)
            return sp.simplify(parsed_expr - self.answer) == 0
        except Exception:
            return False


# First Order Taylor Series
class FirstOrderTaylorQuestion(BaseQuestion):
    def __init__(self):
        self.input_hint = "Enter the linear approximation as a polynomial. Use `*` for multiplication and `**` for powers. Example: `1 + 2*x` or `1 + 3*(x-1)`, multiplication needs to be explicit. Remember about brackets."
        x = sp.symbols('x')
        self.x = x
        
        # Generate the function, expansion point, and Taylor approximation
        self.func, self.expansion_point, self.taylor_approx = self.generate_taylor_question()
        
        # Create question text
        if self.expansion_point == 0:
            point_text = "0"
        else:
            point_text = sp.latex(self.expansion_point)
            
        self.question_text = (
            f"Find the first-order Taylor series (linear approximation) of "
            f"$f(x) = {sp.latex(self.func)}$ about $x = {point_text}$."
        )
        
        # The answer is the first-order Taylor polynomial
        self.answer = self.taylor_approx
        self.latex_answer = self.format_answer(self.answer)
    
    def generate_taylor_question(self):
        x = self.x
        
        # Function types suitable for first-order Taylor expansion
        function_expansion_pairs = [
            # Trigonometric functions
            (sp.sin(x), 0),
            (sp.cos(x), 0),
            (sp.sin(x), sp.pi/4),
            (sp.cos(x), sp.pi/4),
            (sp.sin(x), sp.pi/6),
            (sp.cos(x), sp.pi/6),
            
            # Exponential and logarithmic
            (sp.exp(x), 0),
            (sp.exp(x), 1),
            (sp.log(x), 1),
            (sp.log(x + 1), 0),
            
            # Power functions
            (sp.sqrt(x), 1),
            (sp.sqrt(x), 4),
            ((1 + x)**sp.Rational(1,2), 0),
            ((1 + x)**sp.Rational(1,3), 0),
            ((1 + x)**2, 0),
            ((1 + x)**sp.Rational(-1,2), 0),
            
            # Rational functions
            (1/(1 + x), 0),
            (1/(1 + x), 1),
            (1/x, 1),
            
            # Polynomial modifications
            (x**2 + 2*x + 1, 1),
            (x**3 - x + 2, 0),
        ]
        
        # Choose a random function-point pair
        func, expansion_point = random.choice(function_expansion_pairs)
        
        # Compute the first-order Taylor polynomial: f(a) + f'(a)(x-a)
        try:
            # Evaluate function at expansion point
            f_at_a = func.subs(x, expansion_point)
            
            # Compute derivative
            f_prime = sp.diff(func, x)
            
            # Evaluate derivative at expansion point
            f_prime_at_a = f_prime.subs(x, expansion_point)
            
            # First-order Taylor polynomial
            taylor_approx = f_at_a + f_prime_at_a * (x - expansion_point)
            
            # Simplify the result
            taylor_approx = sp.expand(taylor_approx)
            
            return func, expansion_point, taylor_approx
            
        except Exception:
            # Fallback to a simple case if computation fails
            func = sp.exp(x)
            expansion_point = 0
            taylor_approx = 1 + x  # e^x ≈ 1 + x near x = 0
            return func, expansion_point, taylor_approx
    
    def check_answer(self, latexString: str):
        try:
            # Parse the user's input
            parsed_expr = parse_expr(latexString)
            
            # Check if the difference simplifies to zero
            difference = sp.simplify(parsed_expr - self.answer)
            return difference == 0
            
        except Exception:
            return False
    
    def format_answer(self, ans):
        """Format the Taylor approximation answer."""
        return f"Answer: $f(x) \\approx {sp.latex(ans)}$"




