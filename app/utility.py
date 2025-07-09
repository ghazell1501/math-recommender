import re

def parse_intercepts_to_latex(input_string):
    """
    Parse a string like '1,(sqrt(1)/2):4' into LaTeX format.
    Format: x1_intercept,x2_intercept:t_intercept
    """
    # Split by colon to separate x-intercepts from t-intercept
    parts = input_string.split(':')
    if len(parts) != 2:
        raise ValueError("Invalid format. Expected 'x1,x2:t'")
    
    x_part, t_part = parts
    
    # Split x-intercepts by comma
    x_intercepts = x_part.split(',')
    if len(x_intercepts) != 2:
        raise ValueError("Invalid format. Expected exactly 2 x-intercepts")
    
    x1, x2 = [intercept.strip() for intercept in x_intercepts]
    t = t_part.strip()
    
    # internal method to convert mathematical expressions to LaTeX
    def to_latex_math(expr):
        # Remove outer parentheses if they wrap the entire expression
        expr = expr.strip()
        if expr.startswith('(') and expr.endswith(')'):
            # Check if parentheses are balanced and wrap entire expression
            paren_count = 0
            for i, char in enumerate(expr):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0 and i < len(expr) - 1:
                        break
            else:
                if paren_count == 0:
                    expr = expr[1:-1]
        
        # Replace mathematical functions and operators
        expr = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', expr)
        expr = re.sub(r'\*', r' \\cdot ', expr)
        expr = expr.replace('/', r' / ')
        
        return expr
    
    # Convert each part to LaTeX
    x1_latex = to_latex_math(x1)
    x2_latex = to_latex_math(x2)
    t_latex = to_latex_math(t)
    
    # Format as LaTeX with labels
    latex_output = f"""x_1 = {x1_latex}, \\quad x_2 = {x2_latex}, \\quad y = {t_latex}"""
    
    return latex_output

def parse_vars_to_latex(input_string):
    """
    Parse a string like 'x=2,y=sqrt(3)/3,z=-4/3' into LaTeX format.
    x and y are guaranteed, z is optional. Order can vary.
    """
    # Split by comma to get individual assignments
    assignments = [part.strip() for part in input_string.split(',')]
    
    # Dictionary to store parsed values
    coords = {}
    
    # Parse each assignment
    for assignment in assignments:
        if '=' not in assignment:
            raise ValueError(f"Invalid assignment format: '{assignment}'. Expected 'variable=value'")
        
        var, value = assignment.split('=', 1)
        var = var.strip()
        value = value.strip()
        
        if var not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid variable: '{var}'. Expected x, y, or z")
        
        coords[var] = value
    
    # Check that x and y are present
    if 'x' not in coords or 'y' not in coords:
        raise ValueError("Both x and y variables are required")
    
    # Internal method to convert mathematical expression to LaTeX format
    def to_latex_math(expr):
        expr = expr.strip()
        
        # Remove outer parentheses if they wrap the entire expression
        if expr.startswith('(') and expr.endswith(')'):
            paren_count = 0
            for i, char in enumerate(expr):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0 and i < len(expr) - 1:
                        break
            else:
                if paren_count == 0:
                    expr = expr[1:-1]
        
        # Replace mathematical functions and operators
        expr = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', expr)
        expr = re.sub(r'\*', r' \\cdot ', expr)
        
        # Handle fractions - convert a/b to \frac{a}{b} for better display
        # This regex handles simple fractions
        expr = re.sub(r'([^/\s]+)/([^/\s]+)', r'\\frac{\1}{\2}', expr)
        
        return expr
    
    # Convert variables to LaTeX
    x_latex = to_latex_math(coords['x'])
    y_latex = to_latex_math(coords['y'])
    
    # Build the output string
    latex_parts = [f"x = {x_latex}", f"y = {y_latex}"]
    
    if 'z' in coords:
        z_latex = to_latex_math(coords['z'])
        latex_parts.append(f"z = {z_latex}")
    
    latex_output = ", \\quad ".join(latex_parts)
    
    return latex_output