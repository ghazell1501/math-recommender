import sympy as sp
import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st
from sympy.parsing.sympy_parser import parse_expr
from sympy import simplify

class BaseQuestion:
    def __init__(self):
        self.input_hint = "Use Python-style syntax: `**` for powers, `*` for multiplication, `sqrt(x)` for square root, and `log(x)` for natural log."

    def format_answer(self, ans):
        """Consistently format the answer in LaTeX with an 'Answer: ' prefix."""
        return "Answer: " + sp.latex(ans)
    
    def check_sym_answer(String: str, correct_expr):
        """Check symbolic answers by parsing and simplifying the difference."""
        try:
            parsed_expr = parse_expr(String)
            return sp.simplify(parsed_expr - correct_expr) == 0
        except Exception:
            return False
    
    def check_numeric_answer(String: str, correct_val, tol=1e-3):
        """Check numerical answers using float conversion and a tolerance."""
        try:
            parsed_expr = parse_expr(String)
            return abs(float(parsed_expr) - float(correct_val)) < tol
        except Exception:
            return False
        
    def check_string_answer(self, user_input: str, correct_input: str) -> bool:
        """
        Rejects expressions that are not simplified.
        Ensures the student did not just repeat the original expression.
        """
        try:
            user_expr = parse_expr(user_input, evaluate=True)
            correct_expr = parse_expr(correct_input, evaluate=True)

            # Check if mathematically equivalent
            if sp.simplify(user_expr - correct_expr) != 0:
                return False

            # Check if the user answer is trivially identical to the original (not simplified)
            # This prevents log(x*y) from being accepted when the correct answer is log(x) + log(y)
            original_expr = parse_expr(self.original_expr, evaluate=True) if hasattr(self, 'original_expr') else None
            if original_expr and sp.simplify(user_expr - original_expr) == 0:
                return False  # user answer matches the unsimplified original input

            return True
        except Exception:
            return False


# Trigonometry
class TrigonometryHardcoded(BaseQuestion):
    def __init__(self):
        self.input_hint = "Use trigonometric identities and write the simplified form using `*` for multiplication and `+` for addition."
        self.x = sp.symbols('x')
        self.y = sp.symbols('y')
        self.point = random.choice([0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2])
        self.question_text = f"What is the value of ${sp.latex("sin(x+y)")}$."
        self.answer = "sin(x)*cos(y) + sin(y)*cos(x)"
        self.latex_answer = self.format_answer(self.answer)
    
    def check_answer(self, latexString: str):
        no_whitespace = latexString.replace(" ", "")
        final_string = no_whitespace.strip().lower()
        
        print("Processed input:", final_string)

        if final_string in ["sin(x+y)", "sin(y+x)"]:
            return False  # Reject unsimplified identity

        return self.check_string_answer(latexString, self.answer)
    
class GraphMatchingTrig(BaseQuestion):
    def __init__(self):
        self.x_vals = np.linspace(0, 2 * np.pi, 500)

        # Randomly assign sin and cos to labels
        functions = [("sin(x)", np.sin(self.x_vals)), ("cos(x)", np.cos(self.x_vals))]
        random.shuffle(functions)

        self.graph_labels = {'A': functions[0], 'B': functions[1]}

        self.question_text = (
            "Two graphs are shown below in the interval $0 \\leq x \\leq 2\\pi$. "
            "One is $\\sin(x)$ and the other is $\\cos(x)$. Match the correct function to each graph."
        )

        self.correct_mapping = {
            label: func for label, (func, _) in self.graph_labels.items()
        }

        self.latex_answer = self.format_answer(
            f"Graph A: {self.correct_mapping['A']}, Graph B: {self.correct_mapping['B']}"
        )
        self.answer = f"Graph A: {self.correct_mapping['B']}, Graph B: {self.correct_mapping['A']}"

    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 3))
        for label, (func_label, y_vals) in self.graph_labels.items():
            ax.plot(self.x_vals, y_vals, label=f"Graph {label}")
        ax.set_title("Match Graphs to Functions")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.grid(True)
        ax.legend()
        return fig

    def check_answer(self, user_match: dict):
        score = 0
        for label in ['A', 'B']:
            if user_match.get(label) == self.correct_mapping[label]:
                score += 1
        return score == 2

    def render(self):
        st.pyplot(self.plot(), use_container_width=False)

        with st.form(f"form_graph_match"):
            st.write("Match the graphs to the correct functions:")
            match_A = st.selectbox("Graph A", ["Select", "sin(x)", "cos(x)"], key="graph_A")
            match_B = st.selectbox("Graph B", ["Select", "sin(x)", "cos(x)"], key="graph_B")
            submitted = st.form_submit_button("Submit")

            if submitted:
                if match_A == "Select" or match_B == "Select":
                    st.warning("Please make a selection for both Graph A and Graph B.")
                else:
                    st.success("Your answer has been submitted successfully! (You can change it at any time before the final submit).")
                    is_correct = self.check_answer({'A': match_A, 'B': match_B})

                    # Save both user answer and correct answer
                    correct_ans_str = f"Graph A: {self.correct_mapping['A']}, Graph B: {self.correct_mapping['B']}"
                    user_ans_str = f"Graph A: {match_A}, Graph B: {match_B}"
                    return {"answer": user_ans_str, "is_correct": is_correct, "correct_answer": correct_ans_str}
        return None
  
# Exponential/Log
class LogProductQuestion(BaseQuestion):
    def __init__(self):
        self.x, self.y = sp.symbols('x y')
        self.input_hint = "Use log(x) for logarithms."
        self.question_text = "What is the value of $\\log(x \cdot y)$?"
        self.answer = "log(x) + log(y)"
        self.latex_answer = self.format_answer(self.answer)

    def check_answer(self, user_input: str):
        return self.check_string_answer(user_input, self.answer)

class LogPowerChangeOfBase(BaseQuestion):
    def __init__(self):
        self.x, self.y = sp.symbols('x y')
        self.input_hint = "Answer in lowercase."
        self.question_text = "What is the value of $\\log_x(x^y)$?"
        self.answer = "y"
        self.latex_answer = self.format_answer(self.answer)

    def check_answer(self, user_input: str):
        return self.check_string_answer(user_input, self.answer)

class ExpLnIdentityQuestion(BaseQuestion):
    def __init__(self):
        self.x = sp.symbols('x')
        self.question_text = "What is the value of $e^{\\ln(x)}$?"
        self.input_hint = "Use exp(x) for e to the power of x, and ln(x) or log(x) for natural logarithm."
        self.answer = "x"
        self.latex_answer = self.format_answer(self.answer)

    def check_answer(self, latexString: str):
        no_whitespace = latexString.replace(" ", "")
        final_string = no_whitespace.strip().lower()
        
        print("Processed input:", final_string)

        if final_string in ["exp(ln(x))"]:
            return False  # Reject unsimplified identity

        return self.check_string_answer(latexString, self.answer)

# Complex Number
class ComplexRootSelection(BaseQuestion):
    def __init__(self):
        super().__init__()
        self.question_text = "Find all solutions to $x^2 + 1 = 0$"
        self.options = ["i", "-i", "1", "-1", "√i"]
        self.correct_answers = {"i", "-i"}

        self.answer = "i, -i"
        self.latex_answer = self.format_answer(self.answer)

    def check_answer(self, selected_options):
        return set(selected_options) == self.correct_answers

    def render(self):
        selected = st.multiselect(
            "Select all correct solutions:",
            self.options,
            key="complex_solutions"
        )

        if st.button("Submit", key="check_complex_roots"):
            st.success("Your answer has been submitted successfully! (You can change it at any time before the final submit).")
            return {
                "answer": ", ".join(selected),
                "is_correct": self.check_answer(selected),
                "correct_answer": self.answer
            }
        return None
   
class ComplexNumberHardcoded(BaseQuestion):
    def __init__(self):
        self.points = {
            'A': (1, 0),    # 1
            'B': (1, 1),    # 1+i
            'C': (-1, 1),   # -1+i
            'D': (0, -1),   # -i
            'E': (-1, -1)   # -1-i
        }

        self.correct_mapping = {
            'A': "(i) 1",
            'B': "(ii) 1+i",
            'C': "(iii) -1+i",
            'D': "(iv) -i",
            'E': "(v) -1-i"
        }

        self.options = [
            "(i) 1", "(ii) 1+i", "(iii) -1+i", "(iv) -i", "(v) -1-i"
        ]

        self.question_text = (
            "Match the labeled points on the complex plane with their correct complex numbers."
        )

        self.answer = ', '.join([f"{k}: {v}" for k, v in self.correct_mapping.items()])
        self.latex_answer = self.format_answer(self.answer)

    def plot(self):
        fig, ax = plt.subplots(figsize=(3, 3))  # Smaller plot
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.grid(True)

        for label, (x, y) in self.points.items():
            ax.plot(x, y, 'ro')
            ax.text(x + 0.1, y + 0.1, label, fontsize=12)

        return fig

    def check_answer(self, user_match: dict):
        score = 0
        for label, selected in user_match.items():
            if selected == self.correct_mapping[label]:
                score += 1
        return score

    def render(self):
        st.pyplot(self.plot(), use_container_width=False)

        user_match = {}
        with st.form("complex_plane_form"):
            cols = st.columns(len(self.points))
            for i, label in enumerate(self.points.keys()):
                with cols[i]:
                    user_match[label] = st.selectbox(
                        f"Point {label}",
                        ["Select"] + self.options,
                        key=f"match_{label}"
                    )

            submitted = st.form_submit_button("Submit")

        if submitted:
            if any(v == "Select" for v in user_match.values()):
                st.warning("Please make a selection for all points.")
            else:
                st.success("Your answer has been submitted successfully! (You can change it at any time before the final submit).")
                score = self.check_answer(user_match)
                return {
                    "answer": ', '.join([f"{k}: {v}" for k, v in user_match.items()]),
                    "is_correct": score == 5,
                    "correct_answer": self.answer
                }
        return None
    
class RealFunctionsGraphMatching:
    def __init__(self):
        self.question_text = (
            "Match each graph to the correct function from the list below"
        )

        # Define function labels and their numpy equivalents
        self.functions = {
            "y = ln(x)": lambda x: np.log(x),
            "y = exp(x)": lambda x: np.exp(x),
            "y = x^2": lambda x: x ** 2,
            "y = tan(x)": lambda x: np.tan(x),
        }

        self.labels = list("ABCD")
        self.shuffled_functions = random.sample(list(self.functions.items()), len(self.functions))
        self.graph_labels = dict(zip(self.labels, self.shuffled_functions))
        self.correct_mapping = {
            label: func_label for label, (func_label, _) in self.graph_labels.items()
        }

        self.answer = ', '.join([f"{label}: {func}" for label, func in self.correct_mapping.items()])
        self.latex_answer = "Answer: " + self.answer

    def plot_single(self, label, func_label, func_callable):
        x_vals = np.linspace(-10, 10, 500)
        with np.errstate(divide='ignore', invalid='ignore'):
            y_vals = func_callable(x_vals)
            y_vals = np.where(np.abs(y_vals) > 1e6, np.nan, y_vals)  # avoid infinite spikes

        fig, ax = plt.subplots(figsize=(2.8, 2.2))
        ax.plot(x_vals, y_vals, label=f"Graph {label}", linewidth=1)
        ax.set_title(f"Graph {label}", fontsize=9)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-5, 20)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True)
        return fig

    def check_answer(self, user_match: dict):
        score = 0
        for label in self.labels:
            if user_match.get(label) == self.correct_mapping[label]:
                score += 1
        return score == 4

    def render(self):
        # 2x2 layout for 4 graphs
        cols = st.columns(2)
        for i, (label, (func_label, func_callable)) in enumerate(self.graph_labels.items()):
            with cols[i % 2]:
                st.pyplot(self.plot_single(label, func_label, func_callable), use_container_width=False)

        user_match = {}
        with st.form("real_func_graph_match_form"):
            for label in self.labels:
                user_match[label] = st.selectbox(
                    f"Graph {label}",
                    ["Select", "y = ln(x)", "y = exp(x)", "y = x^2", "y = tan(x)"],
                    key=f"realfunc_{label}"
                )
            submitted = st.form_submit_button("Submit")

        if submitted:
            if any(v == "Select" for v in user_match.values()):
                st.warning("Please make a selection for all graphs.")
            else:
                st.success("Your answer has been submitted successfully! (You can change it at any time before the final submit).")
                is_correct = self.check_answer(user_match)
                user_ans_str = ', '.join([f"{k}: {v}" for k, v in user_match.items()])
                correct_ans_str = self.answer
                return {
                    "answer": user_ans_str,
                    "is_correct": is_correct,
                    "correct_answer": correct_ans_str,
                    "question_text": self.question_text
                }

        return None

