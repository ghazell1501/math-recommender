import streamlit as st
from streamlit_float import *
from collections import defaultdict
import sympy as sp
import numpy as np
import pandas as pd
import os
import json
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor, parse_expr
import re
from unidecode import unidecode
from utility import parse_intercepts_to_latex, parse_vars_to_latex

def all_questions(question_list):
    """
    question_list is the *same* list you stored in session_state
    """
    float_init()
    # Title and Math Notation Helper
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Home"):
            st.session_state.all_questions = False
            st.rerun()

    with col2:
        col2.float("overflow:auto;max-height:500px;background-color:#087099;")
        with st.expander("💡 Math Notation Help"):
            st.markdown("""
            **Common Math Expressions:**
            - Square root: `sqrt(x)`
            - Exponents: `x**2`, `x**3`
            - Fractions: `x/y` or `(a+b)/(c+d)`
            - Pi: `pi`
            - Euler's number: `exp(x)`
            - Natural log: `ln(x)`
            - Log base 10: `log(x)`
            - Sine: `sin(x)`
            - Cosine: `cos(x)`
            - Tangent: `tan(x)`
            - Absolute value: `Abs(x)`
            - Infinity: `oo`
            - Complex number: `2 + 3*I`
            - Multiplication: `2*x`
            
            **Examples:**
            - √2: `sqrt(2)`
            - x²+3x-5: `x**2 + 3*x - 5`
            - sin(π/4): `sin(pi/4)`
            - e^x: `exp(x)`
            - e: `exp(1)`
            - ln(2): `log(2)`
            """)

    # Mapps subtopics to the generators
    class_subtopic_names = {
        question_list[0].__class__: ["Trigonometry", "Graphs of sin(x) and cos(x)"],
        question_list[1].__class__: ["Trigonometry", "Evaluating functions at special angles"],
        question_list[2].__class__: ["Trigonometry", "Angle addition identities"],
        question_list[3].__class__: ["Complex numbers", "Graphing on the complex plane"],
        question_list[4].__class__: ["Complex numbers", "Finding roots of a quadratic equation"],
        question_list[5].__class__: ["Real functions", "Logarithm properties"],
        # question_list[6].__class__: ["Real functions", "Graphing logarithm and exponential"],
        question_list[6].__class__: ["Real functions", "Logarithm properties"],
        question_list[7].__class__: ["Real functions", "Exponential properties"],
        question_list[8].__class__: ["Linear algebra", "Linear Equations"],
        question_list[9].__class__: ["Linear algebra", "Matrix vector products"],
        question_list[10].__class__: ["Linear algebra", "Matrix multiplication"],
        question_list[11].__class__: ["Linear algebra", "2x2 inverses"], # Needs to be changed to matrix inversion
        question_list[12].__class__: ["Calculus", "Standard limits"],
        question_list[13].__class__: ["Calculus", "Limits at infinity"],
        question_list[14].__class__: ["Calculus", "Standard trigonometric limits"],
        # question_list[12] was RealFunctionQuestion and is not included
        question_list[15].__class__: ["Calculus", "Indefinite integrals"],
        question_list[16].__class__: ["Calculus", "Derivative of polynomial equations"],
        question_list[17].__class__: ["Calculus", "Taylor approximations"], #CHANGE TO TAYLOR SERIES
        question_list[18].__class__: ["Calculus", "Definite integrals"],
        question_list[19].__class__: ["Real functions", "Intercepts"],
        question_list[20].__class__: ["Real functions", "Graphing logarithm and exponential"],
        question_list[21].__class__: ["Real functions", "Exponential properties"],
    }

    # class_subtopic_names = {
    #     question_list[0].__class__: ["Trigonometry", "Graphs of sin(x) and cos(x)"],
    #     question_list[1].__class__: ["Trigonometry", "Evaluating functions at special angles"],
    #     question_list[2].__class__: ["Trigonometry", "Angle addition identities"],
    #     question_list[3].__class__: ["Complex numbers", "Graphing on the complex plane"],
    #     question_list[4].__class__: ["Complex numbers", "Finding roots of a quadratic equation"],
    #     question_list[5].__class__: ["Real functions", "Logarithm properties"],
    #     question_list[6].__class__: ["Real functions", "Graphing logarithm and exponential"],
    #     question_list[7].__class__: ["Real functions", "Exponential properties"],
    #     question_list[8].__class__: ["Real functions", "Exponential function graph"],
    #     question_list[9].__class__: ["Linear algebra", "Linear Equations"],
    #     question_list[10].__class__: ["Linear algebra", "Matrix multiplication"],
    #     question_list[11].__class__: ["Linear algebra", "2x2 inverses"],
    #     question_list[12].__class__: ["Linear algebra", "Matrix vector products"],
    #     question_list[13].__class__: ["Linear algebra", "Dot products"],
    #     question_list[14].__class__: ["Calculus", "Derivative of polynomial equations"],
    #     question_list[15].__class__: ["Calculus", "Derivative of sine, cosine, exponential and logarithmic functions"],
    #     question_list[16].__class__: ["Calculus", "Limits at infinity"],
    #     question_list[17].__class__: ["Calculus", "Standard trigonometric limits"],
    #     question_list[18].__class__: ["Calculus", "Taylor approximations"],
    #     question_list[19].__class__: ["Calculus", "Indefinite integrals"],
    #     question_list[20].__class__: ["Calculus", "Definite integrals"],
    #     question_list[21].__class__: ["Real functions", "Graphs"],
    # }


    custom_topic_map = {
        "Graphs of sin(x) and cos(x)": "Trigonometry",
        "Evaluating functions at special angles": "Trigonometry",
        "Angle addition identities": "Trigonometry",

        "Graphing on the complex plane": "Complex numbers",
        "Finding roots of a quadratic equation": "Complex numbers",

        "Derivative of polynomial equations": "Calculus (Limits, Differentiation and Integration)",
        "Taylor approximations": "Calculus (Limits, Differentiation and Integration)",
        "Definite integrals": "Calculus (Limits, Differentiation and Integration)",
        "Indefinite integrals": "Calculus (Limits, Differentiation and Integration)",
        "Limits at infinity": "Calculus (Limits, Differentiation and Integration)",
        "Standard limits": "Calculus (Limits, Differentiation and Integration)",
        "Standard trigonometric limits": "Calculus (Limits, Differentiation and Integration)",

        "Linear Equations": "Linear Algebra (Linear equations, Matrix products and Dot products)",
        "Matrix vector products": "Linear Algebra (Linear equations, Matrix products and Dot products)",
        "Matrix multiplication": "Linear Algebra (Linear equations, Matrix products and Dot products)",
        "2x2 inverses": "Linear Algebra (Linear equations, Matrix products and Dot products)",

        "Intercepts": "Real functions (Graphs, Exponentials and Logarithms)",
        "Logarithm properties": "Real functions (Graphs, Exponentials and Logarithms)",
        "Graphing logarithm and exponential": "Real functions (Graphs, Exponentials and Logarithms)",
        "Exponential properties": "Real functions (Graphs, Exponentials and Logarithms)",
        "Exponential function graph": "Real functions (Graphs, Exponentials and Logarithms)",
    }

    st.title("Math Test")
    
    results = {}

    # Group questions by custom topics
    grouped_questions = defaultdict(list)
    for idx, question in enumerate(question_list):
        topic, subtopic = class_subtopic_names[question.__class__]
        custom_topic = custom_topic_map.get(subtopic, topic)
        initial_fam = st.session_state.initial_familiarity.get(subtopic, "Never studied")
        if initial_fam == "Studied":
            grouped_questions[custom_topic].append((idx, question, subtopic))
    
    # Show a warning if the student marked everything as "Never Studied"
    if all(fam == "Never Studied" for fam in st.session_state.initial_familiarity.values()):
        st.info("📘 Since you indicated that you haven't studied any of the topics, you don't need to take the test. We'll give you personalized recommendations based on what’s most important.")
    else:
        st.warning("⚠️ Make sure you click **Submit** on each question before proceeding. Only submitted answers will be counted. If you have any doubts about how to write certain expression, on the top right corner there is a list of common math notations.")

    # Render groups
    question_counter = 1
    for custom_topic, questions in grouped_questions.items():
        st.header(f"📘 {custom_topic}")
        for idx, question, subtopic in questions:
            st.markdown(f"### Question {question_counter}")
            question_counter += 1

            st.markdown(question.question_text)

            familiarity_options = ["Studied in depth", "Studied but forgotten"]

            new_fam = st.radio(
                "Familiarity level",
                familiarity_options,
                index=0,
                key=f"fam_{idx}"
            )
            st.session_state.user_familiarity[subtopic] = new_fam
            if new_fam == "--Select familiarity--":
                st.warning("⚠️ Please select your familiarity level.")

            # Checks for questions wit the render attribute and checks if the user answer is correct for them
            if hasattr(question, "render"):
                result = question.render()
                store = False
                if isinstance(result, dict):
                    user_answer = result.get("answer", "")
                    is_correct = result.get("is_correct", False)
                    correct_answer = result.get("correct_answer", "Not Available")
                    question_text = result.get("question_text", "Not Available")
                    store = True

                elif isinstance(result, bool):
                    # Only correctness was returned
                    user_answer = ""  # or a fallback string if you can infer an answer
                    is_correct = result
                    store = True

                else:
                    # Nothing submitted or invalid result
                    user_answer = ""
                    is_correct = False
                    store = False

                # Storing the result
                if store:
                    results[idx] = {
                            "Answer": user_answer,
                            "Score": int(is_correct),
                            "Concept": subtopic,
                            "Topic": custom_topic,
                            "Familiarity": new_fam,
                            "CorrectAnswer": correct_answer,
                            "QuestionText": question_text
                        }
                    
                    st.markdown("---")
                continue

            form_key = f"qform_{idx}"
            with st.form(form_key):
                hint = getattr(question, "input_hint", None)
                if hint:
                    st.info(f"*Hint: {hint}*")

                # answer checking for parametarized questions
                user_answer = st.text_input("Your answer", key=f"ans_{idx}").lower()
                submitted = st.form_submit_button("Submit")

                if submitted and user_answer.strip():
                    try:
                        transformations = standard_transformations + (
                            implicit_multiplication_application,
                            convert_xor
                        )
                        user_answer = user_answer.replace("^","**")
                        # Regex matching for matrix
                        patern_matrix = re.compile("^\[.*\]$") # patern that checks for [*]
                        pattern_intercept = re.compile(r'^.*?,.*?:.*$') # patern that checks for x1,x2:y
                        pattern_linear_system = re.compile(
                            r'^(?=.*\bx=)'     # positive lookahead: must contain x=
                            r'(?=.*\by=)'      # positive lookahead: must contain y=
                            r'(?:'             # non-capturing group for the actual pattern
                            r'[xyz]=[^,]*'     # x, y, or z followed by = and non-comma chars
                            r'(?:,[xyz]=[^,]*)*'  # zero or more additional ,key=value pairs
                            r')$'
                        )
                        user_answer = unidecode(user_answer)
                        
                        # Display matrix
                        if (patern_matrix.match(user_answer)):
                            user_answer = f"Matrix({user_answer})"
                            parsed = parse_expr(user_answer, transformations=transformations)
                            st.text("Here is your answer (It may be simplified):")
                            st.latex(sp.latex(parsed))
                        # Display intercept
                        elif (pattern_intercept.match(user_answer)):
                            user_answer = user_answer.replace(" ", "")
                            st.text("Here is your answer (It may be simplified):")
                            parsed = parse_intercepts_to_latex(user_answer)
                            st.latex(parsed)
                        # Display x y and optionaly z
                        elif (pattern_linear_system.match(user_answer)):
                            user_answer = user_answer.replace(" ", "")
                            st.text("Here is your answer (It may be simplified):")
                            parsed = parse_vars_to_latex(user_answer)
                            st.latex(parsed)
                        # Display base case
                        else:
                            parsed = parse_expr(user_answer, transformations=transformations)
                            st.text("Here is your answer (It may be simplified):")
                            st.latex(sp.latex(parsed))
            
                    except Exception as e:
                        st.warning(f"Could not render your input as latex, so here is a plain text representation:")
                        st.text(user_answer)
                    # Checking if the answer is correct
                    try:
                        is_correct = question.check_answer(user_answer)
                        st.success("Your answer has been submitted successfully! (You can change it at any time before the final submit).")
                    except Exception:
                        st.error("Error checking answer.")
                        is_correct = False
                    # Sotring the answer 
                    results[idx] = {
                        "Answer": user_answer.strip(),
                        "Score": int(is_correct),
                        "Concept": subtopic,
                        "Topic": custom_topic,
                        "Familiarity": new_fam,
                        "CorrectAnswer": getattr(question, "answer", "Not Available"),
                        "QuestionText": getattr(question, "question_text", "Not Available")
                    }
                   

            st.markdown("---")

    # ✅ Save after loop
    if "question_answers" not in st.session_state:
        st.session_state.question_answers = {}

    st.session_state.question_answers.update(results)


    
    # ✅ Confirmation checkbox before submission
    st.markdown("---")
    st.subheader("✅ Final Submission")
    confirm = st.checkbox("I have clicked Submit for all the questions I answered and I'm ready to finish the test.")

    if st.button("Submit My Answers"):
        if not confirm:
            st.warning("Please confirm that you have submitted your answers before proceeding.")
        elif "--Select familiarity--" in st.session_state.user_familiarity.values():
            st.warning("⚠️ Please make sure you've selected a familiarity level for every question before submitting.")
        else:
            # Step 1: Ensure all questions are represented
            for idx, question in enumerate(question_list):
                sub_discp = class_subtopic_names[question.__class__]
                subtopic = sub_discp[1]

                if idx not in st.session_state.question_answers:
                    familiarity = st.session_state.user_familiarity.get(
                        subtopic,
                        st.session_state.initial_familiarity.get(subtopic, "Never studied")
                    )
                    st.session_state.question_answers[idx] = {
                        "Answer": "",
                        "Score": 0,
                        "Concept": subtopic,
                        "Topic": custom_topic_map.get(subtopic, sub_discp[0]),
                        "Familiarity": familiarity,
                        "CorrectAnswer": getattr(question, "answer", ""),
                        "QuestionText": getattr(question, "question_text", "") 
                    }

            # Proceed to save and rerun
            answer_rows = []
            for idx, data in st.session_state.question_answers.items():
                answer_rows.append({
                    "Question#": idx,
                    "Answer": data.get("Answer", ""),
                    "Score": data.get("Score", 0),
                    "Concept": data.get("Concept", ""),
                    "Topic": data.get("Topic", ""),
                    "Familiarity": data.get("Familiarity", "")
                })

            st.session_state.all_questions = False
            st.session_state.check_dictionary = True

            # Save JSON and CSV
            user_id = st.session_state.user_data["University ID"]
            def transform_to_nested_json(user_id, flat_answers):
                nested = {}
                for i, data in flat_answers.items():
                    if isinstance(i, int):
                        q_key = f"Q{i + 1}"
                        nested[q_key] = {
                            "Answer": data.get("Answer", ""),
                            "Score": data.get("Score", 0),
                            "Concept": data.get("Concept", ""),
                            "Topic": data.get("Topic", ""),
                            "Familiarity": data.get("Familiarity", "")
                        }
                return {str(user_id): nested}

            json_data = transform_to_nested_json(user_id, st.session_state.question_answers)
            with open(f"answers_{user_id}.json", "w", encoding="utf-8") as jf:
                json.dump(json_data, jf, indent=4)
            st.session_state.json_data = json_data

            flat_rows = []
            for i, data in st.session_state.question_answers.items():
                if isinstance(i, int):
                    flat_rows.append({
                        "student_id": user_id,
                        "question_id": f"Q{i + 1}",
                        "Answer": data.get("Answer", ""),
                        "Score": data.get("Score", 0),
                        "Concept": data.get("Concept", ""),
                        "Topic": data.get("Topic", ""),
                        "Familiarity": data.get("Familiarity", ""),
                        "CorrectAnswer": data.get("CorrectAnswer", ""),
                        "QuestionText": data.get("QuestionText", "") 
                    })

            
            csv_df = pd.DataFrame(flat_rows)
            csv_df = csv_df.astype(str)

            os.makedirs("student_data", exist_ok=True)
            csv_df.to_csv(f"student_data/student_answers_{user_id}.csv", index=False)

            st.rerun()



    