import streamlit as st
import pandas as pd
import os

RECOMMENDED_COURSES = ["", "", "", ""]
TASKS = [
    "Complete Module 1 Quiz",
    "Review Lecture Notes",
    "Join Study Group Session",
    "Submit Assignment 2"
]

def main_interface():
    st.header(f"Welcome, {st.session_state.user_data['Name']}! 👋")
    
    st.title("Learning Readiness Assessment")

    if "test_started" not in st.session_state:
        st.session_state.test_started = False

    if not st.session_state.test_started:
        with st.form("Test Initiation"):
            st.subheader("Subject Familiarity Check")

            TOPICS = [
                "Complex numbers",
                "Calculus (Limits, Differentiation and Integration)",
                "Linear Algebra (Linear equations, Matrix products and Dot products)",
                "Real functions (Graphs, Exponentials and Logarithms)",
                "Trigonometry"
            ]

            initial_familiarity = {}
            all_selected = True  # Flag to check if all topics were answered

            for topic in TOPICS:
                response = st.radio(
                    f"{topic} familiarity:",
                    ["Studied", "Never Studied"],
                    key=topic
                )
                initial_familiarity[topic] = response
                if response == "--Select familiarity--":
                    all_selected = False

            st.session_state.user_familiarity = {
                "Trigonometry": initial_familiarity["Trigonometry"],
                "Graphs of sin(x) and cos(x)": initial_familiarity["Trigonometry"],
                "Evaluating functions at special angles": initial_familiarity["Trigonometry"],
                "Angle addition identities": initial_familiarity["Trigonometry"],

                "Complex Numbers": initial_familiarity["Complex numbers"],
                "Graphing on the complex plane": initial_familiarity["Complex numbers"],
                "Finding roots of a quadratic equation": initial_familiarity["Complex numbers"],

                "Derivative of polynomial equations": initial_familiarity["Calculus (Limits, Differentiation and Integration)"],
                "Derivative of sine, cosine, exponential and logarithmic functions": initial_familiarity["Calculus (Limits, Differentiation and Integration)"],
                "Limits at infinity": initial_familiarity["Calculus (Limits, Differentiation and Integration)"],
                "Standard trigonometric limits": initial_familiarity["Calculus (Limits, Differentiation and Integration)"],
                "Standard limits": initial_familiarity["Calculus (Limits, Differentiation and Integration)"],
                "Taylor approximations": initial_familiarity["Calculus (Limits, Differentiation and Integration)"],
                "Indefinite integrals": initial_familiarity["Calculus (Limits, Differentiation and Integration)"],
                "Definite integrals": initial_familiarity["Calculus (Limits, Differentiation and Integration)"],

                "Linear Equations": initial_familiarity["Linear Algebra (Linear equations, Matrix products and Dot products)"],
                "Matrix multiplication": initial_familiarity["Linear Algebra (Linear equations, Matrix products and Dot products)"],
                "2x2 inverses": initial_familiarity["Linear Algebra (Linear equations, Matrix products and Dot products)"],
                "Matrix vector products": initial_familiarity["Linear Algebra (Linear equations, Matrix products and Dot products)"],
                "Dot products": initial_familiarity["Linear Algebra (Linear equations, Matrix products and Dot products)"],
                "Linear Systems": initial_familiarity["Linear Algebra (Linear equations, Matrix products and Dot products)"],

                "Intercepts": initial_familiarity["Real functions (Graphs, Exponentials and Logarithms)"],
                "Exponential properties": initial_familiarity["Real functions (Graphs, Exponentials and Logarithms)"],
                "Logarithm properties": initial_familiarity["Real functions (Graphs, Exponentials and Logarithms)"],
                "Graphing logarithm and exponential": initial_familiarity["Real functions (Graphs, Exponentials and Logarithms)"],
            }

            if st.form_submit_button("Start Test Now!!"):
                if not all_selected:
                    st.warning("Please select your familiarity for all topics.")
                else:
                    st.session_state.initial_familiarity = st.session_state.user_familiarity.copy()
                    st.session_state.all_questions = True
                    # st.session_state.demo = True
                    st.rerun()
