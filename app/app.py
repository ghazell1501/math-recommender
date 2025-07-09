# app.py
import streamlit as st

# 1) import your generators
from question_generation import (
    LinearSystemQuestion,
    MatrixVectorOperationsQuestion,
    LimitQuestion,
    LinearAlgebraQuestion,
    TrigonometryQuestion,
    # RealFunctionQuestion,
    IndefiniteIntegralQuestion,
    DerivativeQuestion,
    DefiniteIntegralQuestion,
    QuadraticInterceptsQuestion,
    PowerOfTenQuestion,
    MatrixInversionQuestion,
    InfinityLimitQuestion,
    TrigonometricLimitQuestion,
    FirstOrderTaylorQuestion
)
from hardcoded_questions import (
    LogProductQuestion,
    LogPowerChangeOfBase,
    ExpLnIdentityQuestion,
    TrigonometryHardcoded,
    GraphMatchingTrig,
    RealFunctionsGraphMatching,
    ComplexNumberHardcoded,
    ComplexRootSelection,
)
from login import login_form
from main_interface import main_interface
from all_questions import all_questions
from check_dictionary import check_dictionary
from student_dashboard import student_dashboard

# 2) initialize session state defaults
for key, default in {
    'logged_in': False,
    'test_started': False,
    'all_questions': False,
    'check_dictionary': False,
    'student_dashboard': False,
    'rightanswers': False,
    'recomcourse': False,
    'results': False,
    'demo': False,
    'user_data': {},
    'answers': {}, 
    'test_score': {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 3) generate those questions only once
if 'all_questions_stored' not in st.session_state:
    st.session_state.all_questions_stored = [
        GraphMatchingTrig(), #0
        TrigonometryQuestion(), #1
        TrigonometryHardcoded(), #2
        ComplexNumberHardcoded(), #3
        ComplexRootSelection(), #4
        LogProductQuestion(), #5
        LogPowerChangeOfBase(), #6
        ExpLnIdentityQuestion(), #7
        LinearSystemQuestion(), #8
        MatrixVectorOperationsQuestion(), #9
        LinearAlgebraQuestion(), #10
        MatrixInversionQuestion(), #11
        LimitQuestion(), #12
        InfinityLimitQuestion(), #13
        TrigonometricLimitQuestion(), #14
        IndefiniteIntegralQuestion(), #15
        DerivativeQuestion(), #16
        FirstOrderTaylorQuestion(), #17
        DefiniteIntegralQuestion(), #18
        QuadraticInterceptsQuestion(), #19
        RealFunctionsGraphMatching(), #20
        PowerOfTenQuestion()  #21
    ]
if st.session_state.logged_in and 'question_answers' not in st.session_state:
    st.session_state.question_answers = {
        st.session_state.user_data["University ID"]:{}
    }
st.set_page_config(
    page_title="ReadySetMath",
    page_icon="🎓",
    layout="wide"
)

# 4) flow control
if not st.session_state.logged_in:
    login_form()
else:
    if st.session_state.all_questions:
        # pull your *same* questions out of session_state
        all_questions(
            st.session_state.all_questions_stored
        )
    elif st.session_state.check_dictionary:
        check_dictionary()
    elif st.session_state.student_dashboard:
        student_dashboard()
    else:
        main_interface()
