import streamlit as st
import pandas as pd
import os

def login_form():
    with st.form("Login"):
        st.title("Welcome to the Mathematics Self-Assesment")
        st.write("This assessment is designed to help us place you in the right course in block 1 of the Masters in Systems Biology. It is **OK**, indeed expected, that most/all of you will not be able to complete everything here, especially within the time limit specified. The main aim here is to start identifying the gaps in your knowledge, so that we can help you fill them in September.")
        st.write("Set aside just one hour and complete as much as you can in that time. Do this on your own, without looking things up. If you remember that you could do something in the past (but have now forgotten it), just note that down.")
        st.write("Start by entering your details. On the next page, select the topics you've already studied. Based on this, you'll receive a custom test tailored to your background. After completing it, you'll discover your strengths and weaknesses in mathematics—along with personalized course recommendations to help you improve for your courses.")
        st.header("Student Login")
        name = st.text_input("Full Name")
        uni_id = st.text_input("University ID")
        st.caption("Enter your ID without the 'i' (e.g., 6404161)")
        email = st.text_input("University Email")

        if st.form_submit_button("Login"):
            errors = []

            # Validate name 
            if name.isnumeric():
                errors.append("You name must contain only letters")
            
            # Validate University ID
            if not uni_id.isnumeric():
                errors.append("University ID must contain only numbers.")
            elif len(uni_id) != 7:
                errors.append("University ID must be exactly 7 digits long.")

            # Validate Email
            if "@" not in email:
                errors.append("Email must contain '@'.")
            else:
                domain = email.split("@")[-1]
                if domain.lower() != "student.maastrichtuniversity.nl":
                    errors.append("Email must be a Maastricht University student email (e.g., name@student.maastrichtuniversity.nl).")

            # Normalize inputs
            uni_id = uni_id.strip()
            email = email.strip()
            name = name.strip()

            # Show errors or save
            if errors:
                for err in errors:
                    st.error(err)
            else:
                user_data = {
                    "Name": name,
                    "University ID": uni_id,
                    "Email": email
                }

                st.session_state.logged_in = True
                st.session_state.user_data = user_data

                st.success("Login Successful!")
                st.rerun()
