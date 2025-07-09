import streamlit as st
from recommender import load_data, get_recommendations
import os
import pandas as pd
import smtplib
from email.message import EmailMessage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def dataframe_to_pdf(df, filename):
    """
    Convert a pandas DataFrame to a PDF file
    """

    if df.empty:
        return  # Skip empty dataframes

    # Define column width ratios
    col_widths = {
        "Topic": 1.0,
        "Concept": 1.5,
        "Link": 6,
    }

    # Calculate figure size based on column count and row count
    base_col_width = 3  # Base width per unit
    fig_width = sum(col_widths.get(col, 1.0) for col in df.columns) * base_col_width
    fig_height = max(2, min(0.6 * len(df), 20))  # Limit height for readability

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Prepare cell text
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='left',
        colWidths=[col_widths.get(col, 1.0) / sum(col_widths.values()) for col in df.columns]
    )

    # Format table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    # Export to PDF
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def answers_to_pdf(answers_df, filename):
    """
    Export answer feedback to PDF, including question text if available.
    """

    with PdfPages(filename) as pdf:
        for _, row in answers_df.iterrows():
            correct = row['Score'] == 1
            emoji = "[✔]" if correct else "[X]"

            topic = row.get("Topic", "Unknown")
            concept = row.get("Concept", "Unknown")
            question_id = row.get("question_id", "N/A")

            user_answer = row.get("Answer", "")
            if pd.isna(user_answer) or str(user_answer).strip() == "":
                user_answer = "Not answered"
            correct_answer = row.get("CorrectAnswer", "")
            if pd.isna(correct_answer) or str(correct_answer).strip() == "":
                correct_answer = "Not Available"

            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.axis('off')

            lines = [
                f"{emoji} Question {question_id}",
                f"• Topic: {topic}",
                f"• Concept: {concept}",
            ]

            lines.extend([
                f"• Your answer: {user_answer}",
                f"• Correct answer: {correct_answer}",
            ])

            y = 0.95
            for line in lines:
                # Remove LaTeX-specific markers for safety
                clean_line = line.replace("$$", "").replace("$", "").replace("\\", "")
                ax.text(0.05, y, clean_line, fontsize=12, va='top')
                y -= 0.07

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


# TO BE CHANGED
CC_EMAIL = "rachel.cavill@maastrichtuniversity.nl" 

def latin1_safe(s: str) -> str:
    return "".join(ch for ch in str(s) if ord(ch) < 256)

def send_email_with_dynamic_attachments(from_email, from_password, to_email, cc_email, subject, body, attachments):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Cc"] = cc_email
    msg.set_content(body)

    for maintype, subtype, filepath in attachments:
        with open(filepath, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                filename=os.path.basename(filepath)
            )

    # Clean up potential invisible characters
    from_email = from_email.strip().replace('\xa0', '').encode('ascii', 'ignore').decode()
    from_password = from_password.strip().replace('\xa0', '').encode('ascii', 'ignore').decode()

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(from_email, from_password)
        smtp.send_message(msg)

def check_dictionary():
    # Check if the user has requested to go to the dashboard
    if st.session_state.get("student_dashboard", False):
        return  # Exit so the app can display the dashboard view instead

    with st.container():
        if st.button("Home"):
            st.session_state.check_dictionary = False
            st.rerun()

        seen, unseen, total, number_of_questions, score = load_data(st.session_state.json_data)

        st.write(f"Your score on the test is {score}/{number_of_questions}")
        
        st.markdown(
            "Notice that the total number of questions may be greater than the number you answered. "
            "That's because questions related to concepts you've never studied are skipped. "
            "However, your score is still calculated out of the full set to give a consistent measure of overall performance."
        )
        
        st.markdown("### 📊 Want to review your full performance?")
        if st.button("Go to Student Dashboard"):
            st.session_state.student_dashboard = True
            st.session_state.check_dictionary = False
            st.rerun()

        with st.expander("📋 See all your answers with feedback"):
            try:
                st.markdown("## 📌 Answer Review")

                # Load from stored CSV
                user_id = st.session_state.user_data["University ID"]
                student_name = st.session_state.user_data["Name"]
                student_email = st.session_state.user_data["Email"]

                csv_path = f"student_data/student_answers_{user_id}.csv"

                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    answers = df[
                        (df["Familiarity"].isin(["Studied in depth", "Studied but forgotten"]))
                    ]

                    if not answers.empty:
                        for idx, row in answers.iterrows():
                            correct = row['Score'] == 1
                            emoji = "✅" if correct else "❌"
                            color = "green" if correct else "red"

                            st.markdown(f"**{emoji} Question {row['question_id']}**")
                            st.markdown(f"- **Topic:** {row['Topic']}")
                            st.markdown(f"- **Concept:** {row['Concept']}")

                            user_answer = row.get("Answer", "")
                            if pd.isna(user_answer) or str(user_answer).strip() == "":
                                user_answer = "Not answered"

                            correct_answer = row.get("CorrectAnswer", "Not Available")
                            if pd.isna(correct_answer) or str(correct_answer).strip() == "":
                                correct_answer = "Not Available"

                            question_text = row.get("QuestionText", "")
                            if pd.notna(question_text) and isinstance(question_text, str) and question_text.strip():
                                cleaned_question_text = (
                                    question_text.replace("<br>", " ")
                                                .replace("<br/>", " ")
                                                .replace("<br />", " ")
                                )
                                st.markdown(f"- **Question text:** {cleaned_question_text.strip()}")
                                
                            st.markdown(f"- **Your answer:** `{user_answer}`")
                            st.markdown(f"- **Correct answer:** `{correct_answer}`")
                            st.markdown("---")
                    else:
                        st.info("No answers found.")
                else:
                    st.warning("No test data found.")
            except Exception as e:
                st.error("Something went wrong while loading answers.")
                st.exception(e)

        if score >= 0.8 * number_of_questions:
            st.title("🎉 Great job! You did extremely well, looks like you are all set for this Master's. Enjoy the summer :)")
            return

        recommendations_seen, recommendations_unseen = get_recommendations(seen, unseen, total, number_of_questions)

        with st.container():
            if not recommendations_seen.empty:
                st.markdown("**💡 Based on your test results, here are the top 3 things you need to improve:**")
                top_seen = recommendations_seen.iloc[:3]
                for _, row in top_seen.iterrows():
                    st.markdown(f"- {row['Topic']}: [{row['Concept'].title()}]({row['Link']})", unsafe_allow_html=True)

                if len(recommendations_seen) > 3:
                    with st.expander("**Here are the rest of the things you can consider brushing up on**"):
                        for _, row in recommendations_seen.iloc[3:].iterrows():
                            st.markdown(f"- {row['Topic']}: [{row['Concept'].title()}]({row['Link']})", unsafe_allow_html=True)

            if not recommendations_unseen.empty:
                st.markdown("**📘 These are the top 3 things that you haven't seen yet, but will need soon**")
                top_unseen = recommendations_unseen.iloc[:3]
                for _, row in top_unseen.iterrows():
                    st.markdown(f"- {row['Topic']}: [{row['Concept'].title()}]({row['Link']})", unsafe_allow_html=True)

                if len(recommendations_unseen) > 3:
                    with st.expander("**Here are the rest of the things you can consider learning**"):
                        for _, row in recommendations_unseen.iloc[3:].iterrows():
                            st.markdown(f"- {row['Topic']}: [{row['Concept'].title()}]({row['Link']})", unsafe_allow_html=True)

        #st.session_state.question_answers[st.session_state.user_data["University ID"]] =  {}
        st.markdown("---")

        try:
            sender = st.secrets["credentials"]["email"]
            password = st.secrets["credentials"]["app_password"]

            # PDF paths
            seen_pdf_path = f"recommendations_seen_{user_id}.pdf"
            unseen_pdf_path = f"recommendations_unseen_{user_id}.pdf"
            answers_pdf_path = f"answers_{user_id}.pdf"

            # Create attachments list
            attachments = []

            # Read answers (used in both email body and PDF)
            df_answers = pd.read_csv(csv_path)
            answers_to_include = df_answers[df_answers["Familiarity"].isin(["Studied in depth", "Studied but forgotten"])]
            if not answers_to_include.empty:
                answers_to_pdf(answers_to_include, answers_pdf_path)
                attachments.append(("application", "pdf", answers_pdf_path))

            # Always include the CSV
            attachments.append(("text", "csv", csv_path))

            # Default: no recommendations unless score < 80%
            include_seen = include_unseen = False

            if score < 0.8 * number_of_questions:
                if not recommendations_seen.empty:
                    dataframe_to_pdf(recommendations_seen, seen_pdf_path)
                    attachments.append(("application", "pdf", seen_pdf_path))
                    include_seen = True

                if not recommendations_unseen.empty:
                    dataframe_to_pdf(recommendations_unseen, unseen_pdf_path)
                    attachments.append(("application", "pdf", unseen_pdf_path))
                    include_unseen = True

            # Generate personalized message based on what is included
            message_parts = [
                f"Hi {student_name},",
                "",
                f"Congratulations on completing your test! Your score is {score}/{number_of_questions}.",
                "",
                "Attached are:",
                "- A CSV file containing your detailed test results.",
                "- A PDF with your answer review and feedback."
            ]

            if include_seen and include_unseen:
                message_parts.append("- A PDF with recommended topics you've seen but struggled with.")
                message_parts.append("- A PDF with recommended topics you haven't studied yet but will need.")
            elif include_seen:
                message_parts.append("- A PDF with recommendations for topics you have studied but struggled with.")
            elif include_unseen:
                message_parts.append("- A PDF with recommendations for new topics you haven’t seen yet but are important.")
            else:
                message_parts.append("Since you scored 80% or higher, no topic recommendations are included. Great job!")

            message_parts.append("")
            message_parts.append("Best regards,")
            message_parts.append("ReadySetMath")

            email_subject = latin1_safe("Your Test Results & Recommendations")
            email_body = latin1_safe("\n".join(message_parts))

            # Prevent resending if already sent
            if st.session_state.get("email_sent", False):
                return  # Skip resending

            # Send email
            send_email_with_dynamic_attachments(
                from_email=sender,
                from_password=password,
                to_email=student_email,
                cc_email=CC_EMAIL,
                subject=email_subject,
                body=email_body,
                attachments=attachments
            )

            # Set the flag so it won’t send again
            st.session_state["email_sent"] = True

            st.success(
                f"✅ Your test results and personalized recommendations have been emailed to {student_email} from readysetmath01@gmail.com. "
                "Please check your inbox (and spam/junk folder) so you don’t miss it!"
            )

        except smtplib.SMTPAuthenticationError as auth_err:
            st.error(
                "❌ SMTP Authentication Error: Username and/or password not accepted.\n\n"
                "Please verify your .env file and that you are using a valid Gmail App Password.\n"
                f"Detailed error message:\n{auth_err}"
            )
        except Exception as e:
            st.error("❌ Unable to create or send your PDF/email. Please check your SMTP setup and .env.")
            st.exception(e)
    
    st.balloons()
