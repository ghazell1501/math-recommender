import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

def student_dashboard():
    if st.button("Back"):
        st.session_state.student_dashboard = False
        st.session_state.check_dictionary = True
        st.rerun()

    st.title("🎓 Your Performance Summary")

    scores_folder = "student_data"

    # Get current student's ID
    user_id = st.session_state.get("user_data", {}).get("University ID")

    if not user_id:
        st.error("User ID not found. Please return to the home page and log in.")
        return

    student_file = f"student_answers_{user_id}.csv"
    file_path = os.path.join(scores_folder, student_file)

    if not os.path.exists(file_path):
        st.error(f"No data found for student ID: {user_id}")
        return

    # Load and clean data
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if not {'score', 'topic', 'question_id'}.issubset(df.columns):
        st.error("CSV format mismatch. Expected columns: 'score', 'topic', 'question_id'.")
        return

    df['score'] = df['score'].astype(int)
    total_questions = df['question_id'].nunique()
    total_correct = df['score'].sum()
    total_mistakes = (df['score'] == 0).sum()
    percentage = round(total_correct / total_questions * 100, 2)

    # Clean topic names
    df["topic"] = df["topic"].str.extract(r"^([^()]+)").iloc[:, 0].str.strip()

    # 1. Summary
    st.header("1. Your Score Summary")
    st.markdown(f"**Total Questions:** {total_questions}")
    st.markdown(f"✅ **Correct:** {total_correct}")
    st.markdown(f"❌ **Mistakes:** {total_mistakes}")
    st.markdown(f"📊 **Final Score:** {percentage:.2f}%")

    # 2. Strength by Topic (Radar Chart)
    st.header("2. Your Strength by Topic")
    st.markdown("This radar chart shows how strong you are in each topic. Higher values mean fewer mistakes.")

    df["mistake"] = 1 - df["score"]
    topic_mistakes = df.groupby("topic").agg(
        total_mistakes=("score", lambda x: (x == 0).sum()),
        total_questions=("score", "count")
    ).reset_index()
    
    if not topic_mistakes.empty:
        topic_mistakes["mistake_rate"] = topic_mistakes["total_mistakes"] / topic_mistakes["total_questions"]
        topic_mistakes["strength"] = 1 - topic_mistakes["mistake_rate"]

        categories = topic_mistakes["topic"].tolist()
        strengths = topic_mistakes["strength"].tolist()

        # Close the radar chart loop
        categories += [categories[0]]
        strengths += [strengths[0]]

        fig_radar = go.Figure(
            data=[
                go.Scatterpolar(
                    r=strengths,
                    theta=categories,
                    fill='toself',
                    name='Strength',
                    fillcolor="rgba(255,0,0,0.3)",
                    line_color="red"
                )
            ],
            layout=go.Layout(
                title=go.layout.Title(text='Your Strength by Topic (Radar Chart)'),
                polar={'radialaxis': {'visible': True, 'range': [0, 1]}},
                showlegend=False
            )
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Summary of strongest/weakest topics
        strongest = topic_mistakes.sort_values("strength", ascending=False).iloc[0]["topic"]
        weakest = topic_mistakes.sort_values("strength", ascending=True).iloc[0]["topic"]
        st.info(f"**Strongest topic:** {strongest}\n\n**Weakest topic:** {weakest}")

    # 3. Familiarity Chart
    if "familiarity" in df.columns:
        st.header("3. Your Familiarity Across Topics")
        fam_counts = df.groupby(["topic", "familiarity"]).size().reset_index(name="count")
        fig_fam = px.bar(
            fam_counts,
            x="topic",
            y="count",
            color="familiarity",
            barmode="group",
            title="Your Familiarity with Each Topic"
        )
        st.plotly_chart(fig_fam, use_container_width=True)
        st.dataframe(fam_counts, use_container_width=True)
