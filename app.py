import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="NextStep AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 NextStep AI")
st.subheader("Your Personal Smart Learning Recommendation System")

st.write(
    "Discover what to learn next based on your career goal, "
    "interests, existing skills, and available study time."
)

@st.cache_data
def load_courses():
    return pd.read_csv("courses.csv")

try:
    courses = load_courses()
except FileNotFoundError:
    st.error("courses.csv not found. Keep it in the same folder as app.py.")
    st.stop()

st.sidebar.header("👤 Your Learning Profile")

career = st.sidebar.selectbox(
    "What is your career goal?",
    sorted(courses["career"].unique())
)

interests = st.sidebar.multiselect(
    "Choose your interests",
    sorted(courses["category"].unique())
)

skills = st.sidebar.text_input(
    "What skills do you already know?",
    placeholder="Example: python, coding"
)

weak_topics = st.sidebar.text_input(
    "What topics do you want to improve?",
    placeholder="Example: machine learning"
)

study_hours = st.sidebar.slider(
    "How many hours can you study per week?",
    min_value=1,
    max_value=20,
    value=5
)

level = st.sidebar.selectbox(
    "Choose your current level",
    ["Beginner", "Intermediate"]
)

if st.sidebar.button("Recommend Courses", type="primary"):

    profile_text = " ".join(interests) + " " + weak_topics + " " + career

    if skills.strip():
        profile_text += " " + skills

    course_text = (
        courses["course_name"].fillna("") + " " +
        courses["category"].fillna("") + " " +
        courses["skills"].fillna("") + " " +
        courses["career"].fillna("")
    )

    vectorizer = TfidfVectorizer()
    all_text = list(course_text) + [profile_text]
    matrix = vectorizer.fit_transform(all_text)

    similarity_scores = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    ).flatten()

    results = courses.copy()
    results["Similarity"] = similarity_scores

    results["Career Match"] = (
        results["career"] == career
    ).astype(float)

    results["Level Match"] = results["difficulty"].apply(
        lambda difficulty: 1.0
        if difficulty == level else 0.5
    )

    results["Time Match"] = results["hours"].apply(
        lambda hours: 1.0 if hours <= study_hours * 2 else 0.5
    )

    results["Score"] = (
        results["Similarity"] * 0.50 +
        results["Career Match"] * 0.25 +
        results["Level Match"] * 0.15 +
        results["Time Match"] * 0.10
    ) * 100

    results = results.sort_values(
        by="Score",
        ascending=False
    )

    st.success("Your personalized recommendations are ready!")

    st.write("### 📚 Recommended Courses")

    displayed = 0

    for _, course in results.iterrows():
        if course["hours"] <= study_hours * 2:
            displayed += 1

            with st.container(border=True):
                st.markdown(f"### {displayed}. {course['course_name']}")
                st.write(f"*Category:* {course['category']}")
                st.write(f"*Skills:* {course['skills']}")
                st.write(f"*Difficulty:* {course['difficulty']}")
                st.write(f"*Estimated learning time:* {course['hours']} hours")
                st.write(f"*Career:* {course['career']}")
                st.progress(
                    min(int(course["Score"]), 100),
                    text=f"Recommendation match: {course['Score']:.1f}%"
                )

            if displayed == 5:
                break

    if displayed == 0:
        st.info("No courses match your time preference. Try increasing your study hours.")

else:
    st.info(
        "👈 Fill in your learning profile in the sidebar, "
        "then click *Recommend Courses*."
    )

st.divider()
st.caption("NextStep AI | Personalized Learning Recommendation Project")