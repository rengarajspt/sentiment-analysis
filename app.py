import streamlit as st

# Page configuration MUST be first Streamlit command
st.set_page_config(page_title="Customer Review App", layout="centered")

import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re


# -----------------------------
# Load ML artifacts
# -----------------------------
model = joblib.load("sentiment.pkl")
tfidf = joblib.load("vector.pkl")


# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #f8f6f2;
}

/* Main title */
.main-title {
    color: #8b1c2e;
    font-size: 50px;
    font-weight: 700;
    text-align: center;
    letter-spacing: 2px;
}

/* Subtitle */
.sub-title {
    color: #8b1c2e;
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    letter-spacing: 4px;
    margin-top: 20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

/* Predict button */
.stButton>button {
    background-color: #8b1c2e;
    color: white;
    font-size: 18px;
    border-radius: 8px;
    padding: 10px 25px;
}

.stButton>button:hover {
    background-color: #6f1423;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.selectbox(
    "Navigate",
    ["Review Page", "Review Analysis"]
)


# =====================================================
# PAGE 1 : REVIEW INPUT
# =====================================================
if page == "Review Page":

    st.markdown('<div class="sub-title">Customer Review</div>', unsafe_allow_html=True)

    st.markdown(
        "<h3 style='text-align:center;color:#2E86C1;'>Feedback is the compass for greatness</h3>",
        unsafe_allow_html=True
    )

    # Rating input
    star_options = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]

    rating = st.radio(
        "How would you rate the app?",
        options=star_options,
        index=3,
        horizontal=True
    )

    rating_value = star_options.index(rating) + 1


    # Review text
    review = st.text_area(
        "Customer Review",
        placeholder="We truly value your opinion...",
        height=150
    )


    submit = st.button("Submit Review")


    if submit:

        if review.strip() == "":
            st.warning("Please enter a review.")

        else:

            # Convert to dataframe
            input_df = pd.DataFrame({
                "rating": [rating_value],
                "review": [review]
            })


            # Text vectorization
            X_text = tfidf.transform(input_df["review"])


            # Numeric rating feature
            X_num = input_df[["rating"]].values


            # Combine features
            X = hstack([X_text, X_num])


            # Prediction
            pred_ls = model.predict(X)
            pred = np.argmax(pred_ls, axis=1)[0]


            reverse_map = {
                0: "Negative",
                1: "Neutral",
                2: "Positive"
            }

            sentiment = reverse_map[pred]


            # Star visualization
            if rating_value == 5:
                stars = "⭐⭐⭐⭐⭐"
            elif rating_value == 4:
                stars = "⭐⭐⭐⭐☆"
            elif rating_value == 3:
                stars = "⭐⭐⭐☆☆"
            elif rating_value == 2:
                stars = "⭐⭐☆☆☆"
            else:
                stars = "⭐☆☆☆☆"


            # Result card
            st.markdown(
                f"""
                <div style="
                background:white;
                padding:30px;
                border-radius:15px;
                text-align:center;
                box-shadow:0 4px 12px rgba(0,0,0,0.2);
                font-family:sans-serif;
                ">

                <h2 style='color:#4CAF50;'>Thank You For Your Feedback!</h2>

                <h1 style='colo
