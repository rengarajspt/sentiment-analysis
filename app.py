
import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
import numpy as np


# load artifacts
file_name = 'sentiment.pkl'
file_path = 'vector.pkl'

model = joblib.load(file_name)
tfidf = joblib.load(file_path)

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

/* Sidebar background */
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

/* Result card */
.result-card {
    background-color: white;
    padding: 30px;
    border-radius: 10px;
    border-left: 8px solid #8b1c2e;
    text-align: center;
    font-size: 70px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Review App", layout="centered")

# Navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["Review Page", "Review Analysis"]
)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

if page == "Review Page":

  st.markdown('<div class="sub-title">Customer Review</div>', unsafe_allow_html=True)
  st.markdown(""" <h1 style='text-align: center;color:#2E86C1;'>Feedback is the compass for greatness</h3> """, unsafe_allow_html=True)

  star_options = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
  rating = st.radio("How would you like the app", options=star_options, index=3, horizontal=True)

  rating_value = star_options.index(rating) + 1         # Convert stars to numeric rating

  review = st.text_area(
    'Customer review ',
    placeholder="We truly value your opinion",
    height=150 )

  input_df = pd.DataFrame({
        "rating":[rating_value],
        "review":[review]
    })

  X_text = tfidf.transform(input_df["review"])         # TFIDF transform


  X_num = input_df[["rating"]].values           # numeric feature


  X = hstack([X_text, X_num])              # combine features

  pred_ls = model.predict(X)

  pred = np.argmax(pred_ls, axis=1)[0]

  reverse_map = {0:"Negative", 1:"Neutral", 2:"Positive"}

  sentiment = reverse_map[pred]

  if len(rating) == 5 :
        stars = "⭐⭐⭐⭐⭐"
  elif len(rating) == 4 :
        stars = "⭐⭐⭐⭐☆"
  elif len(rating) == 3:
        stars = "⭐⭐⭐☆☆"
  elif len(rating) == 2:
        stars = "⭐⭐☆☆☆"
  else:
        stars = "⭐☆☆☆☆"

  submit = st.button("Submit")

  if submit:
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
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

            <h1 style='color:#f59e0b'>{stars}</h1>

           <p style='font-style:italic; color:#333;'>“{review}”</p>

           <p style="color:#386E1F; font-weight:bold; font-size:28px;">{sentiment}</p>

           </div> """, unsafe_allow_html=True )




# -------------------------------------------------------------------------------------------------------------------------------------------------------

df = pd.read_pickle("reviews_data.pkl")

#  -------------------------------------------------------------------------------------------------------------------------------------------------------
if page == "Review Analysis":
  st.markdown('<div class="sub-title">Review Data Analysis</div>', unsafe_allow_html=True)

  analysis_option = st.selectbox(
    "Select Analysis",
    [
        "Overall Sentiment",
        "Sentiment vs Rating",
        "Keywords by Sentiment",
        "Sentiment vs Helpful Votes",
        "Verified vs Non Verified Users",
        "Review Length vs Sentiment",
        "Sentiment by Location",
        "Sentiment by Platform",
        "Sentiment by Version",
        "Negative Feedback Themes"
    ]
)

# -------- KEYWORD FUNCTION --------
  def get_keywords(texts):
    words = []
    for t in texts:
        w = re.findall(r'\b[a-z]+\b', str(t).lower())
        words.extend(w)
    return Counter(words).most_common(10)

# -------- VISUALS --------

  if analysis_option == "Overall Sentiment":

    st.subheader("Overall Sentiment Distribution")

    sentiment_counts = df["sentiment"].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)


  elif analysis_option == "Sentiment vs Rating":

    st.subheader("Sentiment vs Rating")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="rating", hue="sentiment", ax=ax)
    st.pyplot(fig)


  elif analysis_option == "Keywords by Sentiment":

    st.subheader("Keywords Associated With Sentiment")

    sentiment_type = st.selectbox("Select Sentiment", df["sentiment"].unique())

    keywords = get_keywords(df[df["sentiment"] == sentiment_type]["review"])

    kw_df = pd.DataFrame(keywords, columns=["Word","Frequency"])

    fig, ax = plt.subplots()
    sns.barplot(data=kw_df, x="Frequency", y="Word", ax=ax)
    st.pyplot(fig)


  elif analysis_option == "Sentiment vs Helpful Votes":

    st.subheader("Sentiment vs Helpful Votes")

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="sentiment", y="helpful_votes", ax=ax)
    st.pyplot(fig)


  elif analysis_option == "Verified vs Non Verified Users":

    st.subheader("Verified vs Non Verified Sentiment")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="verified_purchase", hue="sentiment", ax=ax)
    st.pyplot(fig)


  elif analysis_option == "Review Length vs Sentiment":

    st.subheader("Review Length vs Sentiment")

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="sentiment", y="review_length", ax=ax)
    st.pyplot(fig)


  elif analysis_option == "Sentiment by Location":

    st.subheader("Sentiment by Location")

    location_df = df.groupby(["location","sentiment"]).size().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(8,5))
    location_df.plot(kind="bar", ax=ax)
    st.pyplot(fig)


  elif analysis_option == "Sentiment by Platform":

    st.subheader("Sentiment by Platform")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="platform", hue="sentiment", ax=ax)
    st.pyplot(fig)


  elif analysis_option == "Sentiment by Version":

    st.subheader("Sentiment by Version")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="version", hue="sentiment", ax=ax)
    st.pyplot(fig)


  elif analysis_option == "Negative Feedback Themes":

    st.subheader("Common Negative Feedback Words")

    negative_words = get_keywords(df[df["sentiment"]=="Negative"]["review"])
    neg_df = pd.DataFrame(negative_words, columns=["Word","Frequency"])

    fig, ax = plt.subplots()
    sns.barplot(data=neg_df, x="Frequency", y="Word", ax=ax)
    st.pyplot(fig)

