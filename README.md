#  Sentiment Analysis of Customer Reviews

## 🚀 Project Overview

This project analyzes **customer reviews** and classifies them into **Positive, Neutral, or Negative sentiments** using **Machine Learning and Deep Learning (LSTM)** techniques.

The project also includes an **interactive Streamlit dashboard** where users can submit reviews and view **sentiment predictions and visual insights**.

---

## 🎯 Objectives

* Classify customer reviews into **Positive, Neutral, and Negative sentiments**
* Understand how **ratings, platforms, and user behavior affect sentiment**
* Build an **interactive dashboard** for review analysis
* Compare **multiple ML models** and select the **best performing model**

---

## 🧠 Machine Learning Models Used

The following models were trained and evaluated:

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting
* **LSTM Deep Learning Model**

The **best performing model** was selected based on **Accuracy, Precision, Recall, F1 Score, and ROC-AUC**.

---

## 📂 Dataset Features

| Column            | Description                                     |
| ----------------- | ----------------------------------------------- |
| review            | Customer review text                            |
| rating            | Rating given by user (1–5)                      |
| helpful_votes     | Number of helpful votes                         |
| review_length     | Length of review                                |
| platform          | Platform where review was posted                |
| language          | Language of review                              |
| location          | User location                                   |
| version           | Application version                             |
| verified_purchase | Whether the purchase was verified               |
| sentiment         | Target variable (Positive / Neutral / Negative) |

---

## ⚙️ Technologies Used

### Programming

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Joblib

### Deployment

* Streamlit

---

## 🔄 Project Workflow

### 1️⃣ Data Collection

Customer review dataset containing text and metadata.

### 2️⃣ Data Cleaning

* Removed missing values
* Text preprocessing
* Lowercasing
* Removing special characters

### 3️⃣ Feature Engineering

* Tokenization
* Text padding
* Review length extraction
* Encoding categorical variables

### 4️⃣ Model Training

Multiple ML models were trained and compared.

### 5️⃣ Deep Learning Model

An **LSTM neural network** was implemented for text classification.

### 6️⃣ Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### 7️⃣ Best Model Selection

The best model was saved using **Joblib**.

---

## 📊 Streamlit Dashboard Features

The Streamlit app allows users to:

### ✍️ Submit Reviews

Users can:

* Select rating using star slider
* Enter a review
* Get **instant sentiment prediction**

### 📈 Review Analysis Dashboard

Interactive visualizations include:

1. Overall Sentiment Distribution
2. Sentiment vs Rating
3. Keywords Associated with Sentiment
4. Sentiment vs Helpful Votes
5. Verified vs Non-Verified User Sentiment
6. Review Length vs Sentiment
7. Sentiment by Location
8. Sentiment by Platform
9. Sentiment by App Version
10. Common Negative Feedback Themes

---

## 🧪 Model Architecture (LSTM)

```
Embedding Layer
        ↓
LSTM Layer
        ↓
Dropout
        ↓
Dense Layer (Softmax)
```

---

## 📁 Project Structure

```
sentiment-analysis-project
│
├── data
│   └── reviews.csv
│
├── models
│   └── best_model.pkl
│
├── notebooks
│   └── sentiment_analysis.ipynb
│
├── streamlit_app
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```
https://github.com/rengarajspt/sentiment-analysis/blob/main/sentiment_analysis.ipynb
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```
streamlit run app.py
```

---

## 📷 Sample Dashboard

The Streamlit dashboard includes:

* Interactive sentiment prediction
* Customer review input interface
* Visual analytics of review data

---

## 🔮 Future Improvements

* Add **WordCloud visualization**
* Deploy on **Streamlit Cloud**
* Use **BERT / Transformers for better accuracy**
* Add **real-time API integration**

---

## 👨‍💻 Author

**Rengaraj**

Data Science Enthusiast | Machine Learning Developer

---

## ⭐ If you like this project

Please give it a **star on GitHub** ⭐

