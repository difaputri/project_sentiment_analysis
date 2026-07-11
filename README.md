# 📱 Sentiment Analysis MyTelkomsel

Analisis sentimen ulasan pengguna aplikasi **MyTelkomsel** menggunakan Machine Learning. Proyek ini melakukan klasifikasi ulasan menjadi **Positive**, **Neutral**, dan **Negative** berdasarkan ulasan yang diambil dari Google Play Store.

---

## 📌 Project Overview

Tujuan dari proyek ini adalah membangun model klasifikasi sentimen terhadap ulasan pengguna aplikasi MyTelkomsel menggunakan pendekatan Machine Learning.

Tahapan yang dilakukan meliputi:

- Data Collection
- Text Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Modeling
- Model Evaluation
- Deployment menggunakan Streamlit

---

## 📊 Dataset

- **Sumber:** Google Play Store
- **Aplikasi:** MyTelkomsel
- **Jumlah Data Awal:** 5.000 ulasan
- **Jumlah Data Setelah Preprocessing:** 4.130 ulasan

Dataset dikumpulkan menggunakan library:

```python
from google_play_scraper import reviews, Sort
```

---

## ⚙️ Text Preprocessing

Tahapan preprocessing yang dilakukan meliputi:

- Cleaning Text
- Case Folding
- Tokenization
- Slang Word Normalization
- Stopword Removal
- Stemming (Sastrawi)

---

## 📈 Exploratory Data Analysis

Visualisasi yang dilakukan meliputi:

- Distribusi Sentimen
- WordCloud Sentimen Positif
- WordCloud Sentimen Negatif
- Top Frequent Words
- Bigram Analysis

---

## 🧠 Feature Engineering

Representasi fitur menggunakan:

- TF-IDF Vectorizer
- Count Vectorizer

---

## 🤖 Machine Learning Models

Model yang dibandingkan pada penelitian ini adalah:

- Naive Bayes
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)

---

## 📊 Model Evaluation

Evaluasi model menggunakan beberapa metrik, yaitu:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Seluruh model dibandingkan untuk menentukan model dengan performa terbaik berdasarkan hasil evaluasi.

---

## 🚀 Deployment

Model terbaik kemudian di-deploy menggunakan **Streamlit**.

Fitur aplikasi:

- Input ulasan pengguna
- Prediksi sentimen
- Confidence Score
- Menampilkan hasil preprocessing

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Sastrawi
- Streamlit
- Joblib
- Matplotlib
- WordCloud

---

## 📁 Project Structure

```
project_sentiment_analysis/
│
├── app.py
├── requirements.txt
├── svm_model.pkl
├── tfidf.pkl
├── README.md
└── Project_Sentiment.ipynb
```

---

## ▶️ Installation

Clone repository

```bash
git clone https://github.com/username/project_sentiment_analysis.git
```

Masuk ke folder project

```bash
cd project_sentiment_analysis
```

Install seluruh dependency

```bash
pip install -r requirements.txt
```

Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

---

## 📷 Demo

<img width="953" height="631" alt="image" src="https://github.com/user-attachments/assets/6b1afc10-b5bf-42bb-8ad1-b22362082d6a" />


---

## 🌐 Live Demo

https://sentiment-analysis-mytelkomsel-difa.streamlit.app/

---

## 👤 Author

**Difa Putri Chairunisa**
