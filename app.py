import re
import string

import joblib
import numpy as np
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =========================================================
# Konfigurasi halaman
# =========================================================
st.set_page_config(
    page_title="Analisis Sentimen Ulasan MyTelkomsel",
    page_icon="📱",
    layout="centered",
)

# =========================================================
# Load model & vectorizer (hasil training di notebook)
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("svm_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf


model, tfidf = load_artifacts()

# =========================================================
# Kamus slang & stopword (disalin persis dari notebook)
# =========================================================
slang_dict = {
    # Pronoun
    "sy": "saya", "sya": "saya", "gw": "saya", "gua": "saya", "gue": "saya",
    "akuh": "aku", "aq": "aku", "ak": "aku", "km": "kamu", "kmu": "kamu",
    "lu": "kamu", "loe": "kamu", "u": "kamu",

    # Negasi
    "g": "tidak", "ga": "tidak", "gak": "tidak", "gk": "tidak", "ngga": "tidak",
    "nggak": "tidak", "engga": "tidak", "enggak": "tidak", "tdk": "tidak", "tak": "tidak",

    # Singkatan umum
    "yg": "yang", "dg": "dengan", "dgn": "dengan", "dr": "dari", "krn": "karena",
    "utk": "untuk", "bwt": "buat", "jd": "jadi", "jdi": "jadi", "pdhl": "padahal",
    "trs": "terus", "sm": "sama", "klu": "kalau", "kalo": "kalau", "klo": "kalau",
    "kl": "kalau", "tp": "tapi", "tpi": "tapi", "bs": "bisa", "bsa": "bisa",
    "dpt": "dapat", "dapet": "dapat", "bkn": "bukan", "brp": "berapa",
    "kli": "kali", "gj": "tidak jelas",

    # Intensitas
    "bgt": "banget", "bngt": "banget", "bnget": "banget", "bgtt": "banget", "bgtss": "banget",

    # Waktu
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum",
    "hr": "hari", "hri": "hari", "thn": "tahun", "bln": "bulan",

    # Kuantitas
    "byk": "banyak", "bnyk": "banyak", "sdkt": "sedikit",

    # Lain-lain
    "aja": "saja", "doang": "saja", "cm": "cuma", "cmn": "cuma", "kyk": "seperti",
    "org": "orang", "ank": "anak", "ortu": "orang tua", "mrk": "mereka",
    "mnrt": "menurut", "smoga": "semoga", "moga": "semoga", "jgn": "jangan",
    "jngn": "jangan", "pk": "pakai", "pke": "pakai", "pkai": "pakai",
    "byr": "bayar", "akhirakhir": "akhir",

    # Istilah review aplikasi
    "app": "aplikasi", "apk": "aplikasi", "apps": "aplikasi", "apknya": "aplikasinya",
    "eror": "error", "erorr": "error", "errorr": "error",
    "lemot": "lambat", "lelet": "lambat", "ngelag": "lag", "lagg": "lag",
    "gagalin": "gagal", "ngeload": "memuat", "loading": "memuat", "load": "memuat",
    "loginnya": "login", "loginn": "login", "logout": "keluar",
    "otpnya": "otp", "verif": "verifikasi", "notif": "notifikasi",
    "cs": "customer service", "cust": "customer", "adminnya": "admin",
    "telkomselnya": "telkomsel", "mytsel": "mytelkomsel", "mytselnya": "mytelkomsel", "tsel": "telkomsel",
    "paketnya": "paket", "kuotanya": "kuota", "internetan": "internet",
    "signal": "sinyal", "sinyalnya": "sinyal",

    # Sapaan
    "bpk": "bapak", "pak": "bapak", "bu": "ibu",

    "abis": "habis", "aj": "saja", "ajah": "saja", "ajaa": "saja",
    "gpp": "tidak apa apa", "gapapa": "tidak apa apa", "gapapaa": "tidak apa apa",
    "pls": "tolong",
    "udhh": "sudah",
    "bgttt": "banget",
    "mantul": "mantap",
    "ok": "baik", "okey": "baik", "oke": "baik", "okee": "baik",
    "bgs": "bagus", "bgus": "bagus",
    "smpai": "sampai",
    "trus": "terus",
    "skrg": "sekarang", "skrng": "sekarang",
    "jdinya": "jadi",
    "abalabal": "abal abal", "ajh": "saja", "ajhh": "saja", "ahir": "akhir",
    "jlek": "jelek", "kouta": "kuota", "slalu": "selalu",
    "aplnya": "aplikasi", "sndiri": "sendiri",
}

custom_stopword = {
    # Partikel
    "nih", "sih", "deh", "dong", "lah", "mah", "kok",
    # Kata percakapan
    "aja", "ajaa", "ajah", "yah", "ya", "yaa", "nah", "kan",
    # Kata ganti/partikel kurang informatif
    "nya", "pun",
    # Singkatan yang mungkin lolos normalisasi
    "yg", "utk", "dg", "dgn", "dr",
    # Ekspresi
    "wkwk", "wkwkwk", "wk", "hehe", "hehehe", "huhu", "xixi",
    # Sapaan
    "min", "admin",
}

_stopword_factory = StopWordRemoverFactory()
stopword_id = set(_stopword_factory.get_stop_words())
stopword_id.update(custom_stopword)

# Pertahankan kata negasi (penting untuk sentimen)
for negasi in ["tidak", "tak", "bukan", "belum"]:
    stopword_id.discard(negasi)

_stemmer_factory = StemmerFactory()
stemmer = _stemmer_factory.create_stemmer()


# =========================================================
# Pipeline preprocessing (identik dengan notebook)
# =========================================================
def cleaning(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # URL
    text = re.sub(r"@\w+", "", text)                       # mention
    text = re.sub(r"#\w+", "", text)                        # hashtag
    text = text.replace(".", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"\d+", "", text)                         # angka
    text = re.sub(                                          # emoji
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        "",
        text,
        flags=re.UNICODE,
    )
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def case_folding(text: str) -> str:
    return text.lower() if isinstance(text, str) else text


def tokenize(text: str) -> list:
    return text.split()


def normalize_slang(tokens: list) -> list:
    return [slang_dict.get(word, word) for word in tokens]


def remove_stopwords(tokens: list) -> list:
    return [word for word in tokens if word not in stopword_id]


def stemming(tokens: list) -> str:
    return stemmer.stem(" ".join(tokens))


def preprocess(text: str) -> str:
    text = cleaning(text)
    text = case_folding(text)
    tokens = tokenize(text)
    tokens = normalize_slang(tokens)
    tokens = remove_stopwords(tokens)
    return stemming(tokens)


LABEL_STYLE = {
    "positive": {"emoji": "😊", "color": "#4CAF50", "label_id": "Positif"},
    "negative": {"emoji": "😞", "color": "#F44336", "label_id": "Negatif"},
    "neutral": {"emoji": "😐", "color": "#FFC107", "label_id": "Netral"},
}


def predict_sentiment(raw_text: str):
    clean_text = preprocess(raw_text)
    vectorized = tfidf.transform([clean_text])

    prediction = model.predict(vectorized)[0]

    # LinearSVC tidak punya predict_proba, jadi confidence didekati
    # dari decision_function lewat softmax sebagai skor relatif antar kelas.
    confidence = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(vectorized)[0]
        scores = np.atleast_1d(scores)
        if scores.shape[0] == 1:
            # Kasus biner: decision_function hanya 1 nilai
            scores = np.array([-scores[0], scores[0]])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        classes = model.classes_
        confidence = dict(zip(classes, probs))

    return prediction, clean_text, confidence


# =========================================================
# UI Streamlit
# =========================================================
st.title("📱 Analisis Sentimen Ulasan Aplikasi")
st.caption("Model: TF-IDF + Support Vector Machine (SVM) — dilatih dari ulasan MyTelkomsel di Google Play Store")

st.write(
    "Masukkan kalimat ulasan (misalnya tentang aplikasi, layanan, atau produk) "
    "untuk memprediksi sentimennya."
)

user_input = st.text_area(
    "Kalimat ulasan",
    placeholder="Contoh: aplikasinya sering error dan lemot banget, tolong diperbaiki",
    height=120,
)

predict_clicked = st.button("🔍 Prediksi Sentimen", type="primary", use_container_width=True)

if predict_clicked:
    if not user_input.strip():
        st.warning("Silakan masukkan kalimat terlebih dahulu.")
    else:
        with st.spinner("Menganalisis kalimat..."):
            prediction, clean_text, confidence = predict_sentiment(user_input)

        style = LABEL_STYLE.get(
            str(prediction).lower(),
            {"emoji": "🤖", "color": "#607D8B", "label_id": str(prediction)},
        )

        st.markdown("### Hasil Prediksi")
        st.markdown(
            f"<h2 style='color:{style['color']};'>{style['emoji']} {style['label_id']}</h2>",
            unsafe_allow_html=True,
        )

        if confidence:
            st.markdown("### Confidence Score")
            for label, score in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
                st.write(f"{LABEL_STYLE.get(str(label).lower(), {}).get('label_id', label)}")
                st.progress(float(score))
                st.caption(f"{score * 100:.2f}%")

        with st.expander("Lihat teks setelah preprocessing"):
            st.code(clean_text if clean_text else "(kosong setelah preprocessing)")

st.divider()
st.caption(
    "Catatan: confidence score merupakan pendekatan (softmax dari decision_function), "
    "karena model SVM (LinearSVC) yang digunakan tidak menghasilkan probabilitas langsung."
)