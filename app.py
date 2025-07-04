import streamlit as st
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# === Setup ===
st.set_page_config(page_title="Sistem Penyaringan", layout="wide")
st.title("🤖 Sistem Penyaringan Informasi: Menggunakan Ruang Vektor")
st.markdown("---")

# === Stopwords dan stemmer ===
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
stemmer = StemmerFactory().create_stemmer()

# === URL berita default ===
urls_default = [
    "https://tekno.kompas.com/read/2024/03/22/19000027/ai-kemacetan",
    "https://inet.detik.com/cyberlife/d-6965200",
    "https://www.cnnindonesia.com/teknologi/20240610180015-199-1119509",
    "https://kumparan.com/kumparantech/startup-indonesia-rilis-chatbot-ai-65f80d16ab7d3e0001cb221a",
    "https://tekno.tempo.co/read/1825634",
    "https://health.detik.com/berita-detikhealth/d-7031111",
    "https://nasional.kompas.com/read/2025/01/15/ai-dprd",
    "https://www.antaranews.com/berita/3956121",
    "https://finance.detik.com/berita-ekonomi-bisnis/d-7004421"
]

# === Fungsi preprocessing ===
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords]
    text = ' '.join(words)
    text = stemmer.stem(text)
    return text

@st.cache_data(show_spinner="🔄 Mengambil dan memproses berita...")
def fetch_articles(urls):
    articles = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            content = article.text.strip()
            if len(content) < 100:
                raise ValueError("Isi terlalu pendek")
            content = preprocess(content)
            articles.append(content)
        except:
            articles.append("GAGAL MENGAMBIL ARTIKEL DARI LINK INI")
    return articles

# === Input pengguna ===
query = st.text_input("🔍 Topik pencarian", "pemanfaatan AI di sektor publik dan swasta di Indonesia")
threshold = st.slider("🎯 Threshold Kemiripan", 0.0, 1.0, 0.2, 0.05)
user_links = st.text_area("🔗 Masukkan link berita (pisahkan dengan baris baru)")
user_text = st.text_area("📝 Atau masukkan teks/paragraf secara langsung (opsional)")

# === Jalankan pencarian ===
if st.button("🚀 Jalankan Penyaringan"):
    if user_text.strip():
        # Pecah menjadi paragraf (dua baris kosong atau satu baris kosong)
        paragraf_list = [p.strip() for p in user_text.strip().split('\n\n') if len(p.strip()) > 0]
        documents_clean = [preprocess(p) for p in paragraf_list]
        urls = [f"Paragraf {i+1}" for i in range(len(paragraf_list))]
    else:
        urls = [link.strip() for link in user_links.strip().splitlines() if link.strip()] or urls_default
        documents_clean = fetch_articles(urls)

    actual_relevant = np.array([1] * len(documents_clean))

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents_clean)
    query_clean = preprocess(query)
    query_vector = vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vector, doc_vectors).flatten()
    predicted_relevant = similarity_scores >= threshold

    # === Evaluasi ===
    precision = precision_score(actual_relevant, predicted_relevant, zero_division=0)
    recall = recall_score(actual_relevant, predicted_relevant, zero_division=0)
    f1 = f1_score(actual_relevant, predicted_relevant, zero_division=0)

    st.subheader("📄 Hasil Penyaringan")
    results = []
    for i, source in enumerate(urls):
        score = similarity_scores[i]
        status = "✅ Relevan" if predicted_relevant[i] else "❌ Tidak Relevan"
        alasan = "" if predicted_relevant[i] else "Artikel ini mungkin tidak memiliki kata kunci yang cukup kuat sesuai query."
        st.markdown(f"**{status}** (Score: `{score:.2f}`)  \n🔗 {source}  \n_Alasan_: {alasan}")
        results.append({"Sumber": source, "Score": score, "Status": status, "Alasan": alasan})

    df = pd.DataFrame(results)
    st.download_button("📅 Unduh Hasil sebagai CSV", df.to_csv(index=False), "hasil_penyaringan.csv")

    st.subheader("📊 Evaluasi Sistem")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Precision", f"{precision:.2f}")
        st.metric("📈 Recall", f"{recall:.2f}")
        st.metric("📉 F1-Score", f"{f1:.2f}")
    with col2:
        fig, ax = plt.subplots()
        ax.bar(["Precision", "Recall", "F1-Score"], [precision, recall, f1], color=['#4caf50', '#2196f3', '#ff9800'])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    # === Pie chart ===
    st.subheader("📈 Distribusi Relevansi")
    relevan_count = sum(predicted_relevant)
    tidak_relevan_count = len(predicted_relevant) - relevan_count
    pie_labels = ['Relevan', 'Tidak Relevan']
    pie_sizes = [relevan_count, tidak_relevan_count]
    pie_colors = ['#66bb6a', '#ef5350']

    fig2, ax2 = plt.subplots()
    ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)
    ax2.axis('equal')
    st.pyplot(fig2)

    st.success("✅ Penyaringan dan evaluasi selesai.")
