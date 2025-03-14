import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

# Judul
st.title("Aplikasi Clustering Tweet COVID-19 dengan K-Means")

st.markdown("""
Aplikasi ini menampilkan hasil clustering K-Means terhadap data tweet COVID-19.
Anda dapat memilih jumlah klaster dan melihat hasilnya secara interaktif.
""")

# Fungsi cache
@st.cache_data
def load_data():
    df_train = pd.read_csv("corona_train.csv", encoding='latin1')
    df_test = pd.read_csv("corona_test.csv", encoding='latin1')
    df = pd.concat([df_train, df_test], ignore_index=True)
    df['Location'] = df['Location'].fillna("Unknown")
    df = df.drop_duplicates(subset='OriginalTweet')
    df = df[['OriginalTweet']]
    df['OriginalTweet'] = df['OriginalTweet'].astype(str)
    return df

df = load_data()

st.subheader("Preview Data")
st.dataframe(df.head())

@st.cache_resource
def build_vectorizer():
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(df['OriginalTweet'])
    return vectorizer, X

vectorizer, X = build_vectorizer()

# Slider klaster
st.subheader("Pilih Jumlah Klaster")
k = st.slider("Jumlah Klaster", 2, 6, 3)

@st.cache_resource
def train_kmeans(k_value):
    model = KMeans(n_clusters=k_value, random_state=42, n_init='auto')
    model.fit(X)
    return model

model = train_kmeans(k)

df['Cluster'] = model.labels_

# Evaluasi
silhouette = silhouette_score(X, model.labels_)
inertia = model.inertia_

st.success(f"Silhouette Score: {silhouette:.4f}")
st.info(f"Inertia (WCSS): {inertia:.2f}")

# Wordcloud
st.subheader("WordCloud Tiap Klaster")
for i in range(k):
    text = " ".join(df[df['Cluster'] == i]['OriginalTweet'])
    if text.strip():
        wc = WordCloud(width=600, height=300, background_color='white').generate(text)
        st.markdown(f"### Klaster {i}")
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# Tabel hasil
st.subheader("Hasil Clustering")
st.dataframe(df[['OriginalTweet', 'Cluster']])

# Input tweet interaktif
st.subheader("Coba Masukkan Tweet Anda")
user_input = st.text_area("Masukkan tweet:")
if st.button("Cari Klaster") and user_input:
    input_vec = vectorizer.transform([user_input])
    pred_cluster = model.predict(input_vec)[0]
    st.success(f"Tweet Anda termasuk dalam Klaster {pred_cluster}")

# Unduh hasil
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Unduh Hasil Clustering (CSV)", csv, "hasil_clustering.csv", "text/csv")
