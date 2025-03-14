import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

# Judul Aplikasi
st.title("Aplikasi Clustering Tweet COVID-19 dengan K-Means")

# Deskripsi singkat
st.markdown("""
Aplikasi ini menampilkan hasil clustering K-Means terhadap data tweet terkait COVID-19.
Anda dapat menentukan jumlah klaster dan melihat hasil pengelompokan secara interaktif.
""")

@st.cache_data
def load_and_prepare_data():
    df_train = pd.read_csv("corona_train.csv", encoding='latin1')
    df_test = pd.read_csv("corona_test.csv", encoding='latin1')
    df = pd.concat([df_train, df_test], ignore_index=True)
    df['Location'] = df['Location'].fillna("Unknown")
    df = df.drop_duplicates(subset='OriginalTweet')
    df = df[['OriginalTweet']]
    df['OriginalTweet'] = df['OriginalTweet'].astype(str)
    return df

@st.cache_resource
def vectorize_text(texts):
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    vectors = tfidf.fit_transform(texts)
    return tfidf, vectors

@st.cache_resource
def apply_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    model.fit(X)
    return model

# Load & preprocess data
df = load_and_prepare_data()
st.subheader("Preview Dataset")
st.dataframe(df.head())

# TF-IDF Vectorization
tfidf, X = vectorize_text(df['OriginalTweet'])

# Pilih jumlah klaster
st.subheader("Pilih Jumlah Klaster (K)")
k = st.slider("Jumlah Klaster", 2, 6, 3)

# Clustering
model = apply_kmeans(X, k)
df['Cluster'] = model.labels_

# Evaluasi Model
silhouette = silhouette_score(X, model.labels_)
inertia = model.inertia_

st.success(f"Silhouette Score: {silhouette:.4f}")
st.info(f"Inertia (WCSS): {inertia:.2f}")

# WordCloud Tiap Klaster
st.subheader("Wordcloud Tiap Klaster")
for i in range(k):
    cluster_text = " ".join(df[df['Cluster'] == i]['OriginalTweet'])
    if cluster_text.strip():
        wordcloud = WordCloud(width=600, height=300, background_color='white').generate(cluster_text)
        st.markdown(f"### Klaster {i}")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# Tampilkan hasil clustering
st.subheader("Hasil Clustering")
st.dataframe(df[['OriginalTweet', 'Cluster']])

# Fitur interaktif input teks
st.subheader("Coba Masukkan Tweet Anda")
user_input = st.text_area("Masukkan tweet di sini:")
if st.button("Cari Klaster") and user_input:
    input_vec = tfidf.transform([user_input])
    pred_cluster = model.predict(input_vec)[0]
    st.success(f"Tweet Anda termasuk dalam Klaster {pred_cluster}")

# Ekspor hasil
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Unduh Hasil Clustering (CSV)", csv, "hasil_clustering.csv", "text/csv")
