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

# Upload dua dataset
train_file = st.file_uploader("Unggah Dataset Train (CSV)", type=["csv"], key="train")
test_file = st.file_uploader("Unggah Dataset Test (CSV)", type=["csv"], key="test")

if train_file is not None and test_file is not None:
    # Load dataset
    df_train = pd.read_csv(train_file, encoding='latin1')
    df_test = pd.read_csv(test_file, encoding='latin1')
    df = pd.concat([df_train, df_test], ignore_index=True)

    st.subheader("Preview Dataset Gabungan")
    st.write(df.head())

    # Preprocessing
    st.subheader("Preprocessing Data")
    df['Location'] = df['Location'].fillna("Unknown")
    df = df.drop_duplicates(subset='OriginalTweet')
    df = df[['OriginalTweet']]
    df['OriginalTweet'] = df['OriginalTweet'].astype(str)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    X = tfidf.fit_transform(df['OriginalTweet'])

    # Pilih jumlah klaster
    st.subheader("Pilih Jumlah Klaster (K)")
    k = st.slider("Jumlah Klaster", 2, 10, 3)

    # Clustering K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    df['Cluster'] = kmeans.labels_

    # Evaluasi Model
    silhouette = silhouette_score(X, kmeans.labels_)
    inertia = kmeans.inertia_

    st.success(f"Silhouette Score: {silhouette:.4f}")
    st.info(f"Inertia (WCSS): {inertia:.2f}")

    # Visualisasi WordCloud untuk setiap klaster
    st.subheader("Wordcloud Tiap Klaster")
    for i in range(k):
        cluster_text = " ".join(df[df['Cluster'] == i]['OriginalTweet'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
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
        pred_cluster = kmeans.predict(input_vec)[0]
        st.success(f"Tweet Anda termasuk dalam Klaster {pred_cluster}")

    # Ekspor hasil
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Unduh Hasil Clustering (CSV)", csv, "hasil_clustering.csv", "text/csv")
else:
    st.warning("Silakan unggah kedua file dataset (Train dan Test) terlebih dahulu.")
