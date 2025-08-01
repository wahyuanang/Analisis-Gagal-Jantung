import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Gagal Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi
st.title("🫀 Analisis Prediksi Gagal Jantung")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Home", "Dataset", "Visualisasi", "Prediksi", "Model Performance"]
)

# Fungsi untuk load data
@st.cache_data
def load_data():
    # Jika ada file CSV, uncomment baris berikut dan sesuaikan nama file
    # df = pd.read_csv('heart_failure.csv')
    
    # Sample data untuk demo
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(20, 90, n_samples),
        'anaemia': np.random.choice([0, 1], n_samples),
        'creatinine_phosphokinase': np.random.randint(50, 8000, n_samples),
        'diabetes': np.random.choice([0, 1], n_samples),
        'ejection_fraction': np.random.randint(10, 80, n_samples),
        'high_blood_pressure': np.random.choice([0, 1], n_samples),
        'platelets': np.random.randint(100000, 850000, n_samples),
        'serum_creatinine': np.random.uniform(0.5, 9.5, n_samples),
        'serum_sodium': np.random.randint(110, 150, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'smoking': np.random.choice([0, 1], n_samples),
        'time': np.random.randint(1, 300, n_samples),
        'DEATH_EVENT': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    return df

# Load data
df = load_data()

# Halaman Home
if page == "Home":
    st.header("Selamat Datang di Aplikasi Analisis Gagal Jantung")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Tentang Aplikasi
        Aplikasi ini dibuat untuk menganalisis dan memprediksi risiko gagal jantung berdasarkan berbagai faktor klinis.
        
        **Fitur Utama:**
        - 📊 Visualisasi data interaktif
        - 🤖 Prediksi menggunakan Machine Learning
        - 📈 Analisis performa model
        - 📋 Eksplorasi dataset
        """)
    
    with col2:
        st.markdown("""
        ### Faktor yang Dianalisis
        - **Usia**: Faktor usia pasien
        - **Tekanan Darah**: Hipertensi
        - **Diabetes**: Riwayat diabetes
        - **Anemia**: Kondisi anemia
        - **Merokok**: Kebiasaan merokok
        - **Dan lainnya...**
        """)
    
    # Statistik overview
    st.markdown("### 📊 Overview Dataset")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pasien", len(df))
    with col2:
        st.metric("Kasus Meninggal", df['DEATH_EVENT'].sum())
    with col3:
        st.metric("Tingkat Kematian", f"{(df['DEATH_EVENT'].mean()*100):.1f}%")
    with col4:
        st.metric("Usia Rata-rata", f"{df['age'].mean():.1f} tahun")

# Halaman Dataset
elif page == "Dataset":
    st.header("📋 Eksplorasi Dataset")
    
    # Info dataset
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informasi Dataset")
        st.write(f"Jumlah baris: {df.shape[0]}")
        st.write(f"Jumlah kolom: {df.shape[1]}")
        
    with col2:
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("Tidak ada missing values!")
        else:
            st.write(missing[missing > 0])
    
    # Tampilkan data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

# Halaman Visualisasi
elif page == "Visualisasi":
    st.header("📊 Visualisasi Data")
    
    # Distribusi target
    fig1 = px.pie(df, names='DEATH_EVENT', title='Distribusi Kematian',
                  labels={'DEATH_EVENT': 'Status', 0: 'Hidup', 1: 'Meninggal'})
    st.plotly_chart(fig1, use_container_width=True)
    
    # Distribusi usia
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = px.histogram(df, x='age', color='DEATH_EVENT', 
                           title='Distribusi Usia berdasarkan Status')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = px.box(df, x='DEATH_EVENT', y='ejection_fraction',
                      title='Ejection Fraction vs Status')
        st.plotly_chart(fig3, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    fig4, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig4)

# Halaman Prediksi
elif page == "Prediksi":
    st.header("🤖 Prediksi Gagal Jantung")
    
    st.markdown("### Input Data Pasien")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Usia", 20, 90, 50)
        anaemia = st.selectbox("Anemia", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        high_bp = st.selectbox("Tekanan Darah Tinggi", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        sex = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Laki-laki" if x else "Perempuan")
        smoking = st.selectbox("Merokok", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
    
    with col2:
        cpk = st.number_input("Creatinine Phosphokinase", 50, 8000, 500)
        ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 50)
        platelets = st.number_input("Platelets", 100000, 850000, 250000)
        serum_creatinine = st.number_input("Serum Creatinine", 0.5, 9.5, 1.0)
        serum_sodium = st.slider("Serum Sodium", 110, 150, 135)
        time = st.slider("Follow-up Period (days)", 1, 300, 150)
    
    # Prediksi
    if st.button("🔮 Prediksi"):
        # Prepare data
        X = df.drop('DEATH_EVENT', axis=1)
        y = df['DEATH_EVENT']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi input user
        user_input = np.array([[age, anaemia, cpk, diabetes, ejection_fraction, 
                               high_bp, platelets, serum_creatinine, serum_sodium, 
                               sex, smoking, time]])
        
        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)[0]
        
        # Tampilkan hasil
        st.markdown("### 🎯 Hasil Prediksi")
        
        if prediction == 1:
            st.error(f"⚠️ **Risiko Tinggi** - Probabilitas kematian: {probability[1]:.2%}")
        else:
            st.success(f"✅ **Risiko Rendah** - Probabilitas bertahan hidup: {probability[0]:.2%}")
        
        # Progress bar untuk visualisasi probabilitas
        st.progress(probability[1])

# Halaman Model Performance
elif page == "Model Performance":
    st.header("📈 Performa Model")
    
    # Prepare data
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    
    # Tampilkan hasil
    st.subheader("Akurasi Model")
    for model_name, accuracy in results.items():
        st.metric(f"{model_name}", f"{accuracy:.3f}")
    
    # Feature importance (Random Forest)
    st.subheader("Feature Importance")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance.head(10), x='importance', y='feature',
                 title='Top 10 Feature Importance', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💡 **Catatan**: Aplikasi ini untuk tujuan edukasi. Konsultasikan dengan dokter untuk diagnosis medis yang akurat.")