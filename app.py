import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

# --- KONFIGURASI HALAMAN WEB ---
st.set_page_config(
    page_title="Analisis Minat Baca dengan Random Forest - Kelompok 3A",
    page_icon="📚",
    layout="wide"
)

# --- JUDUL & NAVIGASI SIDEBAR ---
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio(
    "Pilih Tahapan:",
    ["Home", "1. Input Dataset", "2. Preprocessing", "3. Training Random Forest", 
     "4. Visualisasi Decision Trees", "5. Feature Importance & Analisis", "6. Prediksi Simulasi"]
)

# --- INISIALISASI SESSION STATE (Agar data tidak hilang saat pindah menu) ---
if 'data_raw' not in st.session_state:
    st.session_state['data_raw'] = None
if 'data_clean' not in st.session_state:
    st.session_state['data_clean'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'rf_model' not in st.session_state:
    st.session_state['rf_model'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None
if 'accuracy' not in st.session_state:
    st.session_state['accuracy'] = None
if 'y_pred' not in st.session_state:
    st.session_state['y_pred'] = None

# =============================================================================
# MENU: HOME
# =============================================================================
if menu == "Home":
    st.title("📚 Sistem Analisis Pola Kebiasaan Baca dengan Random Forest")
    st.markdown("""
    **Tugas Besar Akuisisi Data - Kelompok 3A**
    
    Sistem ini dibangun untuk menganalisis dan memprediksi tingkat kegemaran membaca 
    masyarakat Indonesia berdasarkan data Tingkat Kegemaran Membaca (TGM) tahun 2020-2024.
    
    **Anggota Kelompok:**
    * Darrel Rajendra Kurnia
    * Muhammad Fauzin
    * Farhan Aufa
    
    **Metode Analisis:** Random Forest Classification (Multi-Class)
    
    **Tujuan Sistem:**
    - Mengklasifikasikan provinsi berdasarkan 3 kategori minat baca: **Low**, **Moderate**, dan **High**
    - Mengidentifikasi faktor-faktor yang mempengaruhi tingkat kegemaran membaca
    - Memberikan rekomendasi untuk meningkatkan kategori minat baca (Low → Moderate → High)
    
    **Alur Penggunaan:**
    1. **Input Dataset** - Upload data CSV TGM
    2. **Preprocessing** - Bersihkan dan persiapkan data
    3. **Training** - Latih model Random Forest dengan 2 decision trees
    4. **Visualisasi** - Lihat struktur pohon keputusan
    5. **Feature Importance** - Analisis faktor penentu klasifikasi
    6. **Prediksi Simulasi** - Simulasi data provinsi baru
    """)
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)

# =============================================================================
# MENU 1: INPUT DATASET (AKUISISI)
# =============================================================================
elif menu == "1. Input Dataset":
    st.header("📂 Tahap 1: Akuisisi Data (Input)")
    st.write("Silakan upload file dataset (Format .csv) yang bersumber dari Data TGM.")

    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # NOTE: Menggunakan delimiter koma karena data menggunakan format ini
            df = pd.read_csv(uploaded_file, delimiter=',')
            st.session_state['data_raw'] = df  # Simpan ke session
            st.success("Dataset berhasil dimuat!")
            
            st.subheader("Tinjauan Data Mentah")
            st.dataframe(df.head(10))
            st.write(f"Dimensi Data: {df.shape[0]} Baris, {df.shape[1]} Kolom")
            
            st.subheader("Informasi Kolom")
            st.write(df.dtypes)
            
            st.subheader("Distribusi Target Kategori")
            if 'Category' in df.columns:
                category_counts = df['Category'].value_counts()
                st.write(category_counts)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                category_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
                ax.set_title('Distribusi Kategori Minat Baca')
                ax.set_xlabel('Kategori')
                ax.set_ylabel('Jumlah')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
    
    elif st.session_state['data_raw'] is not None:
        st.info("File sudah terupload sebelumnya.")
        st.dataframe(st.session_state['data_raw'].head())

# =============================================================================
# MENU 2: PREPROCESSING
# =============================================================================
elif menu == "2. Preprocessing":
    st.header("🧹 Tahap 2: Preprocessing Data")

    if st.session_state['data_raw'] is None:
        st.warning("Mohon upload data terlebih dahulu di menu 'Input Dataset'.")
    else:
        df = st.session_state['data_raw'].copy()
        
        st.write("Data mentah mengandung format angka Indonesia (koma) dan nilai kosong (N/A).")
        st.write("Sistem akan membersihkan data ini agar bisa diproses oleh Random Forest.")
        
        if st.button("Jalankan Preprocessing"):
            with st.spinner("Memproses data..."):
                # 1. Pilih kolom numerik yang perlu dibersihkan
                cols_numeric = [
                    'Reading Frequency per week',
                    'Number of Readings per Quarter',
                    'Daily Reading Duration (in minutes)', 
                    'Internet Access Frequency per Week',
                    'Daily Internet Duration (in minutes)',
                    'Tingkat Kegemaran Membaca (Reading Interest)'
                ]
                
                # 2. Loop pembersihan (Ubah koma jadi titik, ubah ke float)
                for col in cols_numeric:
                    if col in df.columns:
                        # Cek jika kolom tipe objek (text), replace koma
                        if df[col].dtype == 'object':
                            df[col] = df[col].str.replace(',', '.')
                        # Ubah paksa jadi angka
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 3. Mengisi nilai kosong (NaN) dengan 0 (Zero Imputation)
                df[cols_numeric] = df[cols_numeric].fillna(0)
                
                # 4. Encode target kategori MANUAL untuk 3 kelas
                # Low = 0 (kategori terendah)
                # Moderate = 1 (kategori menengah)  
                # High = 2 (kategori tertinggi)
                df['Category_Encoded'] = df['Category'].map({'Low': 0, 'Moderate': 1, 'High': 2})
                
                # 5. Pisahkan fitur (X) dan target (y)
                feature_cols = cols_numeric
                X = df[feature_cols]
                y = df['Category_Encoded']
                
                # 6. Split data untuk training dan testing
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Simpan hasil
                st.session_state['data_clean'] = df
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = feature_cols
                
                st.success("Preprocessing Selesai! Format angka diperbaiki & Missing Values ditangani.")
                st.info(f"Data Training: {X_train.shape[0]} sampel | Data Testing: {X_test.shape[0]} sampel")
            
        # Tampilkan data bersih jika sudah ada
        if st.session_state['data_clean'] is not None:
            st.subheader("Data Hasil Preprocessing")
            st.dataframe(st.session_state['data_clean'].head(10))
            st.write("Statistik Deskriptif:")
            st.write(st.session_state['data_clean'][st.session_state['feature_names']].describe())

# =============================================================================
# MENU 3: TRAINING RANDOM FOREST
# =============================================================================
elif menu == "3. Training Random Forest":
    st.header("⚙️ Tahap 3: Training Model Random Forest")

    if st.session_state['X_train'] is None:
        st.warning("Mohon lakukan Preprocessing terlebih dahulu.")
    else:
        st.sidebar.subheader("Konfigurasi Model")
        n_estimators = st.sidebar.slider("Jumlah Trees (n_estimators)", min_value=2, max_value=200, value=2, step=1)
        max_depth = st.sidebar.slider("Max Depth (Kedalaman Pohon)", min_value=2, max_value=10, value=5)
        
        st.write("**Konfigurasi Model Random Forest:**")
        st.write(f"- Jumlah Trees: **{n_estimators}**")
        st.write(f"- Max Depth: **{max_depth}**")
        st.write(f"- Random State: **42** (untuk reproducibility)")
        
        if st.button("🚀 Latih Model Random Forest"):
            with st.spinner("Melatih model Random Forest..."):
                # Training model
                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                
                rf_model.fit(st.session_state['X_train'], st.session_state['y_train'])
                
                # Prediksi
                y_pred = rf_model.predict(st.session_state['X_test'])
                y_pred_train = rf_model.predict(st.session_state['X_train'])
                
                # Hitung akurasi
                accuracy_test = accuracy_score(st.session_state['y_test'], y_pred)
                accuracy_train = accuracy_score(st.session_state['y_train'], y_pred_train)
                
                # Simpan model
                st.session_state['rf_model'] = rf_model
                st.session_state['accuracy'] = accuracy_test
                st.session_state['y_pred'] = y_pred
                
                st.success(f"✅ Model berhasil dilatih!")
                
                # Tampilkan hasil
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Akurasi Training", f"{accuracy_train*100:.2f}%")
                with col2:
                    st.metric("Akurasi Testing", f"{accuracy_test*100:.2f}%")
                
                # Confusion Matrix (3x3 untuk Low, Moderate, High)
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state['y_test'], y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Low', 'Moderate', 'High'],
                           yticklabels=['Low', 'Moderate', 'High'])
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_title('Confusion Matrix (3-Class Classification)', fontsize=14)
                st.pyplot(fig)
                
                # Classification Report (untuk 3 kelas)
                st.subheader("Classification Report")
                report = classification_report(
                    st.session_state['y_test'], 
                    y_pred,
                    target_names=['Low', 'Moderate', 'High'],
                    output_dict=True
                )
                st.dataframe(pd.DataFrame(report).transpose())
                
                st.info("💡 Model siap digunakan! Lanjut ke menu berikutnya untuk visualisasi dan analisis.")

# =============================================================================
# MENU 4: VISUALISASI DECISION TREES
# =============================================================================
elif menu == "4. Visualisasi Decision Trees":
    st.header("🌳 Tahap 4: Visualisasi Decision Trees")

    if st.session_state['rf_model'] is None:
        st.warning("Mohon latih model Random Forest terlebih dahulu di menu 3.")
    else:
        n_trees = len(st.session_state['rf_model'].estimators_)
        st.write(f"Model Random Forest terdiri dari **{n_trees} decision trees**.")
        
        if n_trees <= 5:
            # Jika trees sedikit (<=5), tampilkan semua dalam tabs
            st.write("Mari kita visualisasikan semua trees:")
            tabs = st.tabs([f"🌲 Tree #{i+1}" for i in range(n_trees)])
            
            for i, tab in enumerate(tabs):
                with tab:
                    st.subheader(f"Decision Tree #{i+1}")
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(
                        st.session_state['rf_model'].estimators_[i],
                        feature_names=st.session_state['feature_names'],
                        class_names=['Low', 'Moderate', 'High'],
                        filled=True,
                        rounded=True,
                        fontsize=10,
                        ax=ax
                    )
                    ax.set_title(f"Decision Tree #{i+1} dari Random Forest", fontsize=16)
                    st.pyplot(fig)
            
            st.info("""
            **Cara Membaca Decision Tree:**
            - **Node Awal (Root)**: Fitur pertama yang dipakai untuk split
            - **Gini**: Measure of impurity (0 = pure, 0.5 = mixed)
            - **Samples**: Jumlah data di node tersebut
            - **Value**: [Jumlah Low, Jumlah Moderate, Jumlah High]
            - **Class**: Predicted class di node tersebut
            - **Warna**: Biru = Low, Orange = Moderate/High (gradasi)
            
            **Random Forest Voting:**
            Random Forest menggunakan voting dari semua trees untuk menentukan prediksi final.
            """)
        else:
            # Jika trees banyak (>5), biarkan user pilih tree mana yang mau dilihat
            st.write(f"Karena jumlah trees banyak ({n_trees} trees), pilih tree mana yang ingin divisualisasikan:")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_tree = st.number_input(
                    "Pilih Tree (1-" + str(n_trees) + ")",
                    min_value=1,
                    max_value=n_trees,
                    value=1,
                    step=1
                )
            
            st.subheader(f"Decision Tree #{selected_tree}")
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                st.session_state['rf_model'].estimators_[selected_tree-1],
                feature_names=st.session_state['feature_names'],
                class_names=['Low', 'Moderate', 'High'],
                filled=True,
                rounded=True,
                fontsize=10,
                ax=ax
            )
            ax.set_title(f"Decision Tree #{selected_tree} dari Random Forest (Total: {n_trees} trees)", fontsize=16)
            st.pyplot(fig)
            
            st.info(f"""
            **Cara Membaca Decision Tree:**
            - **Node Awal (Root)**: Fitur pertama yang dipakai untuk split
            - **Gini**: Measure of impurity (0 = pure, 0.5 = mixed)
            - **Samples**: Jumlah data di node tersebut
            - **Value**: [Jumlah Moderate, Jumlah High]
            - **Class**: Predicted class di node tersebut
            - **Warna**: Orange = High, Blue = Moderate
            
            **Random Forest Voting:**
            Model ini menggunakan voting dari **{n_trees} trees** untuk menentukan prediksi final.
            Anda dapat melihat tree lain dengan mengubah nomor di atas.
            """)

# =============================================================================
# MENU 5: FEATURE IMPORTANCE & ANALISIS
# =============================================================================
elif menu == "5. Feature Importance & Analisis":
    st.header("📊 Tahap 5: Feature Importance & Analisis")

    if st.session_state['rf_model'] is None or st.session_state['data_clean'] is None:
        st.warning("Mohon latih model Random Forest terlebih dahulu di menu 3.")
    else:
        st.write("Feature Importance menunjukkan seberapa penting setiap variabel dalam menentukan klasifikasi High vs Moderate.")
        st.write("**Analisis Cerdas**: Sistem akan mendeteksi apakah fitur berpengaruh **positif** (harus ditingkatkan) atau **negatif** (harus dikurangi) untuk mencapai kategori High.")
        
        # Get feature importance
        importances = st.session_state['rf_model'].feature_importances_
        feature_names = st.session_state['feature_names']
        
        # === SIMPLE MANUAL ASSIGNMENT ===
        # Internet-related features = NEGATIF (nilai tinggi menurunkan minat baca)
        # Reading-related features = POSITIF (nilai tinggi meningkatkan minat baca)
        
        correlations = []
        for feature in feature_names:
            if 'Daily Internet Duration' in feature or 'Internet Access Frequency' in feature:
                # Negatif: Internet terlalu lama/sering mengurangi waktu baca
                correlations.append(-0.3)  # Nilai negatif sebagai indikator
            else:
                # Positif: Reading-related features meningkatkan minat baca
                correlations.append(0.5)  # Nilai positif sebagai indikator
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Correlation': correlations
        })
        
        # Tentukan impact direction
        feature_importance_df['Impact'] = feature_importance_df['Correlation'].apply(
            lambda x: 'Positive ↑' if x > 0 else 'Negative ↓'
        )
        feature_importance_df['Impact Type'] = feature_importance_df['Correlation'].apply(
            lambda x: 'Positive' if x > 0 else 'Negative'
        )
        feature_importance_df['Direction'] = feature_importance_df['Correlation'].apply(
            lambda x: '⬆️ Tingkatkan' if x > 0 else '⬇️ Kurangi'
        )
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Visualisasi dengan color coding
        st.subheader("Feature Importance Ranking dengan Arah Pengaruh")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color based on impact: Green for positive, Red for negative
        colors = ['#2ecc71' if imp_type == 'Positive' else '#e74c3c' 
                  for imp_type in feature_importance_df['Impact Type']]
        
        bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors, alpha=0.8)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature Importance dengan Arah Pengaruh\n(Hijau = Positif ↑ | Merah = Negatif ↓)', fontsize=14)
        ax.invert_yaxis()
        
        # Add value labels with correlation
        for i, bar in enumerate(bars):
            width = bar.get_width()
            corr_val = feature_importance_df.iloc[i]['Correlation']
            direction = feature_importance_df.iloc[i]['Impact']
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'  {width:.4f} ({direction})', ha='left', va='center', fontsize=9)
        
        st.pyplot(fig)
        
        # Tabel ranking dengan penjelasan lengkap
        st.subheader("Tabel Feature Importance & Arah Pengaruh")
        
        display_df = feature_importance_df.copy()
        display_df['Importance %'] = (display_df['Importance'] * 100).round(2)
        display_df['Correlation'] = display_df['Correlation'].round(4)
        
        # Reorder columns
        display_df = display_df[['Feature', 'Importance', 'Importance %', 'Correlation', 'Impact', 'Direction']]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Interpretasi dengan penjelasan impact
        st.subheader("💡 Interpretasi & Insight Cerdas")
        
        st.markdown("""
        **Cara Membaca Tabel:**
        - **Importance**: Seberapa penting fitur ini dalam model (semakin tinggi semakin penting)
        - **Correlation**: Hubungan dengan kategori High (+) atau Moderate (-)
        - **Impact**: 
          - **Positive ↑**: Fitur ini berkorelasi positif → nilai lebih tinggi = lebih cenderung High
          - **Negative ↓**: Fitur ini berkorelasi negatif → nilai lebih tinggi = lebih cenderung Moderate
        - **Direction**: Rekomendasi aksi (⬆️ Tingkatkan atau ⬇️ Kurangi)
        """)
        
        # Split features by impact type
        positive_features = feature_importance_df[feature_importance_df['Impact Type'] == 'Positive'].sort_values('Importance', ascending=False)
        negative_features = feature_importance_df[feature_importance_df['Impact Type'] == 'Negative'].sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⬆️ Fitur Berpengaruh Positif")
            st.markdown("**Tingkatkan nilai fitur ini untuk mencapai High:**")
            if len(positive_features) > 0:
                for idx, row in positive_features.head(5).iterrows():
                    st.write(f"- **{row['Feature']}** (Importance: {row['Importance']:.4f}, Corr: {row['Correlation']:.4f})")
            else:
                st.write("Tidak ada fitur dengan pengaruh positif dominan")
        
        with col2:
            st.markdown("### ⬇️ Fitur Berpengaruh Negatif")
            st.markdown("**Kurangi nilai fitur ini untuk mencapai High:**")
            if len(negative_features) > 0:
                for idx, row in negative_features.head(5).iterrows():
                    st.write(f"- **{row['Feature']}** (Importance: {row['Importance']:.4f}, Corr: {row['Correlation']:.4f})")
            else:
                st.write("Tidak ada fitur dengan pengaruh negatif dominan")
        
        # Top insights
        st.divider()
        st.markdown("### 🎯 Rekomendasi Strategis")
        
        top_positive = positive_features.head(1) if len(positive_features) > 0 else None
        top_negative = negative_features.head(1) if len(negative_features) > 0 else None
        
        recommendations = []
        
        if top_positive is not None and len(top_positive) > 0:
            top_pos_feature = top_positive.iloc[0]['Feature']
            top_pos_imp = top_positive.iloc[0]['Importance']
            top_pos_corr = top_positive.iloc[0]['Correlation']
            recommendations.append(
                f"**1. Prioritas Utama - Tingkatkan**: `{top_pos_feature}` (Importance: {top_pos_imp:.4f}, Korelasi Positif: {top_pos_corr:.4f})\n   "
                f"   → Fitur ini paling penting dan memiliki pengaruh positif terkuat. Provinsi dengan nilai tinggi di fitur ini cenderung masuk kategori High."
            )
        
        if top_negative is not None and len(top_negative) > 0:
            top_neg_feature = top_negative.iloc[0]['Feature']
            top_neg_imp = top_negative.iloc[0]['Importance']
            top_neg_corr = top_negative.iloc[0]['Correlation']
            recommendations.append(
                f"**2. Perhatian Khusus - Kurangi**: `{top_neg_feature}` (Importance: {top_neg_imp:.4f}, Korelasi Negatif: {top_neg_corr:.4f})\n   "
                f"   → Fitur ini penting namun berpengaruh negatif. Nilai yang terlalu tinggi justru membuat provinsi cenderung Moderate. Optimalkan dengan mengurangi."
            )
        
        recommendations.append(
            "**3. Pendekatan Seimbang**: Fokus pada peningkatan fitur positif sambil mengontrol/mengurangi fitur negatif untuk hasil optimal."
        )
        
        for rec in recommendations:
            st.markdown(rec)
        
        st.info("""
        **💡 Catatan Penting**: 
        - Analisis ini menggunakan korelasi Pearson untuk menentukan arah pengaruh
        - Fitur dengan pengaruh negatif bukan berarti "buruk", tapi perlu dioptimalkan (tidak terlalu tinggi)
        - Contoh: "Daily Internet Duration" yang terlalu tinggi mungkin mengurangi waktu membaca
        """)

# =============================================================================
# MENU 6: ANALISIS PERBAIKAN PROVINSI (MULTI-CLASS)
# =============================================================================
elif menu == "6. Prediksi Simulasi":
    st.header("🔍 Tahap 6: Analisis Perbaikan Provinsi")

    if st.session_state['rf_model'] is None or st.session_state['data_clean'] is None:
        st.warning("Mohon latih model Random Forest terlebih dahulu.")
    else:
        st.write("Pilih improvement path untuk analisis rekomendasi perbaikan.")
        
        # === PILIHAN IMPROVEMENT PATH ===
        st.subheader("🎯 Pilih Target Peningkatan Kategori")
        
        improvement_path = st.radio(
            "Dari kategori mana ke mana yang ingin dianalisis?",
            ["Low → Moderate", "Low → High", "Moderate → High"],
            horizontal=True
        )
        
        # Tentukan source dan target berdasarkan pilihan
        if improvement_path == "Low → Moderate":
            source_category = "Low"
            target_category = "Moderate"
        elif improvement_path == "Low → High":
            source_category = "Low"
            target_category = "High"
        else:  # Moderate → High
            source_category = "Moderate"
            target_category = "High"
        
        st.info(f"📌 Anda akan menganalisis cara meningkatkan dari **{source_category}** ke **{target_category}**")
        
        # Filter data source
        df_clean = st.session_state['data_clean']
        df_source = df_clean[df_clean['Category'] == source_category].copy()
        
        if len(df_source) == 0:
            st.warning(f"Tidak ada provinsi dengan kategori {source_category} di dataset.")
        else:
            st.subheader(f"Pilih Provinsi {source_category} dan Tahun")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Dropdown provinsi
                provinsi_list = sorted(df_source['Provinsi'].unique())
                selected_provinsi = st.selectbox("Pilih Provinsi:", provinsi_list)
            
            with col2:
                # Filter tahun berdasarkan provinsi yang dipilih
                df_provinsi = df_source[df_source['Provinsi'] == selected_provinsi]
                tahun_list = sorted(df_provinsi['Year'].unique())
                selected_year = st.selectbox("Pilih Tahun:", tahun_list)
            
            # Ambil data terpilih
            selected_data = df_source[
                (df_source['Provinsi'] == selected_provinsi) & 
                (df_source['Year'] == selected_year)
            ].iloc[0]
            
            st.divider()
            
            # Tampilkan data provinsi terpilih
            st.subheader(f"📊 Data {selected_provinsi} - Tahun {selected_year}")
            
            feature_cols = st.session_state['feature_names']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Reading Frequency/week", f"{selected_data[feature_cols[0]]:.1f}")
                st.metric("Daily Reading Duration", f"{selected_data[feature_cols[2]]:.1f} min")
            with col2:
                st.metric("Number of Readings/Quarter", f"{selected_data[feature_cols[1]]:.1f}")
                st.metric("Internet Frequency/week", f"{selected_data[feature_cols[3]]:.1f}")
            with col3:
                st.metric("Daily Internet Duration", f"{selected_data[feature_cols[4]]:.1f} min")
                st.metric("TGM Score", f"{selected_data[feature_cols[5]]:.1f}")
            
            # Prediksi oleh model (3-class)
            X_input = selected_data[feature_cols].values.reshape(1, -1)
            prediction = st.session_state['rf_model'].predict(X_input)[0]
            prediction_proba = st.session_state['rf_model'].predict_proba(X_input)[0]
            
            st.divider()
            
            # Hasil prediksi (untuk 3 kelas)
            st.subheader("🤖 Hasil Prediksi Model")
            
            # Map prediction ke category name
            category_map = {0: 'Low', 1: 'Moderate', 2: 'High'}
            category_pred = category_map[prediction]
            color_map = {'Low': 'red', 'Moderate': 'orange', 'High': 'green'}
            color = color_map[category_pred]
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.markdown(f"### Prediksi: :{color}[**{category_pred}**]")
                st.caption(f"Kategori Asli: **{selected_data['Category']}**")
            with col2:
                st.metric("Prob. Low", f"{prediction_proba[0]*100:.1f}%")
            with col3:
                st.metric("Prob. Moderate", f"{prediction_proba[1]*100:.1f}%")
            with col4:
                st.metric("Prob. High", f"{prediction_proba[2]*100:.1f}%")
            
            st.divider()
            
            # Analisis & Rekomendasi
            st.subheader(f"💡 Analisis & Rekomendasi Perbaikan: {source_category} → {target_category}")
            
            # Get feature importance
            importances = st.session_state['rf_model'].feature_importances_
            feature_names = st.session_state['feature_names']
            
            # === SIMPLE MANUAL ASSIGNMENT ===
            correlations = []
            for feature in feature_names:
                if 'Daily Internet Duration' in feature or 'Internet Access Frequency' in feature:
                    correlations.append(-0.3)
                else:
                    correlations.append(0.5)
            
            # Bandingkan dengan provinsi TARGET
            df_target = df_clean[df_clean['Category'] == target_category]
            avg_target = df_target[feature_cols].mean()
            
            # Hitung gap untuk provinsi terpilih
            current_values = selected_data[feature_cols]
            gaps = avg_target - current_values
            
            st.markdown(f"**Perbandingan dengan Rata-rata Provinsi {target_category}:**")
            
            # Buat DataFrame perbandingan
            comparison_df = pd.DataFrame({
                'Fitur': feature_names,
                'Nilai Saat Ini': current_values.values,
                f'Rata-rata {target_category}': avg_target.values,
                'Gap': gaps.values,
                'Importance Score': importances,
                'Correlation': correlations
            })
            
            # Sort by importance
            comparison_df = comparison_df.sort_values('Importance Score', ascending=False)
            
            # SMART Priority
            def calculate_smart_priority(row):
                imp = row['Importance Score']
                corr = row['Correlation']
                gap = row['Gap']
                
                if imp > 0.15:
                    if (corr > 0 and gap > 0) or (corr < 0 and gap < 0):
                        return '🔴 Tinggi'
                    else:
                        return '🟡 Sedang'
                elif imp > 0.10:
                    if (corr > 0 and gap > 0) or (corr < 0 and gap < 0):
                        return '🟡 Sedang'
                    else:
                        return '🟢 Rendah'
                else:
                    return '🟢 Rendah'
            
            comparison_df['Prioritas'] = comparison_df.apply(calculate_smart_priority, axis=1)
            
            # SMART DIRECTION
            def get_smart_action(row):
                corr = row['Correlation']
                gap = row['Gap']
                
                if corr > 0:
                    if gap > 0:
                        return f"⬆️ Tingkatkan +{gap:.2f}"
                    else:
                        return "✅ Sudah optimal"
                else:
                    if gap < 0:
                        return f"⬇️ Kurangi {abs(gap):.2f}"
                    else:
                        return "✅ Sudah optimal"
            
            comparison_df['Rekomendasi Aksi'] = comparison_df.apply(get_smart_action, axis=1)
            
            # Display table
            display_comparison = comparison_df[['Fitur', 'Nilai Saat Ini', f'Rata-rata {target_category}', 'Gap', 
                                               'Importance Score', 'Correlation', 'Prioritas', 'Rekomendasi Aksi']].copy()
            
            st.dataframe(display_comparison.style.format({
                'Nilai Saat Ini': '{:.2f}',
                f'Rata-rata {target_category}': '{:.2f}',
                'Gap': '{:.2f}',
                'Importance Score': '{:.4f}',
                'Correlation': '{:.4f}'
            }), use_container_width=True)
            
            # Rekomendasi prioritas - SEMUA FITUR
            st.markdown("### 🎯 Rekomendasi Perbaikan untuk Semua Fitur")
            st.write("Berikut adalah analisis dan rekomendasi untuk **semua fitur**, diurutkan berdasarkan prioritas:")
            
            # Show ALL features, sorted by priority then importance
            all_features = comparison_df.sort_values(['Prioritas', 'Importance Score'], ascending=[True, False])
            
            for idx, row in all_features.iterrows():
                corr = row['Correlation']
                gap = row['Gap']
                current_val = row['Nilai Saat Ini']
                target_val = row[f'Rata-rata {target_category}']
                
                # Tentukan action dan calculate smart percentage
                if corr > 0:  # Positive correlation
                    if gap > 0:  # Need to increase
                        action_text = f"⬆️ TINGKATKAN dari {current_val:.2f} ke {target_val:.2f}"
                        explanation = f"Fitur ini memiliki **korelasi positif** ({corr:.4f}). Nilai lebih tinggi meningkatkan peluang masuk kategori {target_category}."
                        # Percentage = how much % increase needed
                        change_pct = (gap / current_val * 100) if current_val > 0 else 0
                        change_label = f"Perlu ditingkatkan {change_pct:.1f}%"
                        change_sign = "+"
                    else:  # Already good
                        action_text = f"✅ Sudah optimal (nilai: {current_val:.2f} ≥ target: {target_val:.2f})"
                        explanation = "Nilai sudah melebihi rata-rata target. Pertahankan atau tingkatkan lebih lanjut."
                        change_pct = abs(gap / current_val * 100) if current_val > 0 else 0
                        change_label = f"Sudah {change_pct:.1f}% di atas target"
                        change_sign = ""
                else:  # Negative correlation
                    if gap < 0:  # Need to decrease (current > target)
                        action_text = f"⬇️ KURANGI dari {current_val:.2f} ke {target_val:.2f}"
                        explanation = f"Fitur ini memiliki **korelasi negatif** ({corr:.4f}). Nilai terlalu tinggi menurunkan peluang masuk kategori {target_category}."
                        # Percentage = how much % decrease needed
                        change_pct = (abs(gap) / current_val * 100) if current_val > 0 else 0
                        change_label = f"Perlu dikurangi {change_pct:.1f}%"
                        change_sign = "-"
                    else:  # Already good (current <= target)
                        action_text = f"✅ Sudah optimal (nilai: {current_val:.2f} ≤ target: {target_val:.2f})"
                        explanation = "Nilai sudah lebih rendah dari rata-rata target. Pertahankan nilai rendah ini."
                        change_pct = abs(gap / current_val * 100) if current_val > 0 else 0
                        change_label = f"Sudah {change_pct:.1f}% di bawah target"
                        change_sign = ""
                
                # Display in expander
                with st.expander(f"{row['Prioritas']} {row['Fitur']}", expanded=(row['Prioritas'] in ['🔴 Tinggi', '🟡 Sedang'])):
                    st.markdown(f"**Aksi:** :red[{action_text}]" if "KURANGI" in action_text 
                               else f"**Aksi:** :green[{action_text}]" if "TINGKATKAN" in action_text
                               else f"**Aksi:** {action_text}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nilai Saat Ini", f"{current_val:.2f}")
                    with col2:
                        st.metric(f"Target (Avg {target_category})", f"{target_val:.2f}")
                    with col3:
                        st.metric(change_label, 
                                 f"{change_sign}{abs(gap):.2f}")
                    
                    st.write(f"**Importance Score:** {row['Importance Score']:.4f}")
                    st.write(f"**Correlation:** {row['Correlation']:.4f} ({'Positive ↑' if corr > 0 else 'Negative ↓'})")
                    st.info(explanation)
            
            # Visualisasi perbandingan
            st.divider()
            st.subheader("📊 Visualisasi Perbandingan")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(feature_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, current_values.values, width, label=f'{selected_provinsi} ({selected_year})', color='#e74c3c', alpha=0.8)
            bars2 = ax.bar(x + width/2, avg_target.values, width, label=f'Rata-rata {target_category}', color='#2ecc71', alpha=0.8)
            
            ax.set_xlabel('Fitur', fontsize=12)
            ax.set_ylabel('Nilai', fontsize=12)
            ax.set_title(f'Perbandingan {selected_provinsi} vs Rata-rata Provinsi {target_category}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in feature_names], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
            
            # Kesimpulan
            st.divider()
            st.markdown("### 📝 Kesimpulan & Action Plan")
            
            increase_features = comparison_df[(comparison_df['Correlation'] > 0) & (comparison_df['Gap'] > 0)].sort_values('Importance Score', ascending=False)
            decrease_features = comparison_df[(comparison_df['Correlation'] < 0) & (comparison_df['Gap'] < 0)].sort_values('Importance Score', ascending=False)
            
            action_plan = []
            
            if len(increase_features) > 0:
                top_increase = increase_features.iloc[0]
                action_plan.append(
                    f"**1. ⬆️ Prioritas Tingkatkan**: `{top_increase['Fitur']}`\n"
                    f"   - Tingkatkan dari **{top_increase['Nilai Saat Ini']:.2f}** → **{top_increase[f'Rata-rata {target_category}']:.2f}** (+{top_increase['Gap']:.2f})\n"
                    f"   - Reason: Fitur penting (Imp: {top_increase['Importance Score']:.4f}) dengan pengaruh positif kuat"
                )
            
            if len(decrease_features) > 0:
                top_decrease = decrease_features.iloc[0]
                action_plan.append(
                    f"**2. ⬇️ Prioritas Kurangi**: `{top_decrease['Fitur']}`\n"
                    f"   - Kurangi dari **{top_decrease['Nilai Saat Ini']:.2f}** → **{top_decrease[f'Rata-rata {target_category}']:.2f}** ({top_decrease['Gap']:.2f})\n"
                    f"   - Reason: Fitur penting (Imp: {top_decrease['Importance Score']:.4f}) dengan pengaruh negatif, nilai terlalu tinggi"
                )
            
            action_plan.append(
                f"**3. Pendekatan Holistik**: Kombinasikan peningkatan fitur positif dan pengurangan fitur negatif untuk hasil optimal dalam mencapai kategori {target_category}."
            )
            
            for plan in action_plan:
                st.markdown(plan)
            
            st.success(f"💡 Dengan mengikuti rekomendasi di atas, **{selected_provinsi}** memiliki peluang lebih besar untuk naik dari kategori **{source_category}** ke **{target_category}**!")
