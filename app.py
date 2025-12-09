import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- KONFIGURASI HALAMAN WEB ---
st.set_page_config(
    page_title="Klasifikasi Minat Baca - Kelompok 3A",
    page_icon="📚",
    layout="wide"
)

# --- JUDUL & NAVIGASI SIDEBAR ---
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio(
    "Pilih Tahapan:",
    ["Home", "1. Input Dataset", "2. Preprocessing", "3. Classification (Random Forest)", "4. Visualisasi"]
)

# --- INISIALISASI SESSION STATE (Agar data tidak hilang saat pindah menu) ---
if 'data_raw' not in st.session_state:
    st.session_state['data_raw'] = None
if 'data_clean' not in st.session_state:
    st.session_state['data_clean'] = None
if 'data_result' not in st.session_state:
    st.session_state['data_result'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = None

# =============================================================================
# MENU: HOME
# =============================================================================
if menu == "Home":
    st.title("📚 Sistem Klasifikasi Minat Baca")
    st.markdown("""
    **Tugas Besar Akuisisi Data - Kelompok 3A**
    
    Sistem ini dibangun untuk mengklasifikasikan tingkat minat baca masyarakat Indonesia 
    berdasarkan data Tingkat Kegemaran Membaca (TGM) tahun 2020-2023.
    
    **Anggota Kelompok:**
    * Darrel Rajendra Kurnia
    * Muhammad Fauzin
    * Farhan Aufa
    
    **Metode Analisis:** Classification dengan Random Forest
    
    **Target Klasifikasi:**
    - **Rendah**: Skor TGM < 36
    - **Sedang**: Skor TGM 36 - 60
    - **Tinggi**: Skor TGM > 60
    """)
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)

# =============================================================================
# MENU 1: INPUT DATASET (VERSI AUTO-DETECT ANTI ERROR)
# =============================================================================
elif menu == "1. Input Dataset":
    st.header("📂 Tahap 1: Akuisisi Data (Input)")
    st.write("Silakan upload file dataset (Format .csv).")

    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # LOGIKA AUTO-DETECT: Coba baca Titik Koma (;) dulu, kalau gagal baru Koma (,)
            try:
                # Prioritas 1: Titik Koma (Format File Asli Anda)
                df = pd.read_csv(uploaded_file, sep=';')
                
                # Cek validitas: Kalau kolom cuma 1, berarti salah baca
                if df.shape[1] <= 1:
                    raise ValueError("Sepertinya bukan titik koma.")
            except:
                # Prioritas 2: Koma (Format Standar)
                uploaded_file.seek(0) # Reset file ke awal
                df = pd.read_csv(uploaded_file, sep=',')
            
            st.session_state['data_raw'] = df
            st.success("Dataset berhasil dimuat! Format file terdeteksi otomatis.")
            
            st.subheader("Tinjauan Data Mentah")
            st.dataframe(df.head(10))
            st.write(f"Dimensi Data: {df.shape[0]} Baris, {df.shape[1]} Kolom")
            
        except Exception as e:
            st.error(f"Gagal membaca file. Detail Error: {e}")
    
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
        
        st.write("Data mentah mengandung format angka Indonesia (koma) dan nilai kosong (NaN).")
        st.write("Sistem akan membersihkan data ini agar bisa dihitung.")
        
        if st.button("Jalankan Pembersihan Data"):
            # 1. Pilih kolom numerik yang perlu dibersihkan
            cols_numeric = [
                'Daily Reading Duration (in minutes)', 
                'Daily Internet Duration (in minutes)',
                'Tingkat Kegemaran Membaca (Reading Interest)'
            ]
            
            # 2. Loop pembersihan (Ubah koma jadi titik, ubah ke float)
            for col in cols_numeric:
                # Cek jika kolom tipe objek (text), replace koma
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(',', '.')
                # Ubah paksa jadi angka
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 3. Mengisi nilai kosong (NaN) dengan 0 (Zero Imputation)
            df = df.fillna(0)
            
            # 4. TAMBAHAN: Membuat Label Kategori untuk Classification
            def categorize_reading_interest(score):
                # Threshold yang lebih seimbang berdasarkan percentile
                if score < 36:
                    return 'Rendah'
                elif score <= 60:
                    return 'Sedang'
                else:
                    return 'Tinggi'
            
            df['Kategori_Minat_Baca'] = df['Tingkat Kegemaran Membaca (Reading Interest)'].apply(categorize_reading_interest)
            
            st.session_state['data_clean'] = df # Simpan hasil cleaning
            st.success("Preprocessing Selesai! Format angka diperbaiki, Missing Values ditangani, dan Label Kategori dibuat.")
            
            # Tampilkan distribusi kategori
            st.info("📊 Distribusi Kategori (Rendah: <36, Sedang: 36-60, Tinggi: >60):")
            category_dist = df['Kategori_Minat_Baca'].value_counts()
            st.write(category_dist)
            
        # Tampilkan data bersih jika sudah ada
        if st.session_state['data_clean'] is not None:
            st.subheader("Data Hasil Preprocessing")
            st.dataframe(st.session_state['data_clean'].head(10))
            st.write("Statistik Deskriptif:")
            st.write(st.session_state['data_clean'].describe())
            
            # Tampilkan distribusi kategori
            st.write("Distribusi Kategori Minat Baca:")
            st.write(st.session_state['data_clean']['Kategori_Minat_Baca'].value_counts())

# =============================================================================
# MENU 3: CLASSIFICATION (ANALISIS)
# =============================================================================
elif menu == "3. Classification (Random Forest)":
    st.header("⚙️ Tahap 3: Analisis Data (Classification)")

    if st.session_state['data_clean'] is None:
        st.warning("Mohon lakukan Preprocessing terlebih dahulu.")
    else:
        df_model = st.session_state['data_clean'].copy()

        st.sidebar.subheader("Konfigurasi Model")
        
        # Pilihan Fitur untuk Classification
        st.write("**Target Variable:** Kategori Minat Baca (Rendah/Sedang/Tinggi)")
        st.write("**Features:**")
        st.code("1. Daily Internet Duration (in minutes)\n2. Daily Reading Duration (in minutes)")
        
        # Parameter Random Forest
        n_estimators = st.sidebar.slider("Jumlah Trees", min_value=50, max_value=300, value=100, step=50)
        max_depth = st.sidebar.slider("Max Depth", min_value=3, max_value=20, value=10)
        test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=40, value=20)

        if st.button("Mulai Training Model"):
            # Persiapan data
            feature_cols = ['Daily Internet Duration (in minutes)', 'Daily Reading Duration (in minutes)']
            X = df_model[feature_cols]
            y = df_model['Kategori_Minat_Baca']
            
            # Cek distribusi kategori
            category_counts = y.value_counts()
            st.write(f"Distribusi kategori: {category_counts.to_dict()}")
            
            # Validasi: Pastikan setiap kategori minimal 2 samples
            min_samples = category_counts.min()
            
            if min_samples < 2:
                st.error(f"⚠️ Error: Ada kategori dengan hanya {min_samples} sample. Minimal 2 sample per kategori untuk train-test split.")
                st.info("💡 Solusi: Gunakan lebih banyak data atau nonaktifkan stratify.")
                use_stratify = False
            else:
                use_stratify = True
            
            # Split data train & test (dengan/tanpa stratify)
            if use_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )
                st.warning("⚠️ Stratify dinonaktifkan karena data tidak mencukupi.")
            
            # Training Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            
            with st.spinner('Training model...'):
                rf_model.fit(X_train, y_train)
            
            # Prediksi
            y_pred = rf_model.predict(X_test)
            
            # Evaluasi
            accuracy = accuracy_score(y_test, y_pred)
            # Dapatkan semua label unik dari data asli
            all_labels = sorted(y.unique())
            conf_matrix = confusion_matrix(y_test, y_pred, labels=all_labels)
            class_report = classification_report(y_test, y_pred, labels=all_labels, output_dict=True, zero_division=0)
            
            # Feature Importance
            feature_imp = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Simpan hasil
            df_model['Prediksi'] = rf_model.predict(X)
            st.session_state['data_result'] = df_model
            st.session_state['model'] = rf_model
            st.session_state['feature_importance'] = feature_imp
            st.session_state['metrics'] = {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'y_test': y_test,
                'y_pred': y_pred,
                'labels': all_labels  # Simpan labels untuk visualisasi
            }
            
            st.success(f"✅ Training selesai! Akurasi Model: {accuracy*100:.2f}%")
            
            # Tampilkan metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
                st.write("**Feature Importance:**")
                st.dataframe(feature_imp)
            
            with col2:
                st.write("**Confusion Matrix:**")
                # Gunakan labels yang tersimpan untuk memastikan dimensi konsisten
                all_labels = sorted(df_model['Kategori_Minat_Baca'].unique())
                st.dataframe(pd.DataFrame(
                    conf_matrix,
                    index=[f'Actual {c}' for c in all_labels],
                    columns=[f'Pred {c}' for c in all_labels]
                ))
            
            st.write("**Classification Report:**")
            st.dataframe(pd.DataFrame(class_report).transpose())
            
            st.write("**Sampel Hasil Prediksi:**")
            st.dataframe(df_model[['Provinsi', 'Year', 'Kategori_Minat_Baca', 'Prediksi']].head(10))

# =============================================================================
# MENU 4: VISUALISASI (HASIL CLASSIFICATION)
# =============================================================================
elif menu == "4. Visualisasi":
    st.header("📊 Tahap 4: Visualisasi & Hasil")

    if st.session_state['data_result'] is None:
        st.warning("Mohon jalankan proses Classification terlebih dahulu di menu 3.")
    else:
        df_viz = st.session_state['data_result']
        metrics = st.session_state.get('metrics')
        feature_imp = st.session_state.get('feature_importance')
        
        # --- TAMPILAN TAB ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Performance Metrics", 
            "Feature Importance", 
            "Distribusi Kategori", 
            "Confusion Matrix",
            "📈 Insight Analysis"
        ])
        
        with tab1:
            st.subheader("📈 Model Performance")
            if metrics:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Accuracy", f"{metrics['accuracy']*100:.2f}%")
                
                # Precision, Recall, F1 per class
                st.write("**Detail Metrics per Kategori:**")
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))
            else:
                st.info("Jalankan training model terlebih dahulu.")
        
        with tab2:
            st.subheader("🎯 Feature Importance")
            if feature_imp is not None:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=feature_imp,
                    x='Importance',
                    y='Feature',
                    palette='viridis',
                    ax=ax1
                )
                ax1.set_title("Importance of Features in Random Forest Model", fontsize=14)
                ax1.set_xlabel("Importance Score", fontsize=12)
                ax1.set_ylabel("Features", fontsize=12)
                st.pyplot(fig1)
                st.info("💡 **Cara Baca:** Semakin tinggi nilai importance, semakin besar pengaruh fitur tersebut dalam prediksi.")
            else:
                st.info("Jalankan training model terlebih dahulu.")

        with tab3:
            st.subheader("📊 Distribusi Kategori Minat Baca")
            
            # Perbandingan Actual vs Predicted
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribusi Kategori Aktual:**")
                actual_counts = df_viz['Kategori_Minat_Baca'].value_counts().sort_index()
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                actual_counts.plot(kind='bar', color='skyblue', ax=ax2)
                ax2.set_title("Distribusi Kategori Aktual", fontsize=12)
                ax2.set_xlabel("Kategori", fontsize=10)
                ax2.set_ylabel("Jumlah", fontsize=10)
                ax2.tick_params(axis='x', rotation=0)
                for i, v in enumerate(actual_counts):
                    ax2.text(i, v + 1, str(v), ha='center', fontsize=10)
                st.pyplot(fig2)
            
            with col2:
                if 'Prediksi' in df_viz.columns:
                    st.write("**Distribusi Prediksi Model:**")
                    pred_counts = df_viz['Prediksi'].value_counts().sort_index()
                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    pred_counts.plot(kind='bar', color='lightcoral', ax=ax3)
                    ax3.set_title("Distribusi Prediksi Model", fontsize=12)
                    ax3.set_xlabel("Kategori", fontsize=10)
                    ax3.set_ylabel("Jumlah", fontsize=10)
                    ax3.tick_params(axis='x', rotation=0)
                    for i, v in enumerate(pred_counts):
                        ax3.text(i, v + 1, str(v), ha='center', fontsize=10)
                    st.pyplot(fig3)
                else:
                    st.info("Jalankan training model terlebih dahulu.")
            
            # Rata-rata nilai per kategori
            st.write("**Rata-rata Durasi Aktivitas per Kategori:**")
            avg_by_category = df_viz.groupby('Kategori_Minat_Baca')[[
                'Daily Reading Duration (in minutes)', 
                'Daily Internet Duration (in minutes)',
                'Tingkat Kegemaran Membaca (Reading Interest)'
            ]].mean()
            st.dataframe(avg_by_category.style.background_gradient(cmap='YlOrRd'))

        with tab4:
            st.subheader("🔄 Confusion Matrix")
            if metrics:
                # Gunakan labels yang tersimpan
                labels = metrics.get('labels', sorted(df_viz['Kategori_Minat_Baca'].unique()))
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    metrics['confusion_matrix'],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels,
                    ax=ax4
                )
                ax4.set_title("Confusion Matrix - Random Forest Classifier", fontsize=14)
                ax4.set_xlabel("Predicted Label", fontsize=12)
                ax4.set_ylabel("True Label", fontsize=12)
                st.pyplot(fig4)
                
                st.info("💡 **Cara Baca:** Diagonal menunjukkan prediksi yang benar. Nilai di luar diagonal adalah kesalahan klasifikasi.")
            else:
                st.info("Jalankan training model terlebih dahulu.")
        
        with tab5:
            st.subheader("📈 Insight Analysis")
            
            # ===== VISUALISASI 1: TREN PENINGKATAN NASIONAL 2020-2023 =====
            st.markdown("### 📈 Tren Peningkatan Minat Baca Nasional (2020-2023)")
            
            # Ambil data nasional (rata-rata semua provinsi per tahun)
            df_yearly = df_viz.groupby('Year').agg({
                'Tingkat Kegemaran Membaca (Reading Interest)': 'mean'
            }).reset_index()
            df_yearly = df_yearly.sort_values('Year')
            
            if len(df_yearly) > 0:
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                
                # Line chart dengan marker
                ax1.plot(df_yearly['Year'], 
                        df_yearly['Tingkat Kegemaran Membaca (Reading Interest)'], 
                        color='green', linewidth=3, marker='o', markersize=10, markerfacecolor='green')
                
                # Tambah label nilai di setiap titik
                for i, row in df_yearly.iterrows():
                    ax1.text(row['Year'], row['Tingkat Kegemaran Membaca (Reading Interest)'] + 0.5, 
                            f"{row['Tingkat Kegemaran Membaca (Reading Interest)']:.2f}", 
                            ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                ax1.set_xlabel('Tahun', fontsize=13, fontweight='bold')
                ax1.set_ylabel('Skor TGM', fontsize=13, fontweight='bold')
                ax1.set_title('Tren Peningkatan Minat Baca Nasional (2020-2023)', fontsize=15, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.set_xticks(df_yearly['Year'])
                
                st.pyplot(fig1)
                
                # Hitung pertumbuhan
                if len(df_yearly) >= 2:
                    growth = df_yearly.iloc[-1]['Tingkat Kegemaran Membaca (Reading Interest)'] - df_yearly.iloc[0]['Tingkat Kegemaran Membaca (Reading Interest)']
                    growth_pct = (growth / df_yearly.iloc[0]['Tingkat Kegemaran Membaca (Reading Interest)']) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Skor 2020", f"{df_yearly.iloc[0]['Tingkat Kegemaran Membaca (Reading Interest)']:.2f}")
                    with col2:
                        st.metric("Skor 2023", f"{df_yearly.iloc[-1]['Tingkat Kegemaran Membaca (Reading Interest)']:.2f}")
                    with col3:
                        st.metric("Pertumbuhan", f"+{growth:.2f} ({growth_pct:.1f}%)")
                
                st.success("✅ **Insight**: Minat baca nasional meningkat konsisten dari 2020 ke 2023!")
            
            st.divider()
            
            # ===== VISUALISASI 2: TOP 5 vs BOTTOM 5 (HORIZONTAL BAR) =====
            st.markdown("### 🏆 Kesenjangan Minat Baca: 5 Provinsi Terbaik vs 5 Terendah (2023)")
            
            # Ambil data tahun 2023
            df_2023 = df_viz[df_viz['Year'] == 2023].copy()
            
            if len(df_2023) > 0:
                # Sorting berdasarkan skor TGM
                df_sorted = df_2023.sort_values('Tingkat Kegemaran Membaca (Reading Interest)', ascending=False)
                
                # Top 5 & Bottom 5
                top5 = df_sorted.head(5).copy()
                bottom5 = df_sorted.tail(5).copy()
                
                # Gabungkan dan reverse urutan bottom5 agar tersusun dari tertinggi di bottom5
                combined = pd.concat([top5, bottom5])
                combined = combined.sort_values('Tingkat Kegemaran Membaca (Reading Interest)', ascending=True)
                
                # Warna: hijau untuk top5, merah untuk bottom5
                colors = ['indianred' if x < 65 else 'green' for x in combined['Tingkat Kegemaran Membaca (Reading Interest)']]
                
                fig2, ax2 = plt.subplots(figsize=(14, 8))
                bars = ax2.barh(combined['Provinsi'], combined['Tingkat Kegemaran Membaca (Reading Interest)'], color=colors)
                
                # Tambah label nilai di ujung bar
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                            f'{width:.2f}', ha='left', va='center', fontsize=11, fontweight='bold')
                
                ax2.set_xlabel('Skor TGM', fontsize=13, fontweight='bold')
                ax2.set_ylabel('Provinsi', fontsize=13, fontweight='bold')
                ax2.set_title('Kesenjangan Minat Baca: 5 Provinsi Terbaik vs 5 Terendah (2023)', fontsize=15, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
                
                st.pyplot(fig2)
                
                # Gap Analysis
                gap = top5.iloc[0]['Tingkat Kegemaran Membaca (Reading Interest)'] - bottom5.iloc[-1]['Tingkat Kegemaran Membaca (Reading Interest)']
                st.warning(f"⚠️ **Gap Kesenjangan**: {gap:.2f} poin antara provinsi terbaik (Yogyakarta) dan terendah (Papua)")
                st.info("💡 **Insight**: Ada kesenjangan signifikan antara provinsi di Jawa dan Indonesia Timur!")
            
            st.divider()
            
            # ===== VISUALISASI 3: KUANTITAS vs KUALITAS (BAR + LINE COMBO) =====
            st.markdown("### 📊 Kuantitas (Durasi) vs Kualitas (Skor) pada 5 Provinsi Teratas")
            
            if len(df_2023) > 0:
                # Ambil top 5 provinsi
                top5_data = df_sorted.head(5).copy()
                
                fig3, ax3 = plt.subplots(figsize=(14, 8))
                
                # Bar chart untuk durasi baca
                x_pos = np.arange(len(top5_data))
                bars = ax3.bar(x_pos, top5_data['Daily Reading Duration (in minutes)'], 
                              color='skyblue', width=0.6, label='Durasi Baca (Menit)')
                
                # Tambah nilai di atas bar
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{int(height)}', ha='center', va='bottom', fontsize=10)
                
                # Line chart untuk skor TGM (secondary axis)
                ax3_2 = ax3.twinx()
                line = ax3_2.plot(x_pos, top5_data['Tingkat Kegemaran Membaca (Reading Interest)'], 
                                 color='red', linewidth=3, marker='o', markersize=10, 
                                 markerfacecolor='red', label='Skor TGM')
                
                # Tambah nilai di line
                for i, val in enumerate(top5_data['Tingkat Kegemaran Membaca (Reading Interest)']):
                    ax3_2.text(i, val + 0.5, f'{val:.2f}', ha='center', va='bottom', 
                              fontsize=10, fontweight='bold', color='red')
                
                # Set labels dan title
                ax3.set_xlabel('Provinsi', fontsize=13, fontweight='bold')
                ax3.set_ylabel('Durasi (Menit)', fontsize=13, fontweight='bold', color='blue')
                ax3_2.set_ylabel('Skor TGM', fontsize=13, fontweight='bold', color='red')
                ax3.set_title('Kuantitas (Durasi) vs Kualitas (Skor) pada 5 Provinsi Teratas', 
                             fontsize=15, fontweight='bold')
                
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(top5_data['Provinsi'], rotation=0, ha='center')
                
                # Legend gabungan
                ax3.legend(loc='upper left', fontsize=11)
                ax3_2.legend(loc='upper right', fontsize=11)
                
                ax3.grid(axis='y', alpha=0.3)
                
                st.pyplot(fig3)
                
                # Tabel perbandingan
                st.write("**📋 Tabel Detail: Durasi vs Skor**")
                comparison_table = top5_data[['Provinsi', 'Daily Reading Duration (in minutes)', 
                                              'Tingkat Kegemaran Membaca (Reading Interest)']].copy()
                comparison_table['Efisiensi (Skor/100 menit)'] = (comparison_table['Tingkat Kegemaran Membaca (Reading Interest)'] / 
                                                                   comparison_table['Daily Reading Duration (in minutes)'] * 100).round(2)
                comparison_table.columns = ['Provinsi', 'Durasi Baca (menit)', 'Skor TGM', 'Efisiensi']
                
                st.dataframe(comparison_table.style.background_gradient(cmap='YlGn', subset=['Skor TGM', 'Efisiensi']).format({
                    'Durasi Baca (menit)': '{:.0f}',
                    'Skor TGM': '{:.2f}',
                    'Efisiensi': '{:.2f}'
                }))
                
                st.success("✅ **Insight**: Yogyakarta (129 menit → 73.27 skor) lebih efisien dibanding Jakarta (126 menit → 69.94 skor). "
                          "**Bukan hanya berapa lama membaca, tetapi KUALITAS bacaan yang menentukan!**")
            
            else:
                st.warning("Data tidak tersedia untuk analisis.")
        
        # --- TABEL RINGKASAN DI BAWAH ---
        st.divider()
        st.subheader("📋 Tabel Hasil Klasifikasi (Sample)")
        if 'Prediksi' in df_viz.columns:
            display_cols = ['Provinsi', 'Year', 'Daily Reading Duration (in minutes)', 
                          'Daily Internet Duration (in minutes)', 'Kategori_Minat_Baca', 'Prediksi']
            st.dataframe(df_viz[display_cols].head(20))
        else:
            st.info("Jalankan training model untuk melihat hasil prediksi.")