import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)

# ---------------- Konfigurasi halaman ----------------
st.set_page_config(page_title="Prediksi Kalori McDonald's", page_icon="🍔", layout="wide")

FEATURES = ['Total Fat', 'Carbohydrates', 'Protein']
TARGET = 'Calories'


# ---------------- Data dan model ----------------
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data.dropna(subset=FEATURES + [TARGET]).copy()


@st.cache_resource
def train_model(data):
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    baseline = DummyRegressor(strategy='mean').fit(X_train, y_train)
    model = LinearRegression().fit(X_train, y_train)
    return model, baseline, X_train, X_test, y_train, y_test


def hitung_metrik(y_true, y_pred):
    mask = y_true > 0  # MAPE tidak bisa dihitung jika nilai aktual = 0
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE (%)': mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100,
    }


# ---------------- Sumber data ----------------
st.title("🍔 Prediksi Kalori Menu McDonald's")
st.caption("Linear Regression | Fundamental Data Analyst")

try:
    data = load_data('menu.csv')
except FileNotFoundError:
    st.warning("File menu.csv tidak ditemukan di folder yang sama. Silakan upload di bawah ini.")
    upload = st.file_uploader("Upload menu.csv", type='csv')
    if upload is None:
        st.stop()
    data = load_data(upload)

model, baseline, X_train, X_test, y_train, y_test = train_model(data)
y_pred = model.predict(X_test)
y_pred_base = baseline.predict(X_test)

tab1, tab2, tab3 = st.tabs(["Prediksi", "Evaluasi Model", "Data"])

# ---------------- Tab 1: Prediksi ----------------
with tab1:
    st.subheader("Prediksi kalori dari kandungan gizi")
    st.write("Masukkan kandungan gizi secara manual, atau pilih menu dari dataset untuk mengisi nilainya otomatis.")

    pilihan = st.selectbox("Pilih menu (opsional)",
                           ['(Input manual)'] + sorted(data['Item'].unique().tolist()))

    if pilihan == '(Input manual)':
        nilai = {'Total Fat': 20.0, 'Carbohydrates': 40.0, 'Protein': 15.0}
        kalori_asli = None
    else:
        baris = data[data['Item'] == pilihan].iloc[0]
        nilai = {f: float(baris[f]) for f in FEATURES}
        kalori_asli = float(baris[TARGET])

    c1, c2, c3 = st.columns(3)
    lemak = c1.number_input("Total Fat (gram)", min_value=0.0, max_value=200.0,
                            value=nilai['Total Fat'], step=1.0, key=f"fat_{pilihan}")
    karbo = c2.number_input("Carbohydrates (gram)", min_value=0.0, max_value=300.0,
                            value=nilai['Carbohydrates'], step=1.0, key=f"carb_{pilihan}")
    protein = c3.number_input("Protein (gram)", min_value=0.0, max_value=150.0,
                              value=nilai['Protein'], step=1.0, key=f"prot_{pilihan}")

    if st.button("Prediksi kalori", type="primary"):
        input_df = pd.DataFrame([{'Total Fat': lemak, 'Carbohydrates': karbo, 'Protein': protein}])
        hasil = float(model.predict(input_df)[0])

        m1, m2 = st.columns(2)
        m1.metric("Prediksi kalori", f"{hasil:,.1f} kkal")
        if kalori_asli is not None:
            m2.metric("Kalori asli di dataset", f"{kalori_asli:,.0f} kkal",
                      delta=f"selisih {hasil - kalori_asli:+.1f} kkal", delta_color="off")

    st.info("Catatan: model ini hanya menghitung hubungan statistik antara zat gizi dan kalori. "
            "Hasilnya bukan penilaian sehat atau tidaknya sebuah menu.")

# ---------------- Tab 2: Evaluasi ----------------
with tab2:
    st.subheader("Perbandingan baseline dan Linear Regression (data uji)")

    tabel = pd.DataFrame({
        'Baseline (rata-rata)': hitung_metrik(y_test, y_pred_base),
        'Linear Regression': hitung_metrik(y_test, y_pred),
    }).T.round(4)
    st.dataframe(tabel)

    c1, c2, c3 = st.columns(3)
    c1.metric("Data latih", len(X_train))
    c2.metric("Data uji", len(X_test))
    c3.metric("R2 latih / uji",
              f"{r2_score(y_train, model.predict(X_train)):.4f} / {r2_score(y_test, y_pred):.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, edgecolors='k')
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=2)
        ax.set_xlabel('Actual Calories')
        ax.set_ylabel('Predicted Calories')
        ax.set_title('Actual vs Predicted')
        st.pyplot(fig)
    with col_b:
        koef = pd.DataFrame({'Fitur': FEATURES, 'Koefisien': model.coef_}).set_index('Fitur')
        st.write("**Koefisien model** (tambahan kkal untuk setiap +1 gram)")
        st.bar_chart(koef)
        st.write(f"Intercept: {model.intercept_:.2f}")

# ---------------- Tab 3: Data ----------------
with tab3:
    st.subheader("Gambaran dataset")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah menu", len(data))
    c2.metric("Jumlah kategori", data['Category'].nunique())

    st.write("**Jumlah menu per kategori**")
    st.bar_chart(data['Category'].value_counts())

    st.write("**Korelasi fitur dengan kalori**")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(data[FEATURES + [TARGET]].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.write("**Cuplikan data**")
    st.dataframe(data[['Category', 'Item'] + FEATURES + [TARGET]].head(50))