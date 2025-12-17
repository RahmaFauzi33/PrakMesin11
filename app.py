import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Sembunyikan log INFO/WARNING TensorFlow di console
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Pakai mode legacy tf.keras (kompatibel dengan model lama) di TensorFlow >= 2.20
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

# =========================================
# Page Config
# =========================================
st.set_page_config(
    page_title="TA-11 ANN Fraud Detection (BankSim)",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔍"
)

# =========================================
# Custom CSS Styling
# =========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global font */
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title styling */
    h1 {
        color: #1f77b4;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        letter-spacing: -0.5px;
    }
    
    /* Subheader styling */
    h2 {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 4px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        padding-bottom: 0.5rem;
        letter-spacing: -0.3px;
    }
    
    h3 {
        color: #34495e;
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.2px;
    }
    
    h4 {
        font-weight: 600;
        letter-spacing: -0.1px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 2px 0 20px rgba(0, 0, 0, 0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        font-size: 1rem;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(40, 167, 69, 0.2);
    }
    
    /* Error message styling */
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(220, 53, 69, 0.2);
    }
    
    /* Info message styling */
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(23, 162, 184, 0.2);
    }
    
    /* Metric card styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #6c757d;
        font-weight: 500;
        letter-spacing: 0.2px;
    }
    
    [data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetricContainer"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Form styling */
    .stForm {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 3rem 0;
        border-radius: 2px;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar header */
    .css-1lcbmhc .css-1outpf7 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 1.3rem;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
        outline: none;
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        height: 8px;
    }
    
    .stSlider > div > div > div {
        background: white;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        border: 3px solid #667eea;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        font-weight: 500;
        color: #495057;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Card-like containers */
    .custom-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    
    /* Badge styling for predictions */
    .fraud-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .fraud-yes {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    .fraud-no {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* Download button styling */
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
    }
    
    [data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, #20c997 0%, #28a745 100%);
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.5);
    }
    
    /* Animated gradient background for main sections */
    .gradient-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradientShift 5s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.2);
        margin: 1rem 0;
    }
    
    /* Warning box styling */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
        margin: 1rem 0;
    }
    
    /* Success box styling */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================
# Helpers
# =========================================
ARTIFACTS = {
    "model": "model.keras",
    "meta": "meta.json",
    "num_imputer": "num_imputer.joblib",
    "cat_imputer": "cat_imputer.joblib",
    "scaler": "scaler.joblib",
}

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(ARTIFACTS["model"])
    with open(ARTIFACTS["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_imputer = joblib.load(ARTIFACTS["num_imputer"])
    cat_imputer = joblib.load(ARTIFACTS["cat_imputer"])
    scaler = joblib.load(ARTIFACTS["scaler"])

    # Basic validation
    required_meta_keys = ["num_cols", "cat_cols", "feature_columns"]
    for k in required_meta_keys:
        if k not in meta:
            raise ValueError(f"meta.json tidak punya key wajib: '{k}'")

    return model, meta, num_imputer, cat_imputer, scaler


def build_template_row(meta: dict) -> dict:
    """
    Buat 1 baris template sesuai kolom TRAINING.
    Nilai default dibuat aman untuk demo.
    """
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    row = {}
    # default numeric = 0
    for c in num_cols:
        row[c] = 0.0
    # default category = "Unknown"
    for c in cat_cols:
        row[c] = "Unknown"

    # kalau ada field umum BankSim, isi default yang lebih masuk akal
    if "amount" in row:
        row["amount"] = 120.50
    if "step" in row:
        row["step"] = 85

    # category default yang sering ada di BankSim
    if "category" in row:
        row["category"] = "es_transportation"

    return row


def ensure_required_columns(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Pastikan df memiliki semua kolom yang digunakan saat training.
    Kolom yang tidak ada akan dibuat (NaN) supaya imputasi jalan dan tidak KeyError.
    """
    df = df.copy()

    # drop target jika ada
    if "fraud" in df.columns:
        df = df.drop(columns=["fraud"])

    # drop columns yang dibuang saat training (kalau ada)
    for c in meta.get("dropped_cols", []):
        if c in df.columns:
            df = df.drop(columns=[c])

    # pastikan kolom numerik+kategori training tersedia
    for c in meta["num_cols"] + meta["cat_cols"]:
        if c not in df.columns:
            df[c] = np.nan

    return df


def preprocess(df_in: pd.DataFrame, meta: dict, num_imputer, cat_imputer, scaler):
    """
    Preprocess sesuai pipeline training:
    - pastikan kolom ada
    - imputasi numeric/kategori
    - one-hot (pd.get_dummies)
    - align kolom dengan training (feature_columns)
    - scaling (StandardScaler)
    """
    df = ensure_required_columns(df_in, meta)

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    feature_columns = meta["feature_columns"]

    # Ambil subset kolom training (kalau df punya kolom ekstra, biarkan diabaikan)
    df_num = df[num_cols].copy() if len(num_cols) else pd.DataFrame(index=df.index)
    df_cat = df[cat_cols].copy() if len(cat_cols) else pd.DataFrame(index=df.index)

    # Imputasi
    if len(num_cols):
        X_num = pd.DataFrame(num_imputer.transform(df_num), columns=num_cols, index=df.index)
    else:
        X_num = pd.DataFrame(index=df.index)

    if len(cat_cols):
        X_cat = pd.DataFrame(cat_imputer.transform(df_cat), columns=cat_cols, index=df.index)
        X_cat_oh = pd.get_dummies(X_cat, columns=cat_cols, drop_first=False)
    else:
        X_cat_oh = pd.DataFrame(index=df.index)

    # Gabung
    X_all = pd.concat([X_num, X_cat_oh], axis=1)

    # Align kolom one-hot agar sama dengan training
    X_all = X_all.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    X_scaled = scaler.transform(X_all)

    return X_scaled, X_all


def predict_proba(model, X_scaled: np.ndarray) -> np.ndarray:
    probs = model.predict(X_scaled, verbose=0).ravel()
    # Clip untuk keamanan tampilan
    return np.clip(probs, 0.0, 1.0)


# =========================================
# UI
# =========================================
st.markdown("""
    <div style="text-align: center; padding: 3rem 0 2rem 0;">
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    padding: 2rem; border-radius: 25px; margin-bottom: 1.5rem; 
                    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);">
            <h1 style="margin-bottom: 1rem;">🔍 TA-11 — ANN Fraud Detection (BankSim)</h1>
            <p style="font-size: 1.2rem; color: #6c757d; margin-top: 0.5rem; font-weight: 500; letter-spacing: 0.3px;">
                <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                             background-clip: text; font-weight: 600;">Model:</span> Keras ANN | 
                <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                             background-clip: text; font-weight: 600;">Preprocessing:</span> Imputasi + One-Hot Encoding + Scaling | 
                <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                             background-clip: text; font-weight: 600;">Output:</span> Probabilitas Fraud
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Load artifacts
try:
    model, meta, num_imputer, cat_imputer, scaler = load_artifacts()
except Exception as e:
    st.error(f"Gagal load artifacts. Pastikan file ada: {list(ARTIFACTS.values())}\n\nDetail: {e}")
    st.stop()

# Sidebar Controls
with st.sidebar:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem 1.5rem; border-radius: 20px; margin-bottom: 2rem;
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); text-align: center;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700; 
                        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);">⚙️ Pengaturan</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); border: 1px solid rgba(102, 126, 234, 0.1);">
            <h4 style="color: #495057; margin-top: 0; margin-bottom: 1rem; font-weight: 600;">🎯 Threshold Fraud</h4>
        </div>
    """, unsafe_allow_html=True)
    
    threshold = st.slider("", 0.05, 0.95, float(meta.get("threshold", 0.5)), 0.01,
                          help="Nilai ambang batas untuk menentukan apakah transaksi dianggap fraud",
                          label_visibility="collapsed")
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); border: 1px solid rgba(102, 126, 234, 0.1);">
            <h4 style="color: #495057; margin-top: 0; margin-bottom: 1rem; font-weight: 600;">📝 Mode Input</h4>
        </div>
    """, unsafe_allow_html=True)
    
    mode = st.radio("", ["Input Manual", "Upload CSV"], index=0, label_visibility="collapsed")
    
    st.markdown("---")
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                    padding: 1.5rem; border-radius: 15px; border-left: 5px solid #4caf50;
                    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2); margin-bottom: 1.5rem;">
            <h4 style="color: #2e7d32; margin-top: 0; font-weight: 700; font-size: 1.2rem;">
                ✅ Artifacts Terdeteksi
            </h4>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Numerik", len(meta['num_cols']))
        st.metric("🏷️ Kategori", len(meta['cat_cols']))
    with col2:
        st.metric("🔢 Total Fitur", len(meta['feature_columns']))

# =========================================
# Guidance + Example Template
# =========================================
st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 2rem; border-radius: 20px; margin: 2rem 0;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); border: 1px solid rgba(102, 126, 234, 0.1);">
        <h2 style="margin-top: 0; color: #2c3e50; font-weight: 700;">📌 Panduan Pemakaian</h2>
        <p style="color: #6c757d; font-size: 1.05rem; margin-top: 0.5rem;">
            Ikuti panduan di bawah ini untuk menggunakan aplikasi deteksi fraud dengan optimal
        </p>
    </div>
""", unsafe_allow_html=True)

template_row = build_template_row(meta)
template_df = pd.DataFrame([template_row])

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%); 
                    padding: 1.5rem; border-radius: 15px; 
                    border-left: 5px solid #ffc107; margin-bottom: 1rem;
                    box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);">
            <h4 style="color: #856404; margin-top: 0; font-weight: 700; font-size: 1.2rem;">
                📝 Contoh Input (Manual)
            </h4>
        </div>
    """, unsafe_allow_html=True)
    st.code("\n".join([f"{k}: {template_row[k]}" for k in list(template_row.keys())[:min(6, len(template_row))]]), language="text")
    if len(template_row) > 6:
        st.info("💡 Catatan: kolom lain (opsional) bisa diisi di bagian 'Advanced fields' saat mode manual.")

with col2:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); 
                    padding: 1.5rem; border-radius: 15px; 
                    border-left: 5px solid #17a2b8; margin-bottom: 1rem;
                    box-shadow: 0 4px 15px rgba(23, 162, 184, 0.2);">
            <h4 style="color: #0c5460; margin-top: 0; font-weight: 700; font-size: 1.2rem;">
                📥 Download Template CSV
            </h4>
        </div>
    """, unsafe_allow_html=True)
    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download template_input.csv",
        data=csv_template,
        file_name="template_input.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h3 style="color: #495057; font-weight: 600; margin-bottom: 1rem;">
            👀 Preview Template (1 Baris)
        </h3>
    </div>
""", unsafe_allow_html=True)
st.dataframe(template_df, use_container_width=True, height=150)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================
# Mode: Manual Input
# =========================================
if mode == "Input Manual":
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2.5rem; border-radius: 20px; margin: 2rem 0;
                    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3); text-align: center;">
            <h2 style="color: white; margin: 0; font-weight: 700; font-size: 2.2rem;
                        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);">
                ✍️ Input Manual (1 Transaksi)
            </h2>
        </div>
    """, unsafe_allow_html=True)

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    # Minimal fields (kalau ada)
    minimal_num = [c for c in ["amount", "step"] if c in num_cols]
    minimal_cat = [c for c in ["category"] if c in cat_cols]

    # Sisanya masuk advanced
    advanced_num = [c for c in num_cols if c not in minimal_num]
    advanced_cat = [c for c in cat_cols if c not in minimal_cat]

    with st.form("manual_form"):
        st.markdown("""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        padding: 1.5rem; border-radius: 15px; 
                        border-left: 5px solid #2196f3; margin-bottom: 1.5rem;
                        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.2);">
                <h4 style="color: #1565c0; margin-top: 0; font-weight: 700; font-size: 1.2rem;">
                    📋 Isi Data Minimal (Disarankan)
                </h4>
            </div>
        """, unsafe_allow_html=True)

        cA, cB, cC = st.columns(3)

        values = {}

        # amount & step (kalau ada)
        for c in minimal_num:
            default_val = float(template_row.get(c, 0.0))
            values[c] = cA.number_input(f"💰 {c}", value=default_val, key=f"min_num_{c}")

        # category (kalau ada)
        for c in minimal_cat:
            default_val = str(template_row.get(c, "es_transportation"))
            values[c] = cB.text_input(f"🏷️ {c}", value=default_val, key=f"min_cat_{c}")

        # kalau dataset kamu tidak punya amount/step/category (jarang), fallback tampilkan 1 numeric & 1 cat pertama
        if not minimal_num and len(num_cols):
            c = num_cols[0]
            values[c] = cA.number_input(f"💰 {c}", value=float(template_row.get(c, 0.0)), key=f"fallback_num")
        if not minimal_cat and len(cat_cols):
            c = cat_cols[0]
            values[c] = cB.text_input(f"🏷️ {c}", value=str(template_row.get(c, "Unknown")), key=f"fallback_cat")

        with st.expander("🔧 Advanced Fields (Opsional)", expanded=False):
            st.info("💡 Kolom ini boleh dikosongkan. Jika kosong, akan diisi otomatis (imputasi).")
            c1, c2 = st.columns(2)

            # advanced numeric
            for i, c in enumerate(advanced_num):
                default_val = template_row.get(c, np.nan)
                # input kosong -> pakai NaN
                # Streamlit number_input tidak bisa empty, jadi kita pakai checkbox 'isi?'
                fill = c1.checkbox(f"Isi {c}?", value=False, key=f"fill_{c}")
                if fill:
                    values[c] = c1.number_input(c, value=float(default_val) if pd.notna(default_val) else 0.0, key=f"num_{c}")
                else:
                    values[c] = np.nan

            # advanced categorical
            for c in advanced_cat:
                default_val = str(template_row.get(c, "Unknown"))
                txt = c2.text_input(c, value=default_val, key=f"cat_{c}")
                # jika user mengosongkan, set NaN supaya imputasi
                values[c] = txt if str(txt).strip() != "" else np.nan

        submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)

    if submitted:
        df_in = pd.DataFrame([values])

        try:
            X_scaled, _ = preprocess(df_in, meta, num_imputer, cat_imputer, scaler)
            prob = float(predict_proba(model, X_scaled)[0])
            pred = int(prob >= threshold)

            # Display results with better styling
            st.markdown("---")
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 3rem 2rem; border-radius: 20px; text-align: center; margin: 2rem 0;
                            box-shadow: 0 10px 40px rgba(240, 147, 251, 0.3);">
                    <h2 style="color: white; margin: 0 0 1rem 0; font-weight: 700; font-size: 2.5rem;
                               text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);">📊 Hasil Prediksi</h2>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%); 
                                padding: 2rem; border-radius: 15px; 
                                border-left: 5px solid #ffc107; text-align: center;
                                box-shadow: 0 6px 20px rgba(255, 193, 7, 0.3);">
                        <h4 style="color: #856404; margin-top: 0; font-weight: 700; font-size: 1.3rem;">
                            Probabilitas Fraud
                        </h4>
                        <h1 style="color: #f57c00; margin: 1rem 0; font-size: 3.5rem; font-weight: 800;
                                    text-shadow: 0 2px 10px rgba(245, 124, 0, 0.3);">{prob:.4f}</h1>
                        <div style="background-color: #f0f0f0; height: 25px; border-radius: 12px; 
                                    margin-top: 1.5rem; overflow: hidden; box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%); 
                                        height: 100%; width: {prob*100}%; border-radius: 12px;
                                        transition: width 0.5s ease; box-shadow: 0 2px 8px rgba(255, 107, 107, 0.4);"></div>
                        </div>
                        <p style="color: #856404; margin-top: 0.5rem; font-size: 0.9rem; font-weight: 500;">
                            {prob*100:.2f}% kemungkinan fraud
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                badge_class = "fraud-yes" if pred == 1 else "fraud-no"
                badge_text = "🚨 FRAUD" if pred == 1 else "✅ NORMAL"
                bg_gradient = "linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)" if pred == 1 else "linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)"
                border_color = "#f44336" if pred == 1 else "#4caf50"
                text_color = "#c62828" if pred == 1 else "#2e7d32"
                st.markdown(f"""
                    <div style="background: {bg_gradient}; 
                                padding: 2rem; border-radius: 15px; 
                                border-left: 5px solid {border_color}; 
                                text-align: center;
                                box-shadow: 0 6px 20px rgba({'244, 67, 54' if pred == 1 else '76, 175, 80'}, 0.3);">
                        <h4 style="color: {text_color}; margin-top: 0; font-weight: 700; font-size: 1.3rem;">
                            Prediksi
                        </h4>
                        <div class="fraud-badge {badge_class}" style="margin: 1.5rem 0;">
                            {badge_text}
                        </div>
                        <p style="color: #666; margin: 1rem 0 0 0; font-weight: 500; font-size: 1rem;">
                            Threshold: <strong>{threshold:.2f}</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 Lihat Input yang Diproses (Debug)", expanded=False):
                st.dataframe(df_in, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Gagal prediksi: {e}")

# =========================================
# Mode: Upload CSV
# =========================================
else:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2.5rem; border-radius: 20px; margin: 2rem 0;
                    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3); text-align: center;">
            <h2 style="color: white; margin: 0; font-weight: 700; font-size: 2.2rem;
                        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);">📂 Upload CSV</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                    padding: 1.5rem; border-radius: 15px; 
                    border-left: 5px solid #2196f3; margin-bottom: 1.5rem;
                    box-shadow: 0 4px 15px rgba(33, 150, 243, 0.2);">
            <p style="margin: 0; color: #1565c0; font-size: 1.05rem; font-weight: 500; line-height: 1.6;">
                📤 Upload CSV yang berisi kolom sesuai template. 
                Jika ada kolom yang kurang, aplikasi akan mengisi otomatis (imputasi).
            </p>
        </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📁 Upload file .csv", type=["csv"], 
                                 help="Pilih file CSV yang berisi data transaksi untuk diprediksi")
    if uploaded is None:
        st.info("ℹ️ Belum ada file. Kamu bisa download template CSV di atas dan isi datanya.")
        st.stop()

    try:
        df_in = pd.read_csv(uploaded)
        st.success(f"✅ File berhasil dibaca! Total {len(df_in)} baris data.")
    except Exception as e:
        st.error(f"❌ Gagal membaca CSV: {e}")
        st.stop()

    st.markdown("""
        <div style="margin: 2rem 0 1rem 0;">
            <h3 style="color: #495057; font-weight: 600; margin-bottom: 1rem;">
                👀 Preview Data
            </h3>
        </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_in.head(25), use_container_width=True, height=400)

    try:
        with st.spinner("🔄 Memproses data dan melakukan prediksi..."):
            X_scaled, _ = preprocess(df_in, meta, num_imputer, cat_imputer, scaler)
            probs = predict_proba(model, X_scaled)
            preds = (probs >= threshold).astype(int)

        df_out = df_in.copy()
        df_out["fraud_prob"] = probs
        df_out["fraud_pred"] = preds

        st.markdown("---")
        st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 2.5rem; border-radius: 20px; margin: 2rem 0;
                        box-shadow: 0 10px 40px rgba(240, 147, 251, 0.3); text-align: center;">
                <h2 style="color: white; margin: 0; font-weight: 700; font-size: 2.5rem;
                            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);">📊 Statistik Prediksi</h2>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("📋 Jumlah Baris", len(df_out))
        with c2:
            fraud_count = int(df_out["fraud_pred"].sum())
            fraud_percent = (fraud_count / len(df_out)) * 100 if len(df_out) > 0 else 0
            st.metric("🚨 Prediksi Fraud", f"{fraud_count} ({fraud_percent:.1f}%)")
        with c3:
            st.metric("📈 Rata-rata Probabilitas", f"{float(df_out['fraud_prob'].mean()):.4f}")

        st.markdown("""
            <div style="margin: 2.5rem 0 1rem 0;">
                <h3 style="color: #495057; font-weight: 600; margin-bottom: 1rem;">
                    📊 Hasil Prediksi Lengkap
                </h3>
            </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_out.head(100), use_container_width=True, height=500)

        st.markdown("---")
        csv_output = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download hasil_prediksi.csv",
            data=csv_output,
            file_name="hasil_prediksi.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ Gagal preprocessing/prediksi: {e}")
        import traceback
        with st.expander("🔍 Detail Error"):
            st.code(traceback.format_exc())
        st.stop()
