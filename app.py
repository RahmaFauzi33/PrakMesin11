import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
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
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Title styling */
    h1 {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subheader styling */
    h2 {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #34495e;
        font-size: 1.4rem;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Error message styling */
    .stError {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
    }
    
    /* Metric card styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Form styling */
    .stForm {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
        font-weight: 600;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3498db, transparent);
        margin: 2rem 0;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background-color: #f4f4f4;
        border-radius: 5px;
        padding: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Sidebar header */
    .css-1lcbmhc .css-1outpf7 {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Card-like containers */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Badge styling for predictions */
    .fraud-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .fraud-yes {
        background-color: #ff6b6b;
        color: white;
    }
    
    .fraud-no {
        background-color: #51cf66;
        color: white;
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
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🔍 TA-11 — ANN Fraud Detection (BankSim)</h1>
        <p style="font-size: 1.1rem; color: #7f8c8d; margin-top: 0.5rem;">
            Model: Keras ANN | Preprocessing: Imputasi + One-Hot Encoding + Scaling | Output: Probabilitas Fraud
        </p>
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
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">⚙️ Pengaturan</h2>
        </div>
    """, unsafe_allow_html=True)
    
    threshold = st.slider("🎯 Threshold Fraud", 0.05, 0.95, float(meta.get("threshold", 0.5)), 0.01,
                          help="Nilai ambang batas untuk menentukan apakah transaksi dianggap fraud")
    mode = st.radio("📝 Mode Input", ["Input Manual", "Upload CSV"], index=0)
    
    st.markdown("---")
    
    st.markdown("""
        <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 10px; border-left: 4px solid #4caf50;">
            <h4 style="color: #2e7d32; margin-top: 0;">✅ Artifacts Terdeteksi</h4>
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
                padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">
        <h2 style="margin-top: 0;">📌 Panduan Pemakaian</h2>
    </div>
""", unsafe_allow_html=True)

template_row = build_template_row(meta)
template_df = pd.DataFrame([template_row])

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 10px; 
                    border-left: 4px solid #ffc107; margin-bottom: 1rem;">
            <h4 style="color: #856404; margin-top: 0;">📝 Contoh Input (Manual)</h4>
        </div>
    """, unsafe_allow_html=True)
    st.code("\n".join([f"{k}: {template_row[k]}" for k in list(template_row.keys())[:min(6, len(template_row))]]), language="text")
    if len(template_row) > 6:
        st.info("💡 Catatan: kolom lain (opsional) bisa diisi di bagian 'Advanced fields' saat mode manual.")

with col2:
    st.markdown("""
        <div style="background-color: #d1ecf1; padding: 1rem; border-radius: 10px; 
                    border-left: 4px solid #17a2b8; margin-bottom: 1rem;">
            <h4 style="color: #0c5460; margin-top: 0;">📥 Download Template CSV</h4>
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
    <div style="margin: 1.5rem 0;">
        <h4>👀 Preview Template (1 Baris)</h4>
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
                    padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: white; margin: 0;">✍️ Input Manual (1 Transaksi)</h2>
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
            <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 10px; 
                        border-left: 4px solid #2196f3; margin-bottom: 1.5rem;">
                <h4 style="color: #1565c0; margin-top: 0;">📋 Isi Data Minimal (Disarankan)</h4>
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
                            padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                    <h2 style="color: white; margin: 0 0 1rem 0;">📊 Hasil Prediksi</h2>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div style="background-color: #fff3cd; padding: 1.5rem; border-radius: 10px; 
                                border-left: 5px solid #ffc107; text-align: center;">
                        <h4 style="color: #856404; margin-top: 0;">Probabilitas Fraud</h4>
                        <h1 style="color: #f57c00; margin: 0.5rem 0;">{prob:.4f}</h1>
                        <div style="background-color: #f0f0f0; height: 20px; border-radius: 10px; margin-top: 1rem;">
                            <div style="background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%); 
                                        height: 100%; width: {prob*100}%; border-radius: 10px;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                badge_class = "fraud-yes" if pred == 1 else "fraud-no"
                badge_text = "🚨 FRAUD" if pred == 1 else "✅ NORMAL"
                st.markdown(f"""
                    <div style="background-color: {'#ffebee' if pred == 1 else '#e8f5e9'}; 
                                padding: 1.5rem; border-radius: 10px; 
                                border-left: 5px solid {'#f44336' if pred == 1 else '#4caf50'}; 
                                text-align: center;">
                        <h4 style="color: {'#c62828' if pred == 1 else '#2e7d32'}; margin-top: 0;">Prediksi</h4>
                        <div class="fraud-badge {badge_class}" style="margin: 1rem 0;">
                            {badge_text}
                        </div>
                        <p style="color: #666; margin: 0.5rem 0 0 0;">
                            Threshold: {threshold:.2f}
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
                    padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: white; margin: 0;">📂 Upload CSV</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 10px; 
                    border-left: 4px solid #2196f3; margin-bottom: 1.5rem;">
            <p style="margin: 0; color: #1565c0;">
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
        <div style="margin: 1.5rem 0;">
            <h4>👀 Preview Data</h4>
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
                        padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">
                <h2 style="color: white; margin: 0; text-align: center;">📊 Statistik Prediksi</h2>
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
            <div style="margin: 2rem 0;">
                <h3>📊 Hasil Prediksi Lengkap</h3>
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
