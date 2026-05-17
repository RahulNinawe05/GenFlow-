import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Detection", page_icon="🩺", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
* { font-family: 'DM Sans', sans-serif; }

.header {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.header h1 { color: #f1f5f9; font-size: 2rem; margin: 0; }
.header p  { color: #94a3b8; margin: 0.3rem 0 0 0; font-size: 1rem; }

.result-yes {
    background: #3b0000;
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: #fca5a5;
    font-size: 1.3rem;
    font-weight: 700;
}
.result-no {
    background: #003b1a;
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: #86efac;
    font-size: 1.3rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🩺 Diabetes Detection — ANN Model</h1>
    <p>diabetes.csv + ANN model — prediction</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    # ── diabetes.csv same folder madhe asto pahije ──
    base = Path(__file__).parent
    csv_path = base / "diabetes.csv"

    df = pd.read_csv(csv_path)

    X = df.drop('Outcome', axis=1)
    Y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = Sequential([
        Input(shape=(8,)),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(8,  activation='relu'),
        Dropout(0.2),
        Dense(1,  activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    return model, scaler, history, acc, df

with st.spinner("⏳ Model train it..."):
    model, scaler, history, test_acc, df = train_model()

st.success(f"✅ Model ready! &nbsp;&nbsp; Test Accuracy: **{test_acc*100:.1f}%**")

st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📋 Give me Patient info. ")

    pregnancies = st.slider("🤰 Pregnancies",               0,   17,   1)
    glucose     = st.slider("🍬 Glucose",                   0,  200, 120)
    bp          = st.slider("💓 Blood Pressure (mm Hg)",    0,  122,  70)
    skin        = st.slider("📏 Skin Thickness (mm)",       0,   99,  20)
    insulin     = st.slider("💉 Insulin",                   0,  846,  80)
    bmi         = st.slider("⚖️ BMI",                     0.0, 67.1,25.0, step=0.1)
    dpf         = st.slider("🧬 Diabetes Pedigree Function",0.0, 2.5, 0.5, step=0.01)
    age         = st.slider("🎂 Age",                       1,  100,  30)

    predict = st.button("🔍 Predict it!", use_container_width=True, type="primary")

with col2:
    st.markdown("### 🎯 Result")

    if predict:
        new_patient = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        new_patient = scaler.transform(new_patient)
        prediction  = model.predict(new_patient, verbose=0)
        prob        = prediction[0][0]

        if prob > 0.5:
            st.markdown(f"""
            <div class="result-yes">
                ⚠️ Diabetes (YES)!<br>
                <span style="font-size:0.85rem; font-weight:400;">
                Probability: {prob*100:.1f}%
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-no">
                ✅ Diabetes (NO)!<br>
                <span style="font-size:0.85rem; font-weight:400;">
                Safe Probability: {(1-prob)*100:.1f}%
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("**Risk Level:**")
        st.progress(float(prob))
        st.caption(f"Diabetes probability: {prob*100:.1f}%")

    else:
        st.info("👈 SET THE VALUES & PREDICT IT")

    st.markdown("### 📊 Training Accuracy")

    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    ax.plot(history.history['accuracy'],     color='#60a5fa', lw=2, label='Train')
    ax.plot(history.history['val_accuracy'], color='#4ade80', lw=2, label='Val', linestyle='--')
    ax.set_xlabel('Epochs', color='#94a3b8', fontsize=9)
    ax.set_ylabel('Accuracy', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.legend(facecolor='#1e293b', labelcolor='#f1f5f9', fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.markdown("### 📁 Dataset Info")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows",   len(df))
c2.metric("Features",     8)
c3.metric("Diabetic",     int(df['Outcome'].sum()))
c4.metric("Non-Diabetic", int((df['Outcome'] == 0).sum()))

with st.expander("📄 Dataset Information"):
    st.dataframe(df.head(10), use_container_width=True)