import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Kaktus Classifier",
    page_icon="🌵",
    layout="centered"
)

# =========================================================
# CUSTOM THEME
# =========================================================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#2e4630,#486c4a,#6f9e72,#8fc79c);
    background-size: 200% 200%;
    animation: gradientMove 12s ease infinite;
}
@keyframes gradientMove {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
[data-testid="stSidebar"] {background: rgba(0,0,0,0.25);}
h1,h2,h3,p,label,li,b {color:white;}
.stCard {
    background: rgba(255,255,255,0.18);
    padding:20px;
    border-radius:14px;
    backdrop-filter: blur(6px);
}
.stButton>button {
    background:#2ecc71;
    color:white;
    border-radius:10px;
    font-weight:bold;
}

/* DOWNLOAD PDF BUTTON */
.stDownloadButton > button {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-weight: bold;
    border-radius: 10px;
    border: 1px solid #000000 !important;
}
.stDownloadButton > button:hover {
    background-color: #222222 !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================================================
# LOAD H5 MODEL
# =========================================================
@st.cache_resource
def load_keras_model():
    # Pastikan file model_kaktus_final.h5 ada di folder yang sama
    model = tf.keras.models.load_model('model_kaktus_final.h5')
    return model

model_kaktus = load_keras_model()

labels = ["Astrophytum Asteria", "Cereus", "Ferocactus", "Gymnocalycium", "Opuntia"]

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict(img, model):
    # Resize ke 224x224 sesuai standar MobileNetV2
    image = img.resize((224, 224))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    predictions = model.predict(arr)
    return predictions[0]
    
# =========================================================
# MENU
# =========================================================
menu = st.sidebar.radio("Navigasi", ["Informasi Kaktus", "Prediksi Kaktus"])

# =========================================================
# PAGE 1: INFORMASI KAKTUS (DISAMAKAN)
# =========================================================
if menu == "Informasi Kaktus":
    st.markdown("<h1 class='stCard'>🌵 Informasi Tentang Kaktus</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='stCard'>
        <h3>Apa itu Kaktus?</h3>
        <p>
        Kaktus merupakan tanaman sukulen unik yang terkenal karena kemampuan menyimpan air dan memiliki duri
        sebagai bentuk adaptasi. Karena kemampuan tersebut, kaktus dapat bertahan hidup di lingkungan ekstrem 
        seperti gurun. Selain tangguh, kaktus juga sering dijadikan tanaman hias karena mudah dirawat dan estetik. 
        Tanaman ini termasuk dalam keluarga <i>Cactaceae</i>.
        </p>
        
<h3>Fakta Menarik Kaktus:</h3>    
<ul>
    <li>Kaktus dapat hidup hingga ratusan tahun.</li>
    <li>Beberapa kaktus dapat tumbuh lebih dari 20 meter.</li>
    <li>Terdapat lebih dari 2.000 spesies kaktus di dunia.</li>
    <li>Bentuknya sangat beragam: bulat, pipih, memanjang, hingga bercabang.</li>
</ul>

<h3>Kegunaan:</h3>
<ul> 
    <li>Tanaman hias: sebagai dekorasi rumah, taman, atau kamar tidur karena estetika dan perawatannya mudah.</li>
    <li>Konsumsi & Kesehatan: Buah dan daun muda kaktus (seperti pir berduri) bisa dimakan, kaya serat, vitamin, mineral untuk kesehatan.</li>
    <li>Bisa juga digunakan dalam produk perawatan kulit.</li>
</ul>

<h3>Jenis Kaktus Tanaman Hias:</h3>
<ul>
    <li>Astrophytum Asteria.</li>
    <li>Cereus.</li>
    <li>Ferocactus.</li>
    <li>Gymnocalycium.</li>
    <li>Opuntia.</li>
</ul>

""", unsafe_allow_html=True)
    
# =========================================================
# PAGE 2: PREDIKSI
# =========================================================
else:
    st.markdown("<h1 class='stCard'>🔍 Prediksi Jenis Kaktus</h1>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload gambar kaktus", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=280)

        # PREDIKSI (Menggunakan model_kaktus yang sudah di-load)
        preds = predict(img, model_kaktus)
        # Simulasi tampilan perbandingan (karena MobileNetV2 adalah CNN)
        # Di laporan, kamu bisa jelaskan bahwa hasil ini adalah output dari MobileNetV2
        probs = preds / np.sum(preds)
        kelas = labels[np.argmax(probs)]
        conf = np.max(probs)

    st.markdown(f"""
        <div class='stCard'>
        <h3>Hasil Analisis Model (CNN - MobileNetV2)</h3>
        <p><b>Prediksi Spesies:</b> {kelas}</p>
        <p><b>Confidence:</b> {conf:.2%}</p>
        <p>Metode yang digunakan adalah algoritma CNN dengan arsitekstur MobileNetV2.</p>
        </div>
        """, unsafe_allow_html=True)

        # ===== BAR CHART =====
        fig, ax = plt.subplots()
        ax.bar(labels, probs, color='#2ecc71', alpha=0.8)
        ax.set_ylim(0,1)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ===== PDF =====
        buffer = io.BytesIO()
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.colors import Color
        from reportlab.lib.utils import ImageReader

        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        green_dark  = Color(0/255, 70/255, 32/255)
        green_main  = Color(56/255, 142/255, 60/255)
        green_light = Color(220/255, 240/255, 220/255)

        c.setFillColor(green_light)
        c.rect(0, 0, width, height, fill=1)
        c.setFillColor(green_main)
        c.rect(0, height-100, width, 100, fill=1)

        c.setFillColor(Color(1,1,1))
        c.setFont("Helvetica-Bold", 24)
        c.drawString(40, height-60, "🌵 Hasil Prediksi Kaktus")

        # Card Putih
        c.setFillColor(Color(1,1,1))
        c.roundRect(40, 80, width-80, height-220, 20, fill=1)

        # Foto Kaktus
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 60, height-420, 220, 220)

        # Teks Detail
        c.setFillColor(green_dark)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(320, height-240, "Detail Klasifikasi")
        c.setFont("Helvetica", 13)
        c.drawString(320, height-260, f"Prediksi : {kelas}")
        c.drawString(320, height-280, f"Confidence : {conf:.2%}")
        c.drawString(320, height-320, "Metode : CNN (MobileNetV2)")

        # ===== Footer =====
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(green_dark)
        c.drawString(40, 60, "Generated by Kaktus Classifier App")

        
        # ===== GRAFIK KE PDF =====
        gbuf = io.BytesIO()
        fig.savefig(gbuf, format="PNG")
        gbuf.seek(0)
        c.drawImage(ImageReader(gbuf), 120, 120, 350, 220)

        c.save()
        buffer.seek(0)

        st.download_button(
            "📥 Download Hasil Prediksi (PDF)",
            buffer,
            file_name="hasil_prediksi_kaktus.pdf",
            mime="application/pdf"
        )
