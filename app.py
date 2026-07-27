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
# CUSTOM THEME (GREEN CACTUS + DARK OVERLAY)
# =========================================================

page_bg = """
<style>

/* ===== Premium Green Gradient Background ===== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #2e4630 0%,
        #486c4a 35%,
        #6f9e72 70%,
        #8fc79c 100%
    ) !important;
    background-size: 200% 200%;
    animation: gradientMove 12s ease infinite;
}

/* Animasi halus (biar keliatan mahal) */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}



/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.25) !important;
    backdrop-filter: blur(4px);
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ===== ALL TEXT COLOR (biar terlihat) ===== */
h1, h2, h3, h4, h5, h6,
p, label, li, strong, b {
    color: #ffffff !important;
}

/* ===== FILE UPLOADER LABEL ===== */
.stFileUploader > label {
    color: #ffffff !important;
    font-weight: bold;
}

/* ===== Kotak “Card” (semi transparan) ===== */
.stCard {
    background: rgba(255,255,255,0.18) !important;
    padding: 20px;
    border-radius: 14px;
    backdrop-filter: blur(6px);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

/* ===== Input teks dan selectbox ===== */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    color: #ffffff !important;
}

/* ===== Buttons ===== */
.stButton>button {
    background-color: #2ecc71 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
    border: 1px solid #27ae60;
}
.stButton>button:hover {
    background-color: #27ae60 !important;
}

/* ===== Buat ul / li terlihat ===== */
ul li {
    color: #ffffff !important;
    font-size: 16px;
}

/* ===== FORCE VISIBILITY UNTUK FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(0,0,0,0.25) !important;
    padding: 15px !important;
    border-radius: 12px !important;
}

/* ===== BACKGROUND GELAP UNTUK H3 YANG DI DALAM CARD ===== */
.stCard h3 {
    background: rgba(0,0,0,0.20) !important;
    padding: 6px 12px !important;
    border-radius: 8px !important;
    display: inline-block;
    color: #ffffff !important;
}

/* ===== FIX TANPA KOTAK DI ICON UPLOAD ===== */

/* Hilangkan fill di elemen-elemen icon */
[data-testid="stFileUploaderDropzone"] svg rect,
[data-testid="stFileUploaderDropzone"] svg path,
[data-testid="stFileUploaderDropzone"] svg polygon,
[data-testid="stFileUploaderDropzone"] svg line,
[data-testid="stFileUploaderDropzone"] svg circle {
    fill: none !important;
}

/* Styling icon upload */
[data-testid="stFileUploaderDropzone"] svg {
    stroke: #000000 !important;
    background: transparent !important;
    width: 40px !important;
    height: 40px !important;
    margin-bottom: 10px !important;
    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ===== Styling Button Download PDF ===== */
.stDownloadButton > button {
    background-color: #000000 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 10px;
    border: 1px solid #000000 !important;
}
.stDownloadButton > button:hover {
    background-color: #ffffff !important;
}


</style>

"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================================================
# LOAD MODEL TFLITE
# =========================================================
@st.cache_resource
def load_tflite():
    interpreter = tf.lite.Interpreter(model_path="mobilenetv2_kaktus.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# label kelas
labels = ["Astrophytum Asteria", "Cereus", "Ferocactus", "Gymnocalycium", "Opuntia"]

# =========================================================
# FUNCTION PREDIKSI
# =========================================================
def predict(img):
    image = img.resize((224,224))
    arr = np.array(image)/255.0
    arr = np.expand_dims(arr, axis=0).astype("float32")

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    return preds

# =========================================================
# HALAMAN MENU
# =========================================================

menu = st.sidebar.radio("Navigasi", ["Informasi Kaktus", "Klasifikasi Kaktus"])

# =========================================================
# PAGE 1: INFORMASI KAKTUS
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
# PAGE 2: PREDIKSI KAKTUS
# =========================================================
else:
    st.markdown("<h1 class='stCard'>🔍 Klasifikasi Jenis Kaktus</h1>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload gambar kaktus", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=280)

        # PREDIKSI (Menggunakan model_kaktus yang sudah di-load)
        preds = predict(img)

        probs = preds / np.sum(preds)
        kelas = labels[np.argmax(probs)]
        conf = np.max(probs)

                # =====================================================
        # CEK APAKAH GAMBAR TERMASUK DATA LATIH
        # =====================================================
        ranking = np.argsort(probs)

        top1 = probs[ranking[-1]]
        top2 = probs[ranking[-2]]

        margin = top1 - top2

        # Deteksi gambar bukan kaktus
        if top1 < 0.60:

            st.error("""
        ❌ Gambar yang diunggah bukan tanaman kaktus.

        Silakan unggah gambar kaktus.
        """)

        # Deteksi kaktus tapi tidak termasuk dataset
    elif margin < 0.15:

        st.warning("""
        ⚠️ Gambar merupakan tanaman kaktus,
        namun tidak termasuk ke dalam lima jenis kaktus
        yang digunakan pada penelitian ini.
        """)

        # Jika lolos
        else:

            # ==========================
            # INTERPRETASI
            # ==========================
            if conf >= 0.80:
                status = "Sangat yakin"
            elif conf >= 0.60:
                status = "Yakin"
            elif conf >= 0.40:
                status = "Cukup yakin"
            elif conf >= 0.20:
                status = "Rendah"
            else:
                status = "Sangat rendah"

            st.markdown(f"""
            <div class='stCard'>
            <h3>Hasil Klasifikasi Model (CNN dengan Arsitektur MobileNetV2)</h3>
            <p><b>Hasil Klasifikasi:</b> {kelas}</p>
            <p><b>Tingkat Keyakinan Model:</b> {conf:.2%}</p>
            <p><b>Interpretasi:</b> {status}</p>
            <p>Metode yang digunakan adalah algoritma CNN dengan arsitektur MobileNetV2.</p>
            </div>
            """, unsafe_allow_html=True)

            # ==========================
            # GRAFIK
            # ==========================
            ranking = np.argsort(probs)

            colors = ['#BDBDBD'] * len(probs)
            colors[ranking[-1]] = '#2ECC71'
            colors[ranking[-2]] = '#F1C40F'
            colors[ranking[-3]] = '#E67E22'
            colors[ranking[-4]] = '#E74C3C'

            fig, ax = plt.subplots(figsize=(8,5))
            ax.bar(labels, probs, color=colors)

            ax.set_ylim(0,1)
            ax.set_ylabel("Probabilitas")
            ax.set_xlabel("Jenis Kaktus")
            ax.set_title("Hasil Klasifikasi Setiap Jenis Kaktus")

            ax.axhline(
                y=0.20,
                color="gray",
                linestyle="--",
                linewidth=1
            )

            ax.text(
                4.15,
                0.215,
                "Threshold 0.20",
                fontsize=9,
                color="gray"
            )

            plt.xticks(rotation=35)
            st.pyplot(fig)

        # ===== PDF =====
        buffer = io.BytesIO()
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.colors import Color
        from reportlab.lib.utils import ImageReader

        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        green_dark = Color(0/255, 70/255, 32/255)
        green_main = Color(56/255, 142/255, 60/255)
        green_light = Color(220/255, 240/255, 220/255)
        
        c.setFillColor(green_light)
        c.rect(0, 0, width, height, fill=1)
        c.setFillColor(green_main)
        c.rect(0, height-100, width, 100, fill=1)

        c.setFillColor(Color(1,1,1))
        c.setFont("Helvetica-Bold", 24)
        c.drawString(40, height-60, "🌵 Hasil Klasifikasi Kaktus")

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
        c.drawString(320, height-260, f"Klasifikasi : {kelas}")
        c.drawString(320, height-280, f"Tingkat Keyakinan : {conf:.2%}")
        c.drawString(320, height-300, f"Interpretasi : {status}")
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
            "📥 Download Hasil Klasifikasi (PDF)",
            buffer,
            file_name="hasil_klasifikasi_kaktus.pdf",
            mime="application/pdf"
        )
