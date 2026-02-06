import streamlit as st
import torch
import os
from model import CSRNet
from PIL import Image
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
import pandas as pd

# -----------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------
st.set_page_config(
    page_title="Crowd Counting System",
    page_icon="📸",
    layout="wide"
)

# -----------------------------
# TÍTULO PERSONALIZADO
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #1e88e5; font-size: 42px; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.6);'>
        Crowd Counting System
    </h1>
    <p style='text-align: center; margin-top: -10px; color: #90caf9; font-size: 18px;'>
        
    </p>
    <br>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """<hr style="border:1px solid #e6e6e6; margin-top:-20px; margin-bottom:30px;">""",
    unsafe_allow_html=True
)

# -----------------------------
# TRANSFORMAÇÃO PARA O MODELO
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# FUNÇÃO PARA CARREGAR O MODELO
# -----------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "weights.pth")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = CSRNet()
    model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_model()

# -----------------------------
# UPLOAD DA IMAGEM
# -----------------------------
st.subheader("📤 Envie uma imagem para análise")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    help="Envie fotos aéreas, de multidões ou grandes aglomerações."
)

# -----------------------------
# PROCESSAMENTO DA IMAGEM
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Preprocessamento
    img_tensor = transform(image)
    output = model(img_tensor.unsqueeze(0))
    count = int(output.detach().cpu().sum().numpy())

    # -----------------------------
    # CARD DO RESULTADO
    # -----------------------------
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 12px; background: #f0f7ff; border: 1px solid #cce0ff;">
            <h2 style="color:#004c99; margin:0; font-size:10px;">
                📊 Estimativa de pessoas: <b>{count}</b>
            </h2>
        </div>
        <br>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # DADOS ADICIONAIS E ACURÁCIA
    # -----------------------------
    st.markdown("### 🔢 Detalhes da previsão")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Contagem prevista", f"{count}")

    with col2:
        usar_real = st.checkbox(
            "Tenho a contagem real",
            help="Marque para informar a contagem real e calcular a acurácia."
        )

    real_count = None
    if usar_real:
        real_count = st.number_input(
            "Informe a contagem real",
            min_value=1,
            step=1,
            value=1,
            help="Insira o número real de pessoas para comparar com a previsão."
        )

    if real_count is not None and real_count > 0:
        erro_abs = abs(real_count - count)
        erro_rel = erro_abs / real_count if real_count != 0 else 0
        acuracia = max(0.0, 1 - erro_rel) * 100

        st.markdown("### 📈 Acurácia da previsão")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Contagem real", f"{real_count}")
        with c2:
            st.metric("Erro absoluto", f"{erro_abs}")
        with c3:
            st.metric("Acurácia", f"{acuracia:.2f}%")

        # Gráfico comparando previsão x real
        df_comp = pd.DataFrame(
            {"Contagem": [count, real_count]},
            index=["Prevista", "Real"]
        )
        st.bar_chart(df_comp, use_container_width=True)

    # -----------------------------
    # HEATMAP
    # -----------------------------
    st.subheader("Mapa de Densidade")

    density_map = output.detach().cpu().numpy()[0][0]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(density_map, cmap="jet")
    ax.axis("off")
    st.pyplot(fig)
