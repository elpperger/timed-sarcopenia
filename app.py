import streamlit as st
import os
import io
import tempfile
import torch
import numpy as np
import pydicom
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Resized, EnsureTyped, Lambdad
)
from monai.networks.nets import AttentionUnet

# ==========================================
# CONFIGURAÇÕES GERAIS E DISPOSITIVO
# ==========================================
st.set_page_config(page_title="TIMed - Sarcopenia IA", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# PIPELINES MONAI E FUNÇÕES AUXILIARES
# ==========================================
base_pipeline = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Lambdad(keys=["image"], func=lambda x: x.squeeze(-1) if x.ndim == 4 and x.shape[-1] == 1 else x),
])

net_pipeline = Compose([
    ScaleIntensityRanged(keys=["image"], a_min=-190, a_max=150, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image"], spatial_size=[512, 512], mode="bilinear"),
    EnsureTyped(keys=["image"], dtype=torch.float32)
])

def pos_processar_mascara(mask_binaria, area_pixel_cm2, min_area_cm2=0.50):
    if min_area_cm2 <= 0: return mask_binaria
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binaria.astype(np.uint8), connectivity=8)
    mask_limpa = np.zeros_like(mask_binaria, dtype=np.uint8)
    for i in range(1, num_labels):
        area_cm2 = stats[i, cv2.CC_STAT_AREA] * area_pixel_cm2
        if area_cm2 >= min_area_cm2:
            mask_limpa[labels == i] = 1
    return mask_limpa

def auto_alinhar_anatomia(img_hu, mask_total):
    corpo = img_hu > -500
    y, x = np.where(corpo)
    if len(y) == 0: return img_hu, mask_total
    if (y.max() - y.min()) > (x.max() - x.min()):
        img_hu = np.rot90(img_hu, k=-1)
        mask_total = np.rot90(mask_total, k=-1)
    ossos = img_hu > 300
    yo, _ = np.where(ossos)
    if len(yo) > 0 and yo.mean() < (img_hu.shape[0] / 2):
        img_hu = np.rot90(img_hu, k=2)
        mask_total = np.rot90(mask_total, k=2)
    return img_hu.copy(), mask_total.copy()

# ==========================================
# CARREGAMENTO DOS MODELOS (CACHE)
# ==========================================
@st.cache_resource
def carregar_modelos():
    def carregar_modelo_unico(nome_arquivo):
        net = AttentionUnet(
            spatial_dims=2, in_channels=1, out_channels=1,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
        ).to(device)
        
        # Lê os pesos da mesma pasta onde está o app.py
        if not os.path.exists(nome_arquivo):
            # Fallback amigável caso esqueça os arquivos
            st.error(f"Arquivo de pesos não encontrado: {nome_arquivo}. Coloque-o na mesma pasta do app.py.")
            return None
            
        net.load_state_dict(torch.load(nome_arquivo, map_location=device))
        net.eval()
        return net

    return {
        "T12": carregar_modelo_unico("best_attention_unet_t12_v5.pth"),
        "T4": carregar_modelo_unico("best_attention_unet_t4_v5.pth")
    }

modelos_especialistas = carregar_modelos()

# ==========================================
# FUNÇÃO PRINCIPAL DE INFERÊNCIA
# ==========================================
def processar_exame_ia(dicom_upload, nivel_vertebral, altura_m, sexo_str):
    sexo = 'M' if sexo_str == "Masculino" else 'F'
    modelo_ativo = modelos_especialistas[nivel_vertebral]
    
    # 1. Salva o upload em um arquivo temporário para o MONAI e PyDICOM lerem
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
        tmp.write(dicom_upload.getvalue())
        tmp_path = tmp.name

    try:
        # 2. Leitura e Preparação
        ds = pydicom.dcmread(tmp_path)
        ps_x, ps_y = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
        area_por_pixel_cm2 = (ps_x * ps_y) / 100.0

        dado_base = base_pipeline({"image": tmp_path})
        img_hu = dado_base["image"].squeeze().numpy()

        dado_net = net_pipeline(dado_base)
        input_tensor = dado_net["image"].unsqueeze(0).to(device)

        # 3. Inferência IA
        th_operacional = 0.50 if nivel_vertebral == "T12" else 0.40
        with torch.no_grad():
            output = modelo_ativo(input_tensor)
            mask_pred = (output.sigmoid() > th_operacional).squeeze().cpu().numpy().astype(np.uint8)

        # 4. Pós-processamento e Alinhamento
        mask_total = cv2.resize(mask_pred, (img_hu.shape[1], img_hu.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_hu_valida = (img_hu >= -29) & (img_hu <= 150)
        mask_total = mask_total & mask_hu_valida
        mask_total = pos_processar_mascara(mask_total, area_por_pixel_cm2, min_area_cm2=0.50)
        img_hu, mask_total = auto_alinhar_anatomia(img_hu, mask_total)

        # 5. Cálculos Clínicos
        area_total_cm2 = np.sum(mask_total) * area_por_pixel_cm2
        imme = area_total_cm2 / (altura_m ** 2)

        # 6. Regras Diagnósticas Rigorosas
        if nivel_vertebral == "T12":
            corte = 30.06 if sexo == 'M' else 23.20
            diag = "IMME-reduzido (Sarcopenia)" if imme <= corte else "Normal"
        else: # T4
            if sexo == 'M':
                diag = "IMME-reduzido (Sarcopenia)" if imme < 39.79 else "Normal"
            else:
                diag = "IMME-reduzido (Sarcopenia)" if imme <= 36.12 else "Normal"

        # 7. Qualidade Muscular (Mioesteatose)
        mask_musculo_puro = (mask_total == 1) & (img_hu >= 30) & (img_hu <= 150)
        mask_mioesteatose = (mask_total == 1) & (img_hu >= -29) & (img_hu < 30)
        area_mioesteatose_cm2 = np.sum(mask_mioesteatose) * area_por_pixel_cm2
        pixels_totais = np.sum(mask_total)
        pct_mioesteatose = (np.sum(mask_mioesteatose) / pixels_totais * 100) if pixels_totais > 0 else 0.0
        densidade_media_hu = float(np.mean(img_hu[mask_total == 1])) if pixels_totais > 0 else 0.0

        # 8. Plotagem do Matplotlib
        img_norm = np.clip((img_hu - (-150)) / (250 - (-150)), 0, 1)
        overlay_qualidade = np.stack([img_norm, img_norm, img_norm], axis=-1)
        overlay_qualidade[mask_musculo_puro] = 0.4 * overlay_qualidade[mask_musculo_puro] + 0.6 * np.array([0.0, 0.9, 1.0])
        overlay_qualidade[mask_mioesteatose] = 0.3 * overlay_qualidade[mask_mioesteatose] + 0.7 * np.array([1.0, 0.25, 0.0])

        fig = plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1)
        plt.title(f"TC Alinhada ({nivel_vertebral})", fontsize=12)
        plt.imshow(img_hu, cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title(f"Especialista {nivel_vertebral} V5 (th={th_operacional})\nÁrea: {area_total_cm2:.1f} cm² | IMME: {imme:.1f}", fontsize=12)
        plt.imshow(img_hu, cmap="gray", vmin=-150, vmax=250)
        plt.imshow(np.ma.masked_where(mask_total == 0, mask_total), cmap="autumn", alpha=0.45)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title(f"Qualidade Muscular\nCiano: Funcional | Laranja: Mioesteatose ({pct_mioesteatose:.1f}%)", fontsize=12)
        plt.imshow(overlay_qualidade)
        plt.axis("off")
        plt.tight_layout()

        # 9. Retorno para o Streamlit
        resultados = {
            "Nível": nivel_vertebral, "Área (cm²)": round(area_total_cm2, 2), "IMME": round(imme, 2),
            "Diagnóstico": diag, "Mioesteatose (cm²)": round(area_mioesteatose_cm2, 2),
            "Infiltração (%)": round(pct_mioesteatose, 1), "Densidade (HU)": round(densidade_media_hu, 1)
        }
        return fig, resultados

    finally:
        # Garante que o arquivo temporário será apagado mesmo se houver erro
        os.remove(tmp_path)


# ==========================================
# INTERFACE FRONTEND (STREAMLIT)
# ==========================================
st.title("🩺 TIMed - Análise Automatizada de Sarcopenia (V5)")

st.sidebar.header("Dados do Paciente")
altura = st.sidebar.number_input("Altura (m) - Para cálculo do IMME", min_value=1.00, max_value=2.50, value=1.70, step=0.01)
sexo = st.sidebar.selectbox("Sexo", ["Masculino", "Feminino"])
nivel_vert = st.sidebar.selectbox("Nível Vertebral", ["T12", "T4"])

# Texto de referência dinâmico
if nivel_vert == "T12":
    corte_sarcopenia = 30.06 if sexo == "Masculino" else 23.20
    operador_texto = "<="
else:
    corte_sarcopenia = 39.79 if sexo == "Masculino" else 36.12
    operador_texto = "<" if sexo == "Masculino" else "<="

st.sidebar.info(
    f"ℹ️ **Referência Diagnóstica**\n\n"
    f"**Cálculo:** Área Muscular (cm²) / Altura² (m²)\n\n"
    f"**Ponto de Corte ({sexo} - {nivel_vert}):**\n"
    f"IMME-reduzido {operador_texto} **{corte_sarcopenia} cm²/m²**"
)

st.sidebar.markdown("---")
st.sidebar.header("Upload do Exame")
st.sidebar.warning("⚠️ Certifique-se de usar o Horos para anonimizar o DICOM antes do upload.")
dicom_file = st.sidebar.file_uploader("Selecione o arquivo DICOM (.dcm)", type=["dcm"])

if dicom_file is not None:
    if st.sidebar.button("Processar Exame", type="primary"):
        # Verifica se os modelos foram carregados corretamente
        if modelos_especialistas[nivel_vert] is None:
            st.error("Erro interno: Pesos do modelo não encontrados. Verifique os arquivos .pth.")
        else:
            with st.spinner('Processando imagem e extraindo métricas biomarcadoras...'):
                fig, res_num = processar_exame_ia(dicom_file, nivel_vert, altura, sexo)
                
                st.subheader(f"Resultados da Avaliação: Nível {nivel_vert}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Área Muscular Total", f"{res_num['Área (cm²)']} cm²")
                
                cor_delta = "normal" if res_num['Diagnóstico'] == "Normal" else "inverse"
                col2.metric("IMME", f"{res_num['IMME']} cm²/m²", res_num['Diagnóstico'], delta_color=cor_delta)
                col3.metric("Infiltração Lipídica", f"{res_num['Infiltração (%)']} %")
                
                st.pyplot(fig)
                
                # Prepara Downloads
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format="png", bbox_inches='tight')
                img_buffer.seek(0)
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(
                        label="💾 Baixar Painel de Imagens (PNG)",
                        data=img_buffer,
                        file_name=f"resultado_{nivel_vert}_paciente.png",
                        mime="image/png"
                    )
                
                nome_arquivo_excel = "banco_dados_sarcopenia.xlsx"
                nova_linha = {
                    "Data_Hora": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Sexo": sexo, "Altura_m": altura, "Nivel": res_num['Nível'], 
                    "Area_cm2": res_num['Área (cm²)'], "IMME": res_num['IMME'], 
                    "Diagnostico": res_num['Diagnóstico'], "Mioesteatose_cm2": res_num['Mioesteatose (cm²)'],
                    "Infiltracao_Perc": res_num['Infiltração (%)'], "Densidade_HU": res_num['Densidade (HU)']
                }
                df_nova_linha = pd.DataFrame([nova_linha])
                
                if os.path.exists(nome_arquivo_excel):
                    df_final = pd.concat([pd.read_excel(nome_arquivo_excel), df_nova_linha], ignore_index=True)
                else:
                    df_final = df_nova_linha
                df_final.to_excel(nome_arquivo_excel, index=False)
                
                with col_btn2:
                    with open(nome_arquivo_excel, "rb") as f:
                        st.download_button(
                            label="📊 Baixar Planilha Excel Atualizada",
                            data=f, file_name="banco_dados_sarcopenia.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )