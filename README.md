# 🩺 TIMed - Análise Automatizada de Sarcopenia e Qualidade Muscular (SaMD)

## 📌 Sobre o Projeto
Este repositório contém o protótipo prospectivo de *Software as a Medical Device* (SaMD) desenvolvido pela **TIMed** para quantificação automática da área muscular transversa e infiltração lipídica (mioesteatose) em exames de Tomografia Computadorizada (TC).

O sistema utiliza arquiteturas de *Deep Learning* baseadas em **Attention U-Net (V5)**, contando com modelos especialistas dedicados aos níveis vertebrais **T12** e **T4**. O objetivo da ferramenta é auxiliar o médico no diagnóstico quantitativo do **IMME-reduzido (Sarcopenia)** de forma rápida, reprodutível e embasada em biomarcadores de imagem.

## ✨ Principais Funcionalidades
- **Roteamento Anatômico Especializado:** Inferência dinâmica baseada no nível vertebral (T12 ou T4) com limiares de ativação calibrados.
- **Processamento Robusto:** Pipeline construído com **MONAI**, garantindo leitura de DICOM, alinhamento automático da rotação do paciente e filtro estrito de unidades Hounsfield (HU).
- **Diagnóstico Clínico Automatizado:** Cálculo do Índice de Massa Muscular Esquelética (IMME) e cruzamento com pontos de corte clínicos consolidados na literatura para sexo e nível anatômico.
- **Análise de Mioesteatose:** Segmentação de sub-regiões musculares para detectar infiltração de gordura (-29 a 30 HU).
- **Exportação para Validação Prospectiva:** Interface interativa que gera painéis radiológicos (PNG) e consolida os dados automaticamente em uma planilha Excel para acompanhamento clínico.

## 🛠️ Stack Tecnológico
- **Frontend & Dashboard:** Streamlit, Pandas, Matplotlib
- **Inteligência Artificial:** PyTorch, MONAI (*Medical Open Network for AI*)
- **Processamento de Imagens Médicas:** PyDICOM, OpenCV, NumPy

## 🚀 Como Executar Localmente

### Pré-requisitos
- Python 3.9+
- Ambiente virtual configurado (`venv` ou `conda`)

### Instalação
1. Clone o repositório:
```bash
git clone [https://github.com/SEU_USUARIO/timed-sarcopenia.git](https://github.com/SEU_USUARIO/timed-sarcopenia.git)
cd timed-sarcopenia
