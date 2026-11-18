import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "modelo_mlp.keras"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
CATEGORIAS_PATH = MODELS_DIR / "categorias.json"

# Ordem EXATA das features usada no treino (não alterar)
FEATURES = ["BAIRRO_GRUPO", "DIA_SEMANA", "PERIODO", "TIPO_LOCAL"]

# Rótulos em PT-BR para o formulário
LABELS = {
    "BAIRRO_GRUPO": "Bairro",
    "DIA_SEMANA": "Dia da semana",
    "PERIODO": "Período",
    "TIPO_LOCAL": "Tipo de local",
}

st.set_page_config(page_title="Roubo vs Furto - Celulares SP", page_icon="📱")


# ---------------------------------------------------------------------------
# Carregamento (com cache) de categorias, encoder e modelo
# ---------------------------------------------------------------------------
@st.cache_data
def load_categorias():
    with open(CATEGORIAS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_artifacts():
    import joblib
    from tensorflow import keras

    encoder = joblib.load(ENCODER_PATH)
    model = keras.models.load_model(MODEL_PATH)
    return encoder, model


def classificar(encoder, model, bairro, dia, periodo, tipo_local):
    X = pd.DataFrame(
        [[bairro, dia, periodo, tipo_local]],
        columns=FEATURES,
    )
    X_enc = encoder.transform(X)
    prob_roubo = float(model.predict(X_enc, verbose=0).ravel()[0])
    return prob_roubo


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("📱 Classificador: Roubo vs Furto de Celular (SP)")
st.write(
    "Informe as características de uma ocorrência **já registrada** de subtração de "
    "celular na capital de São Paulo e o modelo (MLP treinada) estima se o caso foi "
    "**Roubo** (com violência/ameaça) ou **Furto** (sem violência)."
)

# Verificação de arquivos ausentes
faltando = [p.name for p in (MODEL_PATH, ENCODER_PATH, CATEGORIAS_PATH) if not p.exists()]
if faltando:
    st.error(
        "Arquivos necessários não encontrados em `models/`: "
        + ", ".join(faltando)
        + ".\n\nVerifique se `modelo_mlp.keras`, `encoder.pkl` e `categorias.json` "
        "estão presentes na pasta `models/`."
    )
    st.stop()

try:
    categorias = load_categorias()
except Exception as e:  # noqa: BLE001
    st.error(f"Não foi possível ler `categorias.json`: {e}")
    st.stop()

# Formulário: um selectbox por feature, na ordem do contrato
with st.form("form_ocorrencia"):
    selecoes = {}
    for feat in FEATURES:
        opcoes = categorias.get(feat, [])
        selecoes[feat] = st.selectbox(LABELS[feat], opcoes)
    enviar = st.form_submit_button("Classificar", type="primary")

if enviar:
    try:
        encoder, model = load_artifacts()
    except ModuleNotFoundError as e:
        st.error(
            "Dependências de ML não instaladas (tensorflow / scikit-learn / joblib). "
            f"Rode `pip install -r requirements.txt`. Detalhe: {e}"
        )
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(
            "Falha ao carregar o modelo ou o encoder. Isso costuma ocorrer por "
            "incompatibilidade de versão do scikit-learn/tensorflow usada para gerar "
            f"os arquivos. Detalhe: {e}"
        )
        st.stop()

    try:
        prob_roubo = classificar(
            encoder,
            model,
            selecoes["BAIRRO_GRUPO"],
            selecoes["DIA_SEMANA"],
            selecoes["PERIODO"],
            selecoes["TIPO_LOCAL"],
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Erro ao executar a classificação: {e}")
        st.stop()

    prob_furto = 1.0 - prob_roubo
    classe = "ROUBO" if prob_roubo >= 0.5 else "FURTO"

    st.markdown("---")
    if classe == "ROUBO":
        st.markdown(f"## 🔴 Classificação: **{classe}**")
    else:
        st.markdown(f"## 🔵 Classificação: **{classe}**")

    col1, col2 = st.columns(2)
    col1.metric("Roubo", f"{prob_roubo * 100:.1f}%")
    col2.metric("Furto", f"{prob_furto * 100:.1f}%")

    st.progress(min(max(prob_roubo, 0.0), 1.0))
    st.caption(
        f"Probabilidade de Roubo: {prob_roubo * 100:.1f}% · "
        f"Probabilidade de Furto: {prob_furto * 100:.1f}% (limiar de decisão: 50%)."
    )

st.markdown("---")
st.warning(
    "**Aviso importante:** este modelo classifica o **tipo** de uma ocorrência que "
    "**já foi registrada** (roubo x furto), a partir de características do boletim. "
    "Ele **não prevê** se, quando ou onde um crime vai acontecer, e não deve ser usado "
    "para esse fim. Trata-se de um trabalho acadêmico com dados públicos agregados."
)
