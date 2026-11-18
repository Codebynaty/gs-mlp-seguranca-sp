"""Smoke test: carrega encoder + modelo + categorias e roda UMA predição
com a primeira categoria de cada feature, imprimindo a probabilidade de roubo.

Uso:
    python scripts/smoke_test.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

FEATURES = ["BAIRRO_GRUPO", "DIA_SEMANA", "PERIODO", "TIPO_LOCAL"]


def main():
    with open(MODELS_DIR / "categorias.json", encoding="utf-8") as f:
        categorias = json.load(f)

    encoder = joblib.load(MODELS_DIR / "encoder.pkl")
    model = keras.models.load_model(MODELS_DIR / "modelo_mlp.keras")

    valores = [categorias[feat][0] for feat in FEATURES]
    X = pd.DataFrame([valores], columns=FEATURES)
    print("Entrada:", dict(zip(FEATURES, valores)))

    X_enc = encoder.transform(X)
    print("Shape codificado:", X_enc.shape)

    prob_roubo = float(model.predict(X_enc, verbose=0).ravel()[0])
    classe = "ROUBO" if prob_roubo >= 0.5 else "FURTO"
    print(f"Probabilidade de roubo: {prob_roubo:.4f}")
    print(f"Probabilidade de furto: {1 - prob_roubo:.4f}")
    print(f"Classe prevista: {classe}")


if __name__ == "__main__":
    main()
