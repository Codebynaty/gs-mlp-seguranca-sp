# GS MLP Segurança SP — Roubo vs Furto de Celulares

App web (Streamlit) que usa uma rede neural MLP treinada em Keras/TensorFlow para
classificar uma ocorrência **já registrada** de subtração de celular na capital de
São Paulo como **Roubo** (art. 157, com violência/ameaça) ou **Furto** (art. 155,
sem violência).

## Integrantes

- Natalia Heloisa Rocha de Alcântara Bezerra — RM 568570
- Raissa Marinho de Jesus Viana — RM 568301

Turma: **2CCPA** — FIAP, Global Solution 2026.2.

## Fonte dos dados

- **Órgão:** Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP)
- **Base:** "Celulares Subtraídos 2026"
- **Link:** https://www.ssp.sp.gov.br/estatistica/consultas

Os dados foram obtidos a partir das consultas públicas de estatísticas da SSP-SP
(base de celulares subtraídos). Aplicou-se o seguinte recorte: apenas ocorrências da
**capital de São Paulo**, no período de **01/01/2026 a 30/09/2026**, mantendo somente
os registros rotulados como **roubo** (`Roubo (art. 157)`) ou **furto**. A partir
desse recorte foi criado o rótulo binário `ROUBO` (1 = roubo, 0 = furto) e uma amostra
estratificada para o treino, preservando a proporção original entre as duas classes.

## O que o modelo faz

A partir de 4 atributos categóricos de uma ocorrência — **Bairro** (`BAIRRO_GRUPO`),
**Dia da semana** (`DIA_SEMANA`), **Período** (`PERIODO`) e **Tipo de local**
(`TIPO_LOCAL`) — o app aplica o `OneHotEncoder` (118 colunas) e passa o vetor à MLP,
cuja saída sigmoide é a **probabilidade de ser roubo**. O limiar de decisão é 0,5:
acima disso a ocorrência é classificada como Roubo; abaixo, como Furto.

> ⚠️ **Importante:** o modelo classifica o **tipo** de uma ocorrência **já registrada**
> (roubo x furto). Ele **NÃO prevê** se, quando ou onde um crime vai acontecer, e não
> deve ser usado com essa finalidade. É um exercício acadêmico sobre dados públicos.

## Estrutura do projeto

```
gs-mlp-seguranca-sp/
  app.py                 # App Streamlit
  requirements.txt
  README.md
  .gitignore
  models/
    modelo_mlp.keras     # modelo Keras treinado (saída sigmoide = P(roubo))
    encoder.pkl          # OneHotEncoder já ajustado
    categorias.json      # categorias permitidas por atributo (dropdowns)
  notebook/
    GS_CNN.ipynb         # notebook de treino
  scripts/
    smoke_test.py        # teste rápido de carga + 1 predição
```

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app abre no navegador (normalmente em http://localhost:8501).

Para um teste rápido via terminal (carrega os artefatos e faz uma predição):

```bash
python scripts/smoke_test.py
```

> **NOTA sobre versões:** os artefatos foram gerados com **scikit-learn 1.6.1**
> (o `encoder.pkl`) e **Keras 3.13.2 / TensorFlow** (o `modelo_mlp.keras`). O
> `requirements.txt` já fixa `scikit-learn==1.6.1` para coincidir com o encoder e
> evitar *warnings* de versão. O modelo é uma MLP simples (camadas Dense/Dropout) e
> carrega normalmente com o `tensorflow-cpu` fixado. Se aparecer algum aviso de
> versão do TensorFlow/Keras ao carregar, basta atualizar o `tensorflow-cpu` para
> uma versão mais recente.

## Como publicar no Streamlit Community Cloud

1. Envie este repositório para o GitHub (repositório público ou privado).
2. Acesse https://share.streamlit.io e conecte sua conta do GitHub.
3. Clique em **New app**, selecione o repositório, o branch (`main`) e aponte o
   **Main file path** para `app.py`.
4. Clique em **Deploy**. O Streamlit Cloud instala o `requirements.txt`
   automaticamente.

**App online:** _(colar o link aqui após o deploy)_

## Uso de IA generativa

Ferramentas de IA generativa foram usadas como apoio na organização do repositório e
na redação da documentação; a modelagem e a análise dos dados são de autoria das
integrantes.
