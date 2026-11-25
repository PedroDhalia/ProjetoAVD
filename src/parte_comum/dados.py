"""Carga e pré-processamento do dataset Obesity (UCI 544).

"Estimation of obesity levels based on eating habits and physical condition"
(Palechor & de la Hoz Manotas, 2019). 2.111 pessoas do México, Peru e Colômbia,
17 atributos. Segundo os autores, 77% dos registros foram gerados
sinteticamente com SMOTE (Weka) e só 23% vêm diretamente do questionário.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
CAMINHOS = {
    "obesity": RAIZ / "data" / "parte_comum" / "ObesityDataSet_raw_and_data_sinthetic.csv",
}
ALVO = "NObeyesdad"

ORDEM_CLASSES = [
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]
ROTULO_CLASSE = {
    "Insufficient_Weight": "Abaixo do peso",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidade I",
    "Obesity_Type_II": "Obesidade II",
    "Obesity_Type_III": "Obesidade III",
}
CLASSES_PT = [ROTULO_CLASSE[c] for c in ORDEM_CLASSES]

# Dicionário de variáveis (pilar estrutural)
DICIONARIO = {
    "Gender": ("Gênero", "Qualitativa nominal (binária)"),
    "Age": ("Idade (anos)", "Quantitativa contínua"),
    "Height": ("Altura (m)", "Quantitativa contínua"),
    "Weight": ("Peso (kg)", "Quantitativa contínua"),
    "family_history_with_overweight": ("Histórico familiar de sobrepeso", "Qualitativa nominal (binária)"),
    "FAVC": ("Come alimentos calóricos com frequência", "Qualitativa nominal (binária)"),
    "FCVC": ("Frequência de consumo de vegetais (1=nunca, 2=às vezes, 3=sempre)", "Quantitativa ordinal (escala 1–3)"),
    "NCP": ("Número de refeições principais por dia (1 a 4)", "Quantitativa ordinal (escala 1–4)"),
    "CAEC": ("Come entre as refeições", "Qualitativa ordinal (no < Sometimes < Frequently < Always)"),
    "SMOKE": ("Fuma", "Qualitativa nominal (binária)"),
    "CH2O": ("Água por dia (1=<1L, 2=1–2L, 3=>2L)", "Quantitativa ordinal (escala 1–3)"),
    "SCC": ("Monitora as calorias que consome", "Qualitativa nominal (binária)"),
    "FAF": ("Frequência de atividade física (0=nenhuma ... 3=4–5 dias/semana)", "Quantitativa ordinal (escala 0–3)"),
    "TUE": ("Tempo usando dispositivos eletrônicos (0=0–2h, 1=3–5h, 2=>5h)", "Quantitativa ordinal (escala 0–2)"),
    "CALC": ("Consumo de álcool", "Qualitativa ordinal (no < Sometimes < Frequently < Always)"),
    "MTRANS": ("Meio de transporte habitual", "Qualitativa nominal (5 categorias)"),
    "NObeyesdad": ("Nível de obesidade (CLASSE)", "Qualitativa ordinal (7 classes)"),
}

CONTINUAS = ["Age", "Height", "Weight"]
ORDINAIS_QUESTIONARIO = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
NUMERICAS = CONTINUAS + ORDINAIS_QUESTIONARIO
BINARIAS = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
ORDINAIS_CATEGORICAS = ["CAEC", "CALC"]
NOMINAIS = ["MTRANS"]
CATEGORICAS = BINARIAS + ORDINAIS_CATEGORICAS + NOMINAIS

ESCALA_FREQUENCIA = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
CODIFICACAO_BINARIA = {
    "Gender": {"Female": 0, "Male": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1},
    "SMOKE": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},
}

# Variáveis que entram no K-Means (15). MTRANS fica de fora — ver justificativa em montar_matriz().
FEATURES_KMEANS = NUMERICAS + BINARIAS + ORDINAIS_CATEGORICAS

ROTULO_CURTO = {
    "Age": "Idade", "Height": "Altura", "Weight": "Peso", "FCVC": "Vegetais", "NCP": "Refeições/dia",
    "CH2O": "Água", "FAF": "Ativ. física", "TUE": "Telas", "Gender": "Gênero (masc.)",
    "family_history_with_overweight": "Hist. familiar", "FAVC": "Alim. calórico", "SMOKE": "Fuma",
    "SCC": "Monitora calorias", "CAEC": "Belisca", "CALC": "Álcool", "IMC": "IMC", "MTRANS": "Transporte",
}


def carregar_bruto(dataset: str = "obesity") -> pd.DataFrame:
    if dataset not in CAMINHOS:
        raise ValueError(f"Dataset '{dataset}' não suportado. Use: {list(CAMINHOS)}")
    return pd.read_csv(CAMINHOS[dataset])


def limpar(bruto: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tratamentos documentados. Devolve (df_limpo, log)."""
    df = bruto.copy()
    log = []

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log.append({
        "etapa": "Duplicatas", "linhas_afetadas": n_dup,
        "diagnostico": "24 linhas idênticas a outra em todas as 17 colunas, todas na parte coletada pelo "
                       "questionário web (valores inteiros). Coincidir em 17 respostas é improvável: o mais "
                       "provável é a mesma pessoa ter enviado o formulário mais de uma vez.",
        "tratamento": "Removidas (1,1% da base). Cópias dariam peso dobrado a esses pontos no cálculo dos centróides.",
    })

    n_frac = int((df[ORDINAIS_QUESTIONARIO] != df[ORDINAIS_QUESTIONARIO].round()).any(axis=1).sum())
    df[ORDINAIS_QUESTIONARIO] = df[ORDINAIS_QUESTIONARIO].round().astype(int)
    log.append({
        "etapa": "Inconsistência (dados sintéticos)", "linhas_afetadas": n_frac,
        "diagnostico": "FCVC, NCP, CH2O, FAF e TUE são respostas de questionário em escala inteira, mas aparecem "
                       "com decimais (ex.: FCVC = 2,45) em ~60% das linhas — efeito da interpolação do SMOTE.",
        "tratamento": "Arredondadas para o inteiro mais próximo, restaurando o significado da escala.",
    })

    log.append({
        "etapa": "Faltantes", "linhas_afetadas": int(df.isna().sum().sum()),
        "diagnostico": "Nenhum valor ausente em nenhuma coluna.",
        "tratamento": "Nenhum tratamento necessário.",
    })
    log.append({
        "etapa": "Outliers", "linhas_afetadas": 0,
        "diagnostico": "Regra de Tukey: 167 idades acima de ~35 anos, 1 altura (1,98 m) e 1 peso (173 kg). Todos são "
                       "plausíveis (idade 14–61, altura 1,45–1,98 m, peso 39–173 kg). NCP e CH2O também 'acusam' "
                       "centenas de outliers, mas só porque o IQR deles é 0 (escala discreta concentrada em um valor): "
                       "a regra de Tukey não se aplica bem a escalas de 3–4 pontos.",
        "tratamento": "Mantidos: são pessoas reais/plausíveis e as classes extremas (Obesidade III, Abaixo do peso) "
                      "dependem justamente desses valores. A padronização reduz seu efeito de escala.",
    })

    df["IMC"] = df["Weight"] / df["Height"] ** 2
    df["classe"] = pd.Categorical(df[ALVO].map(ROTULO_CLASSE), categories=CLASSES_PT, ordered=True)
    df["classe_codigo"] = df[ALVO].map({c: i for i, c in enumerate(ORDEM_CLASSES)})
    return df, pd.DataFrame(log)


def montar_matriz(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica as variáveis para o K-Means (antes da padronização).

    * Numéricas (8): usadas como estão (ordinais já arredondadas).
    * Binárias (5): 0/1.
    * Ordinais categóricas CAEC e CALC: 0 (no) a 3 (Always) — preserva a ordem.
    * MTRANS (nominal, 5 categorias) NÃO entra. Não tem ordem, então não pode
      virar um número (0..4 inventaria distâncias falsas). A alternativa,
      one-hot, transforma UMA variável em CINCO dimensões, dando ao meio de
      transporte ~5x o peso das outras variáveis na distância euclidiana.
      Além disso, as categorias raras (Moto = 11 pessoas, Bicicleta = 7) viram,
      após a padronização, colunas com z de até ~17.
      Limitação registrada: o mesmo efeito de "categoria rara" aparece em SMOKE
      (2% fumam) e SCC (5%) — a Passada 2 dedica um grupo a cada uma (ver
      relatório). Elas foram mantidas por serem binárias legítimas (1 dimensão).
    """
    X = pd.DataFrame(index=df.index)
    for c in NUMERICAS:
        X[c] = df[c].astype(float)
    for c, mapa in CODIFICACAO_BINARIA.items():
        X[c] = df[c].map(mapa).astype(float)
    for c in ORDINAIS_CATEGORICAS:
        X[c] = df[c].map(ESCALA_FREQUENCIA).astype(float)
    return X[FEATURES_KMEANS]


def padronizar(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """z = (x - média) / desvio. Todas as variáveis passam a ter média 0 e desvio 1."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def preparar(dataset: str = "obesity") -> dict:
    """Atalho usado pelos scripts e notebooks."""
    bruto = carregar_bruto(dataset)
    df, log = limpar(bruto)
    X = montar_matriz(df)
    Xs, scaler = padronizar(X)
    return {"bruto": bruto, "df": df, "log": log, "X": X, "Xs": Xs, "scaler": scaler,
            "y": df["classe"].astype(str).to_numpy()}


def tabela_estrutural(bruto: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for col in bruto.columns:
        desc, tipo = DICIONARIO[col]
        s = bruto[col]
        linhas.append({
            "coluna": col, "descricao": desc, "tipo_estatistico": tipo, "dtype_pandas": str(s.dtype),
            "valores_unicos": int(s.nunique()), "faltantes": int(s.isna().sum()),
            "exemplos": ", ".join(map(str, s.drop_duplicates().head(3).tolist())),
        })
    return pd.DataFrame(linhas)
