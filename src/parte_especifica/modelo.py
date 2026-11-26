"""Modelo de classificação do risco de cancelamento.

Objetivo: responder à 2ª parte da pergunta de pesquisa ("o risco é previsível
no momento da reserva?") e alimentar o simulador do dashboard.

Decisões importantes:
* Só entram variáveis CONHECIDAS NO MOMENTO DA RESERVA. Ficam de fora:
  - reservation_status / reservation_status_date: são o próprio desfecho;
  - assigned_room_type: definido no check-in (quem cancelou nunca recebe quarto);
  - booking_changes: acontece ao longo do tempo, depois da reserva;
  - required_car_parking_spaces: 0% de cancelamento entre quem pediu vaga, sinal
    de que a informação é registrada na chegada -> vazamento;
  - arrival_date_year, agent, company (IDs): não generalizam.
* Divisão TEMPORAL: treino com chegadas até dez/2016 e teste com chegadas de
  2017. Simula o uso real (prever o futuro) e evita que linhas repetidas de um
  mesmo grupo caiam ao mesmo tempo no treino e no teste.
* HistGradientBoostingClassifier (scikit-learn): lida com variáveis
  categóricas de forma nativa e treina em poucos segundos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

NUMERICAS = [
    "lead_time", "mes_num", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "days_in_waiting_list", "adr", "total_of_special_requests", "tem_agente", "tem_empresa",
]
CATEGORICAS = [
    "hotel", "meal", "pais_grupo", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
]
FEATURES = NUMERICAS + CATEGORICAS
ALVO = "is_canceled"
DATA_CORTE = pd.Timestamp("2017-01-01")

ROTULOS_FEATURES = {
    "lead_time": "Antecedência", "mes_num": "Mês de chegada", "stays_in_weekend_nights": "Noites fim de semana",
    "stays_in_week_nights": "Noites de semana", "adults": "Adultos", "children": "Crianças", "babies": "Bebês",
    "is_repeated_guest": "Hóspede recorrente", "previous_cancellations": "Cancelamentos anteriores",
    "previous_bookings_not_canceled": "Reservas anteriores mantidas", "days_in_waiting_list": "Dias em lista de espera",
    "adr": "Diária (ADR)", "total_of_special_requests": "Pedidos especiais", "tem_agente": "Via agente",
    "tem_empresa": "Via empresa", "hotel": "Hotel", "meal": "Refeição", "pais_grupo": "País de origem",
    "market_segment": "Segmento de mercado", "distribution_channel": "Canal de distribuição",
    "reserved_room_type": "Tipo de quarto", "deposit_type": "Tipo de depósito", "customer_type": "Tipo de cliente",
}


@dataclass
class ResultadoModelo:
    modelo: HistGradientBoostingClassifier
    categorias: dict[str, list]
    metricas: dict
    matriz_confusao: pd.DataFrame
    roc: pd.DataFrame
    importancia: pd.DataFrame | None
    n_treino: int
    n_teste: int


def preparar_X(df: pd.DataFrame, categorias: dict[str, list]) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATEGORICAS:
        X[c] = pd.Categorical(X[c].astype(str), categories=categorias[c])
    for c in NUMERICAS:
        X[c] = X[c].astype(float)
    return X


def treinar(df: pd.DataFrame, calcular_importancia: bool = True, semente: int = 42) -> ResultadoModelo:
    categorias = {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAS}
    treino = df[df["data_chegada"] < DATA_CORTE]
    teste = df[df["data_chegada"] >= DATA_CORTE]

    X_tr, y_tr = preparar_X(treino, categorias), treino[ALVO].to_numpy()
    X_te, y_te = preparar_X(teste, categorias), teste[ALVO].to_numpy()

    modelo = HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=300, learning_rate=0.08,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=semente,
    )
    modelo.fit(X_tr, y_tr)

    proba = modelo.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    taxa_base = y_tr.mean()
    metricas = {
        "auc": roc_auc_score(y_te, proba),
        "acuracia": accuracy_score(y_te, pred),
        "precisao": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "acuracia_baseline": max(1 - y_te.mean(), y_te.mean()),  # sempre prever a classe majoritária
        "taxa_cancelamento_treino": taxa_base,
        "taxa_cancelamento_teste": y_te.mean(),
    }
    cm = pd.DataFrame(
        confusion_matrix(y_te, pred),
        index=["Real: não cancelou", "Real: cancelou"],
        columns=["Previsto: não cancela", "Previsto: cancela"],
    )
    fpr, tpr, _ = roc_curve(y_te, proba)
    passo = max(1, len(fpr) // 400)
    roc = pd.DataFrame({"fpr": fpr[::passo], "tpr": tpr[::passo]})

    importancia = None
    if calcular_importancia:
        amostra = X_te.sample(n=min(8000, len(X_te)), random_state=semente)
        r = permutation_importance(
            modelo, amostra, y_te[X_te.index.get_indexer(amostra.index)],
            scoring="roc_auc", n_repeats=5, random_state=semente, n_jobs=1,
        )
        importancia = (
            pd.DataFrame({"variavel": amostra.columns, "queda_auc": r.importances_mean, "desvio": r.importances_std})
            .assign(rotulo=lambda d: d["variavel"].map(ROTULOS_FEATURES))
            .sort_values("queda_auc", ascending=False)
            .reset_index(drop=True)
        )

    return ResultadoModelo(modelo, categorias, metricas, cm, roc, importancia, len(treino), len(teste))


def reserva_tipica(df: pd.DataFrame) -> dict:
    """Valores de referência (mediana / moda) para as variáveis que o simulador não expõe."""
    base = {}
    for c in NUMERICAS:
        base[c] = float(df[c].median())
    for c in CATEGORICAS:
        base[c] = str(df[c].mode().iat[0])
    return base


def prever(resultado: ResultadoModelo, reservas: pd.DataFrame) -> np.ndarray:
    X = preparar_X(reservas, resultado.categorias)
    return resultado.modelo.predict_proba(X)[:, 1]
