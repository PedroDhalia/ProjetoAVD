"""Mapeamento cluster -> classe, matriz de confusão e métricas das duas passadas.

Uso:
    python -m src.parte_comum.metricas --dataset obesity --out ./out/parte_comum/

Lê out/parte_comum/kmeans_{livre,fixo}_rotulos.csv (gerados por kmeans.py);
se não existirem, roda o K-Means antes.

Como funciona:
1. Cada cluster recebe o rótulo da classe real MAJORITÁRIA dentro dele.
2. Tabela de contingência classe real x cluster (retangular: 7 x k).
3. Matriz de confusão classe real x classe mapeada (quadrada 7 x 7). Na
   Passada 1, com k < 7, algumas classes nunca são "previstas" -> colunas zeradas.
4. A partir da matriz:
      acurácia  = soma da diagonal / total
      precisão  = TP / total da coluna (dos que o mapeamento chamou de X, quantos são X)
      recall    = TP / total da linha  (dos que são X, quantos caíram num cluster de X)
   Precisão de classe nunca prevista = indefinida (0/0) -> reportada como "—"
   e tratada como 0 na média macro (convenção conservadora).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, normalized_mutual_info_score, precision_score, recall_score,
)

from src import estilo
from src.parte_comum import dados as D
from src.parte_comum import kmeans as K

# Agrupamentos de classes para discutir a granularidade.
AGRUPAMENTOS = {
    "7 classes originais": {c: c for c in D.CLASSES_PT},
    "6 classes (Sobrepeso I + II)": {
        "Abaixo do peso": "Abaixo do peso", "Peso normal": "Peso normal", "Sobrepeso I": "Sobrepeso",
        "Sobrepeso II": "Sobrepeso", "Obesidade I": "Obesidade I", "Obesidade II": "Obesidade II",
        "Obesidade III": "Obesidade III",
    },
    "4 classes OMS (Abaixo, Normal, Sobrepeso, Obesidade)": {
        "Abaixo do peso": "Abaixo do peso", "Peso normal": "Peso normal", "Sobrepeso I": "Sobrepeso",
        "Sobrepeso II": "Sobrepeso", "Obesidade I": "Obesidade", "Obesidade II": "Obesidade",
        "Obesidade III": "Obesidade",
    },
    "3 classes (Abaixo/Normal, Sobrepeso, Obesidade)": {
        "Abaixo do peso": "Até normal", "Peso normal": "Até normal", "Sobrepeso I": "Sobrepeso",
        "Sobrepeso II": "Sobrepeso", "Obesidade I": "Obesidade", "Obesidade II": "Obesidade",
        "Obesidade III": "Obesidade",
    },
}


# --------------------------------------------------------------------------
# Cálculos
# --------------------------------------------------------------------------
def mapear_clusters(clusters: np.ndarray, y: np.ndarray, ordem: list[str] = D.CLASSES_PT) -> dict[int, str]:
    """cluster -> classe majoritária. Empate: a classe que vem primeiro na ordem."""
    ct = pd.crosstab(clusters, y).reindex(columns=[c for c in ordem if c in set(y)], fill_value=0)
    return {int(g): ct.loc[g].idxmax() for g in ct.index}


def tabela_contingencia(clusters: np.ndarray, y: np.ndarray, ordem: list[str] = D.CLASSES_PT) -> pd.DataFrame:
    ct = pd.crosstab(pd.Series(y, name="Classe real"), pd.Series(clusters, name="Cluster"))
    return ct.reindex([c for c in ordem if c in ct.index])


def matriz_confusao(y: np.ndarray, y_pred: np.ndarray, classes: list[str] = D.CLASSES_PT) -> pd.DataFrame:
    cm = pd.crosstab(pd.Series(y, name="Real"), pd.Series(y_pred, name="Previsto (cluster mapeado)"))
    return cm.reindex(index=classes, columns=classes, fill_value=0)


def metricas_da_matriz(cm: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Calcula as métricas diretamente da matriz (sem sklearn)."""
    m = cm.to_numpy().astype(float)
    tp = np.diag(m)
    total_coluna = m.sum(axis=0)
    total_linha = m.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precisao = np.where(total_coluna > 0, tp / total_coluna, np.nan)
        recall = np.where(total_linha > 0, tp / total_linha, np.nan)
        f1 = np.where((precisao + recall) > 0, 2 * precisao * recall / (precisao + recall), np.nan)
    acuracia = tp.sum() / m.sum()
    por_classe = pd.DataFrame(
        {"classe": cm.index, "suporte_real": total_linha.astype(int), "previstos": total_coluna.astype(int),
         "acertos": tp.astype(int), "precisao": precisao, "recall": recall, "f1": f1}
    )
    return acuracia, por_classe


def conferir_com_sklearn(y, y_pred, acuracia, por_classe) -> bool:
    """Confere o cálculo manual com o scikit-learn."""
    classes = list(por_classe["classe"])
    p = precision_score(y, y_pred, labels=classes, average=None, zero_division=np.nan)
    r = recall_score(y, y_pred, labels=classes, average=None, zero_division=np.nan)
    return (
        np.isclose(accuracy_score(y, y_pred), acuracia)
        and np.allclose(p, por_classe["precisao"], equal_nan=True)
        and np.allclose(r, por_classe["recall"], equal_nan=True)
    )


def avaliar_passada(clusters: np.ndarray, y: np.ndarray) -> dict:
    mapa = mapear_clusters(clusters, y)
    y_pred = np.array([mapa[int(c)] for c in clusters])
    ct = tabela_contingencia(clusters, y)
    cm = matriz_confusao(y, y_pred)
    acuracia, por_classe = metricas_da_matriz(cm)
    ok = conferir_com_sklearn(y, y_pred, acuracia, por_classe)
    return {
        "k": int(len(np.unique(clusters))), "mapa": mapa, "contingencia": ct, "matriz": cm,
        "acuracia": acuracia, "por_classe": por_classe,
        "precisao_macro": float(np.nan_to_num(por_classe["precisao"], nan=0.0).mean()),
        "recall_macro": float(por_classe["recall"].mean()),
        "classes_nunca_previstas": [c for c, n in zip(por_classe["classe"], por_classe["previstos"]) if n == 0],
        "ari": adjusted_rand_score(y, clusters), "nmi": normalized_mutual_info_score(y, clusters),
        "conferido_sklearn": bool(ok),
    }


def granularidade(clusters: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Reavalia o MESMO agrupamento contra versões mais grossas das classes."""
    linhas = []
    for nome, mapa in AGRUPAMENTOS.items():
        y2 = np.array([mapa[c] for c in y])
        m = mapear_clusters(clusters, y2, ordem=list(dict.fromkeys(mapa.values())))
        y_pred = np.array([m[int(c)] for c in clusters])
        linhas.append({
            "agrupamento_de_classes": nome, "n_classes": len(set(y2)),
            "acuracia_mapeada": accuracy_score(y2, y_pred), "ari": adjusted_rand_score(y2, clusters),
            "nmi": normalized_mutual_info_score(y2, clusters),
            "acuracia_chute_majoritario": pd.Series(y2).value_counts(normalize=True).iloc[0],
        })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def fig_contingencia(ct: pd.DataFrame, mapa: dict, titulo: str):
    fig, ax = plt.subplots(figsize=(1.0 * ct.shape[1] + 4, 4.8))
    im = ax.imshow(ct.to_numpy(), cmap=estilo.cmap_sequencial(estilo.SEQ_AZUL), aspect="auto")
    ax.set_xticks(range(ct.shape[1]), [f"Grupo {c}\n→ {mapa[int(c)]}" for c in ct.columns], fontsize=8)
    ax.set_yticks(range(ct.shape[0]), ct.index)
    ax.grid(False)
    vmax = ct.to_numpy().max()
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.iat[i, j]
            maj = mapa[int(ct.columns[j])] == ct.index[i]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9, fontweight="bold" if maj else "normal",
                    color="white" if v > vmax * 0.55 else estilo.TINTA)
    fig.colorbar(im, ax=ax, label="Pessoas")
    ax.set_xlabel("Grupo do K-Means → classe majoritária atribuída")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo)
    return fig


def fig_matriz(cm: pd.DataFrame, titulo: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
    im = ax.imshow(norm.to_numpy(), cmap=estilo.cmap_sequencial(estilo.SEQ_AZUL), vmin=0, vmax=1)
    ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=35, ha="right")
    ax.set_yticks(range(cm.shape[0]), cm.index)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v, p = cm.iat[i, j], norm.iat[i, j]
            ax.text(j, i, f"{v}\n{p:.0%}" if v else "0", ha="center", va="center", fontsize=8,
                    color="white" if p > 0.55 else (estilo.TEXTO_MUDO if v == 0 else estilo.TINTA),
                    fontweight="bold" if i == j else "normal")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("% da linha (recall quando na diagonal)")
    ax.set_xlabel("Classe prevista (classe majoritária do cluster)")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo)
    return fig


def fig_comparacao(res_livre: dict, res_fixo: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    for ax, metrica, nome in [(axes[0], "precisao", "Precisão"), (axes[1], "recall", "Recall")]:
        x = np.arange(len(D.CLASSES_PT))
        for desloc, res, cor, rot in [(-0.2, res_livre, estilo.LARANJA, f"Passada 1 (k={res_livre['k']})"),
                                      (0.2, res_fixo, estilo.AZUL, f"Passada 2 (k={res_fixo['k']})")]:
            v = res["por_classe"][metrica].to_numpy()
            ax.bar(x + desloc, np.nan_to_num(v), width=0.38, color=cor, label=rot)
            for xi, vi in zip(x + desloc, v):
                ax.text(xi, (0 if np.isnan(vi) else vi) + 0.015, "—" if np.isnan(vi) else f"{vi:.2f}",
                        ha="center", fontsize=7)
        ax.set_xticks(x, D.CLASSES_PT, rotation=30, ha="right")
        ax.set_title(nome)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Classe")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Valor da métrica (0 a 1)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Precisão e recall por classe nas duas passadas (— = classe nunca prevista, precisão indefinida)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def _carregar_rotulos(dataset: str, modo: str, out: Path) -> pd.DataFrame:
    caminho = out / f"kmeans_{modo}_rotulos.csv"
    if not caminho.exists():
        K.executar(dataset, modo, out)
    return pd.read_csv(caminho)


def executar(dataset: str, out: Path) -> dict:
    estilo.aplicar_estilo_mpl()
    pasta = out / "metricas"
    pasta.mkdir(parents=True, exist_ok=True)
    resultados = {}
    for modo, nome in [("livre", "Passada 1 — k livre"), ("fixo", "Passada 2 — k fixo")]:
        r = _carregar_rotulos(dataset, modo, out)
        res = avaliar_passada(r["cluster"].to_numpy(), r["classe"].to_numpy())
        resultados[modo] = res
        res["contingencia"].to_csv(pasta / f"{modo}_contingencia.csv")
        res["matriz"].to_csv(pasta / f"{modo}_matriz_confusao.csv")
        res["por_classe"].to_csv(pasta / f"{modo}_metricas_por_classe.csv", index=False)
        granularidade(r["cluster"].to_numpy(), r["classe"].to_numpy()).to_csv(
            pasta / f"{modo}_granularidade.csv", index=False)
        titulo = f"{nome} (k = {res['k']})"
        for arq, fig in {
            f"{modo}_contingencia": fig_contingencia(res["contingencia"], res["mapa"],
                                                     f"{titulo}: tabela de contingência classe real x grupo"),
            f"{modo}_matriz_confusao": fig_matriz(res["matriz"], f"{titulo}: matriz de confusão (acurácia "
                                                                 f"{res['acuracia']:.1%})"),
        }.items():
            fig.savefig(pasta / f"{arq}.png")
            plt.close(fig)

    fig = fig_comparacao(resultados["livre"], resultados["fixo"])
    fig.savefig(pasta / "comparacao_precisao_recall.png")
    plt.close(fig)

    resumo = pd.DataFrame([
        {"passada": "1 — k livre (cotovelo)" if m == "livre" else "2 — k fixo (nº de classes)", "k": r["k"],
         "acuracia": r["acuracia"], "precisao_macro": r["precisao_macro"], "recall_macro": r["recall_macro"],
         "ari": r["ari"], "nmi": r["nmi"], "classes_nunca_previstas": ", ".join(r["classes_nunca_previstas"]) or "—",
         "mapeamento": "; ".join(f"G{g}→{c}" for g, c in r["mapa"].items())}
        for m, r in resultados.items()
    ])
    resumo.to_csv(pasta / "resumo_passadas.csv", index=False)
    json_out = {m: {k: v for k, v in r.items() if k in ("k", "acuracia", "precisao_macro", "recall_macro", "ari", "nmi",
                                                         "classes_nunca_previstas", "conferido_sklearn")}
                | {"mapa": {str(g): c for g, c in r["mapa"].items()}} for m, r in resultados.items()}
    (pasta / "resumo_passadas.json").write_text(json.dumps(json_out, indent=2, ensure_ascii=False, default=float),
                                               encoding="utf-8")
    return resultados


def main() -> None:
    matplotlib.use("Agg")  # só na linha de comando: salva PNGs sem abrir janelas
    parser = argparse.ArgumentParser(description="Métricas das passadas do K-Means (Parte Comum)")
    parser.add_argument("--dataset", default="obesity", choices=list(D.CAMINHOS))
    parser.add_argument("--out", default="./out/parte_comum/")
    args = parser.parse_args()
    res = executar(args.dataset, Path(args.out))
    for modo, r in res.items():
        print(f"[{modo}] k={r['k']} acurácia={r['acuracia']:.3f} precisão macro={r['precisao_macro']:.3f} "
              f"recall macro={r['recall_macro']:.3f} ARI={r['ari']:.3f} | conferido c/ sklearn: {r['conferido_sklearn']}")
        print("   mapeamento:", r["mapa"])


if __name__ == "__main__":
    main()
