"""K-Means em duas passadas sobre o dataset Obesity.

Uso:
    python -m src.parte_comum.kmeans --dataset obesity --modo livre --out ./out/parte_comum/
    python -m src.parte_comum.kmeans --dataset obesity --modo fixo  --out ./out/parte_comum/

* Passada 1 (livre): roda k = 1..10, desenha o Método do Cotovelo e escolhe k
  automaticamente pelo ponto de maior distância à reta que liga o primeiro e
  o último ponto da curva (critério "kneedle"). k = 1 entra só como linha de
  base da curva (inércia total).
* Passada 2 (fixo): k = número real de classes (7).

O K-Means NÃO vê a classe: ela só é usada depois, em metricas.py, para
comparar os grupos encontrados com os rótulos conhecidos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import estilo
from src.parte_comum import dados as D

SEMENTE = 42
N_INIT = 20
KS = list(range(1, 11))
CORES_CLASSE = dict(zip(D.CLASSES_PT, estilo.CORES_CLASSE_OBESIDADE))


# --------------------------------------------------------------------------
# Cálculos
# --------------------------------------------------------------------------
def ajustar(Xs: np.ndarray, k: int, semente: int = SEMENTE) -> KMeans:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=semente).fit(Xs)


def curva_cotovelo(Xs: np.ndarray, ks: list[int] = KS, semente: int = SEMENTE) -> pd.DataFrame:
    linhas = []
    for k in ks:
        km = ajustar(Xs, k, semente)
        sil = silhouette_score(Xs, km.labels_) if k > 1 else np.nan
        linhas.append({"k": k, "inercia": km.inertia_, "silhueta": sil})
    curva = pd.DataFrame(linhas)
    curva["queda_pct"] = -100 * curva["inercia"].pct_change()
    curva["distancia_corda"] = _distancia_corda(curva["k"].to_numpy(), curva["inercia"].to_numpy())
    return curva


def _distancia_corda(ks: np.ndarray, inercia: np.ndarray) -> np.ndarray:
    """Normaliza a curva para [0,1]x[0,1] e mede a distância de cada ponto à reta
    que liga o primeiro ao último ponto. O cotovelo é o ponto mais distante."""
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (inercia - inercia.min()) / (inercia.max() - inercia.min())
    return np.abs(y - (1 - x)) / np.sqrt(2)


def escolher_k(curva: pd.DataFrame) -> int:
    return int(curva.loc[curva["distancia_corda"].idxmax(), "k"])


def estabilidade_cotovelo(Xs: np.ndarray, sementes=(0, 7, 42, 123, 2026)) -> pd.DataFrame:
    """Repete a escolha do cotovelo com outras sementes, com e sem k = 1."""
    linhas = []
    for s in sementes:
        c = curva_cotovelo(Xs, KS, s)
        c2 = c[c["k"] >= 2].copy()
        c2["distancia_corda"] = _distancia_corda(c2["k"].to_numpy(), c2["inercia"].to_numpy())
        linhas.append({"semente": s, "k_cotovelo_1a10": escolher_k(c), "k_cotovelo_2a10": escolher_k(c2)})
    return pd.DataFrame(linhas)


def projecao_pca(Xs: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=SEMENTE)
    return pca.fit_transform(Xs), pca


def rotulos_por_k(Xs: np.ndarray, ks=range(2, 11)) -> pd.DataFrame:
    """Rótulos do K-Means para vários k (usado na animação do dashboard)."""
    return pd.DataFrame({k: ajustar(Xs, k).labels_ for k in ks})


def centroides_originais(km: KMeans, scaler: StandardScaler, features: list[str]) -> pd.DataFrame:
    """Centróides de volta à escala original (para interpretar os grupos)."""
    return pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=features).rename_axis("cluster")


def sensibilidade_features(df: pd.DataFrame, k: int = 7) -> pd.DataFrame:
    """Mesma passada (k fixo) com outros conjuntos de variáveis.

    Não é usada para ESCOLHER as variáveis (isso seria usar a classe para
    ajustar um método não supervisionado). Serve à discussão crítica: mostra
    o quanto a classe depende de Altura/Peso e o quanto os hábitos ajudam.
    """
    X15 = D.montar_matriz(df)
    mtrans = pd.get_dummies(df["MTRANS"], prefix="MTRANS").astype(float)
    conjuntos = {
        "15 variáveis (usado)": X15,
        "15 + MTRANS one-hot (20)": pd.concat([X15, mtrans], axis=1),
        "Só hábitos e perfil (sem Altura/Peso)": X15.drop(columns=["Height", "Weight"]),
        "Só Altura e Peso": X15[["Height", "Weight"]],
    }
    y = df["classe"].astype(str).to_numpy()
    linhas = []
    for nome, X in conjuntos.items():
        Xs = StandardScaler().fit_transform(X)
        km = ajustar(Xs, k)
        ct = pd.crosstab(km.labels_, y)
        linhas.append({
            "conjunto": nome, "n_variaveis": X.shape[1],
            "acuracia_mapeada": ct.max(axis=1).sum() / len(y),
            "ari": adjusted_rand_score(y, km.labels_),
            "silhueta": silhouette_score(Xs, km.labels_),
        })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def fig_cotovelo(curva: pd.DataFrame, k_escolhido: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.plot(curva["k"], curva["inercia"], color=estilo.AZUL, marker="o", linewidth=2)
    ax1.plot([curva["k"].iat[0], curva["k"].iat[-1]], [curva["inercia"].iat[0], curva["inercia"].iat[-1]],
             color=estilo.TEXTO_MUDO, linestyle=":", linewidth=1, label="Reta entre o 1º e o último ponto")
    linha = curva[curva["k"] == k_escolhido].iloc[0]
    ax1.scatter([k_escolhido], [linha["inercia"]], s=180, facecolors="none", edgecolors=estilo.LARANJA,
                linewidths=2.5, zorder=5, label=f"Cotovelo escolhido: k = {k_escolhido}")
    ax1.axvline(7, color=estilo.TEXTO_MUDO, linestyle="--", linewidth=1)
    ax1.text(7.1, curva["inercia"].min(), "nº real de classes (7)", fontsize=8.5,
             color=estilo.TEXTO_SECUNDARIO, va="bottom")
    ax1.set_xticks(curva["k"])
    ax1.set_xlabel("Número de grupos (k)")
    ax1.set_ylabel("Inércia (soma das distâncias² ao centróide)")
    ax1.set_title("Método do Cotovelo")
    ax1.legend(loc="upper right")

    c2 = curva[curva["k"] >= 2]
    cores = [estilo.LARANJA if k == k_escolhido else estilo.CINZA_CONTEXTO for k in c2["k"]]
    ax2.bar(c2["k"], c2["queda_pct"], color=cores, width=0.6)
    for k, v in zip(c2["k"], c2["queda_pct"]):
        ax2.text(k, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
    ax2.set_xticks(c2["k"])
    ax2.set_xlabel("Número de grupos (k)")
    ax2.set_ylabel("Redução da inércia vs. k−1 (%)")
    ax2.set_title("Ganho marginal de cada grupo a mais")
    ax2.grid(axis="x", visible=False)
    fig.suptitle(f"Passada 1 — curva suave, sem cotovelo nítido; critério da maior distância à reta indica k = {k_escolhido}",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def fig_silhueta(curva: pd.DataFrame, k_escolhido: int):
    c = curva[curva["k"] >= 2]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cores = [estilo.LARANJA if k == k_escolhido else (estilo.TEXTO_MUDO if k == 7 else estilo.AZUL) for k in c["k"]]
    ax.bar(c["k"], c["silhueta"], color=cores, width=0.6)
    for k, v in zip(c["k"], c["silhueta"]):
        ax.text(k, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(c["k"])
    ax.set_ylim(0, max(0.3, c["silhueta"].max() * 1.2))
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Coeficiente de silhueta médio")
    ax.set_title("Silhueta (complementar): todos os valores < 0,25 → estrutura de grupos fraca")
    ax.grid(axis="x", visible=False)
    return fig


def fig_clusters_pca(coords: np.ndarray, labels: np.ndarray, y: np.ndarray, pca: PCA, titulo: str,
                     centros: np.ndarray | None = None):
    k = len(np.unique(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True, sharey=True)
    for c in range(k):
        m = labels == c
        ax1.scatter(coords[m, 0], coords[m, 1], s=9, color=estilo.CATEGORICA_ESTENDIDA[c], label=f"Grupo {c}",
                    alpha=0.75, edgecolors="none")
    if centros is not None:
        ax1.scatter(centros[:, 0], centros[:, 1], marker="X", s=160, color=estilo.TINTA, edgecolors="white",
                    linewidths=1.5, label="Centróide", zorder=5)
    ax1.set_title(f"Grupos do K-Means (k = {k})")
    leg = ax1.legend(loc="upper right", fontsize=8, ncol=2 if k > 4 else 1)
    for h in leg.legend_handles:
        h.set_sizes([40])
    for classe in D.CLASSES_PT:
        m = y == classe
        ax2.scatter(coords[m, 0], coords[m, 1], s=9, color=CORES_CLASSE[classe], label=classe, alpha=0.75,
                    edgecolors="none")
    ax2.set_title("Classes reais (o K-Means não viu estes rótulos)")
    leg = ax2.legend(loc="upper right", fontsize=8, ncol=2)
    for h in leg.legend_handles:
        h.set_sizes([40])
    v = pca.explained_variance_ratio_
    for ax in (ax1, ax2):
        ax.set_xlabel(f"Componente principal 1 ({v[0]:.0%} da variância)")
    ax1.set_ylabel(f"Componente principal 2 ({v[1]:.0%} da variância)")
    fig.suptitle(titulo, x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.text(0.0, -0.02, f"Projeção PCA das 15 variáveis padronizadas: os 2 eixos explicam só {v.sum():.0%} da variância, "
             "então grupos que parecem sobrepostos aqui podem estar separados em outras dimensões.",
             fontsize=8, color=estilo.TEXTO_MUDO)
    return fig


def fig_perfil_centroides(km: KMeans, features: list[str], titulo: str):
    z = pd.DataFrame(km.cluster_centers_, columns=[D.ROTULO_CURTO[f] for f in features])
    z.index = [f"Grupo {i}" for i in range(len(z))]
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(z) + 1.8))
    lim = max(1.5, float(np.abs(z.to_numpy()).max()))
    im = ax.imshow(z.to_numpy(), cmap=estilo.cmap_divergente(), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(z)), z.index)
    ax.grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z.iat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > lim * 0.6 else estilo.TINTA)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Centróide em z-score\n(0 = média geral)")
    ax.set_xlabel("Variável")
    ax.set_title(titulo)
    return fig


def fig_composicao(labels: np.ndarray, y: np.ndarray, titulo: str):
    tab = pd.crosstab(labels, y, normalize="index")[D.CLASSES_PT]
    n = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(tab) + 1.8))
    esquerda = np.zeros(len(tab))
    for classe in D.CLASSES_PT:
        ax.barh([f"Grupo {g} (n={n[g]})" for g in tab.index], tab[classe], left=esquerda,
                color=CORES_CLASSE[classe], label=classe, edgecolor="white", linewidth=1, height=0.7)
        esquerda += tab[classe].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("% das pessoas do grupo")
    ax.set_ylabel("Grupo do K-Means")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=7, fontsize=8)
    ax.set_title(titulo)
    ax.grid(False)
    return fig


# --------------------------------------------------------------------------
def executar(dataset: str, modo: str, out: Path) -> dict:
    estilo.aplicar_estilo_mpl()
    pasta = out / "kmeans"
    pasta.mkdir(parents=True, exist_ok=True)
    prep = D.preparar(dataset)
    Xs, y, df = prep["Xs"], prep["y"], prep["df"]
    features = list(prep["X"].columns)
    coords, pca = projecao_pca(Xs)

    resumo: dict = {"dataset": dataset, "modo": modo, "n": int(len(df)), "features": features}
    if modo == "livre":
        curva = curva_cotovelo(Xs)
        curva.to_csv(out / "kmeans_cotovelo.csv", index=False)
        k = escolher_k(curva)
        estab = estabilidade_cotovelo(Xs)
        estab.to_csv(out / "kmeans_estabilidade_cotovelo.csv", index=False)
        for nome, fig in {"livre_cotovelo": fig_cotovelo(curva, k), "livre_silhueta": fig_silhueta(curva, k)}.items():
            fig.savefig(pasta / f"{nome}.png")
            plt.close(fig)
        resumo["curva"] = curva.round(4).to_dict("records")
        resumo["estabilidade"] = estab.to_dict("records")
        titulo = f"Passada 1 — k livre (Cotovelo): k = {k}"
    elif modo == "fixo":
        k = int(len(np.unique(y)))
        titulo = f"Passada 2 — k fixo no número real de classes: k = {k}"
        sensibilidade_features(df, k).to_csv(out / "kmeans_sensibilidade_variaveis.csv", index=False)
    else:
        raise ValueError("modo deve ser 'livre' ou 'fixo'")

    km = ajustar(Xs, k)
    resumo.update({"k": k, "inercia": float(km.inertia_), "silhueta": float(silhouette_score(Xs, km.labels_))})
    pd.DataFrame({"indice": df.index, "cluster": km.labels_, "classe": y,
                  "pc1": coords[:, 0], "pc2": coords[:, 1]}).to_csv(out / f"kmeans_{modo}_rotulos.csv", index=False)
    centroides_originais(km, prep["scaler"], features).to_csv(out / f"kmeans_{modo}_centroides.csv")
    pd.DataFrame(km.cluster_centers_, columns=features).to_csv(out / f"kmeans_{modo}_centroides_z.csv")

    centros_pca = pca.transform(km.cluster_centers_)
    figs = {
        f"{modo}_pca": fig_clusters_pca(coords, km.labels_, y, pca, titulo, centros_pca),
        f"{modo}_centroides": fig_perfil_centroides(km, features, f"{titulo} — perfil dos centróides"),
        f"{modo}_composicao": fig_composicao(km.labels_, y, f"{titulo} — composição de classes reais em cada grupo"),
    }
    for nome, fig in figs.items():
        fig.savefig(pasta / f"{nome}.png")
        plt.close(fig)
    (out / f"kmeans_{modo}_resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=float),
                                                     encoding="utf-8")
    return resumo


def main() -> None:
    matplotlib.use("Agg")  # só na linha de comando: salva PNGs sem abrir janelas
    parser = argparse.ArgumentParser(description="K-Means (Parte Comum)")
    parser.add_argument("--dataset", default="obesity", choices=list(D.CAMINHOS))
    parser.add_argument("--modo", required=True, choices=["livre", "fixo"])
    parser.add_argument("--out", default="./out/parte_comum/")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r = executar(args.dataset, args.modo, out)
    print(f"Passada '{args.modo}': k = {r['k']} | inércia = {r['inercia']:.1f} | silhueta = {r['silhueta']:.3f}")


if __name__ == "__main__":
    main()
