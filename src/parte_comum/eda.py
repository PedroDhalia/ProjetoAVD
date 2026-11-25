"""EDA da Parte Comum (Obesity) — pilares estrutural, distribucional, relacional e outliers.

Uso:
    python -m src.parte_comum.eda --dataset obesity --out ./out/parte_comum/

As funções fig_* devolvem figuras matplotlib para serem reaproveitadas no
notebook notebooks/parte_comum/eda.ipynb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from src import estilo
from src.parte_comum import dados as D

CORES_CLASSE = dict(zip(D.CLASSES_PT, estilo.CORES_CLASSE_OBESIDADE))
FONTE = "Fonte: UCI 544 — Estimation of obesity levels (Palechor & de la Hoz Manotas, 2019)."


def _rodape(fig, texto=FONTE):
    fig.text(0.0, -0.02, texto, fontsize=8, color=estilo.TEXTO_MUDO, ha="left")


# --------------------------------------------------------------------------
# Distribucional
# --------------------------------------------------------------------------
def fig_classes(df: pd.DataFrame):
    cont = df["classe"].value_counts().reindex(D.CLASSES_PT)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    barras = ax.bar(cont.index, cont.values, color=[CORES_CLASSE[c] for c in cont.index], width=0.65)
    for b, v in zip(barras, cont.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}\n({v / len(df):.1%})", ha="center", fontsize=8.5)
    ax.set_ylim(0, cont.max() * 1.25)
    ax.set_ylabel("Número de pessoas")
    ax.set_xlabel("Classe (nível de obesidade, em ordem de IMC)")
    ax.set_title("Distribuição da variável de classe — as 7 classes são praticamente balanceadas")
    ax.grid(axis="x", visible=False)
    _rodape(fig, FONTE + " Balanceamento produzido pelo SMOTE.")
    return fig


def fig_numericas(df: pd.DataFrame):
    variaveis = D.CONTINUAS + ["IMC"] + D.ORDINAIS_QUESTIONARIO
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, c in zip(axes.flat, variaveis):
        if c in D.ORDINAIS_QUESTIONARIO:
            cont = df[c].value_counts().sort_index()
            ax.bar(cont.index.astype(str), cont.values, color=estilo.AZUL, width=0.6)
            ax.set_xlabel(f"{D.ROTULO_CURTO[c]} ({c}) — resposta na escala")
            ax.grid(axis="x", visible=False)
        else:
            ax.hist(df[c], bins=30, color=estilo.AZUL, edgecolor="white", linewidth=0.6)
            ax.axvline(df[c].median(), color=estilo.LARANJA, linewidth=1.5, linestyle="--")
            unidade = {"Age": "anos", "Height": "m", "Weight": "kg", "IMC": "kg/m²"}[c]
            ax.set_xlabel(f"{D.ROTULO_CURTO[c]} ({unidade}) — tracejado = mediana {df[c].median():.2f}")
        ax.set_ylabel("Pessoas")
        assim = df[c].skew()
        ax.set_title(f"{D.ROTULO_CURTO[c]}  ·  assimetria = {assim:+.2f}", fontsize=10)
    fig.suptitle("Distribuição das variáveis numéricas (ordinais já arredondadas)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _rodape(fig)
    return fig


def fig_categoricas(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    ordem_freq = ["no", "Sometimes", "Frequently", "Always"]
    for ax, c in zip(axes.flat, D.CATEGORICAS):
        cont = df[c].value_counts()
        if c in D.ORDINAIS_CATEGORICAS:
            cont = cont.reindex(ordem_freq).fillna(0)
        else:
            cont = cont.sort_values(ascending=True)
        ax.barh(cont.index.astype(str), cont.values, color=estilo.AZUL, height=0.6)
        for y, v in enumerate(cont.values):
            ax.text(v + 15, y, f"{v / len(df):.0%}", va="center", fontsize=8)
        ax.set_xlim(0, len(df) * 1.05)
        ax.set_title(f"{D.ROTULO_CURTO.get(c, c)} ({c})", fontsize=10)
        ax.set_xlabel("Pessoas")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Distribuição das variáveis qualitativas", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    _rodape(fig, FONTE + " Fuma (2%), monitora calorias (5%) e transporte Moto/Bicicleta (<1%) são muito raros.")
    return fig


def fig_sintetico(bruto: pd.DataFrame):
    """Evidência da geração sintética: valores 'entre' as opções do questionário."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, c in zip(axes, ["FCVC", "FAF", "TUE"]):
        s = bruto[c]
        inteiro = s == s.round()
        ax.hist([s[inteiro], s[~inteiro]], bins=60, stacked=True,
                color=[estilo.AZUL, estilo.LARANJA], label=["Valor inteiro (resposta válida)", "Valor fracionário (SMOTE)"])
        ax.set_title(f"{D.ROTULO_CURTO[c]} ({c}) — {(~inteiro).mean():.0%} fracionários", fontsize=10)
        ax.set_xlabel(f"{c} (valor bruto)")
        ax.set_ylabel("Pessoas")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Inconsistência: respostas de questionário com casas decimais (dados gerados por SMOTE)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    _rodape(fig)
    return fig


# --------------------------------------------------------------------------
# Relacional
# --------------------------------------------------------------------------
def fig_correlacao(df: pd.DataFrame):
    X = D.montar_matriz(df)
    X["IMC"] = df["IMC"]
    X["Classe (ordem)"] = df["classe_codigo"]
    c = X.corr(method="spearman").rename(index=D.ROTULO_CURTO, columns=D.ROTULO_CURTO)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    mascara = np.triu(np.ones_like(c, dtype=bool), k=1)
    sns.heatmap(c, mask=mascara, cmap=estilo.cmap_divergente(), vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, linecolor="white", square=True,
                cbar_kws={"label": "Correlação de Spearman (−1 a +1)", "shrink": 0.7}, ax=ax)
    ax.set_title("Correlação entre todas as variáveis (codificadas) e a classe")
    ax.set_xlabel("Variável")
    ax.set_ylabel("Variável")
    ax.grid(False)
    fig.text(0.0, -0.06, FONTE + " Classe codificada de 0 (Abaixo do peso) a 6 (Obesidade III).",
             fontsize=8, color=estilo.TEXTO_MUDO, ha="left")
    return fig


def fig_numericas_por_classe(df: pd.DataFrame):
    variaveis = ["Age", "Height", "Weight", "FCVC", "FAF", "TUE"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, c in zip(axes.flat, variaveis):
        sns.boxplot(data=df, x="classe", y=c, hue="classe", palette=CORES_CLASSE, legend=False,
                    order=D.CLASSES_PT, ax=ax, fliersize=2, linewidth=1)
        ax.set_title(D.ROTULO_CURTO[c], fontsize=10)
        ax.set_ylabel(D.DICIONARIO[c][0].split(" (")[0] if c not in D.CONTINUAS else D.DICIONARIO[c][0])
        ax.set_xlabel("Classe")
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Variáveis numéricas por classe — Peso separa as classes; hábitos variam pouco",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    _rodape(fig)
    return fig


def fig_imc_classe(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.stripplot(data=df, y="classe", x="IMC", hue="classe", palette=CORES_CLASSE, order=D.CLASSES_PT,
                  legend=False, size=2.5, alpha=0.6, jitter=0.3, ax=ax)
    for limite, rot in [(18.5, "18,5"), (25, "25"), (30, "30"), (35, "35"), (40, "40")]:
        ax.axvline(limite, color=estilo.TINTA, linestyle="--", linewidth=0.8)
        ax.text(limite + 0.2, 6.45, f"IMC {rot}", fontsize=8, color=estilo.TINTA, va="top")
    ax.set_xlabel("IMC = peso / altura² (kg/m²) — tracejados: cortes da OMS")
    ax.set_ylabel("Classe")
    ax.set_title("As classes são faixas de IMC — e Sobrepeso I e II se sobrepõem na faixa 25–30")
    _rodape(fig, FONTE + " IMC calculado a partir de Height e Weight (variável derivada, só para a EDA).")
    return fig


def fig_altura_peso(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    for classe in D.CLASSES_PT:
        sub = df[df["classe"] == classe]
        ax.scatter(sub["Height"], sub["Weight"], s=10, color=CORES_CLASSE[classe], label=classe, alpha=0.75,
                   edgecolors="white", linewidths=0.2)
    h = np.linspace(1.44, 2.0, 50)
    for imc in [18.5, 25, 30, 35, 40]:
        ax.plot(h, imc * h ** 2, color=estilo.TEXTO_MUDO, linewidth=0.8, linestyle=":")
        ax.text(1.995, imc * 1.995 ** 2, f"IMC {imc:g}", fontsize=7.5, color=estilo.TEXTO_SECUNDARIO, va="center")
    ax.set_xlim(1.43, 2.08)
    ax.set_xlabel("Altura (m)")
    ax.set_ylabel("Peso (kg)")
    ax.set_title("Altura x Peso por classe: as classes são fatias entre curvas de IMC")
    ax.legend(title="Classe", loc="upper left", markerscale=2)
    _rodape(fig)
    return fig


def fig_categoricas_por_classe(df: pd.DataFrame):
    variaveis = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "CALC", "MTRANS"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
    for ax, c in zip(axes.flat, variaveis):
        tab = pd.crosstab(df["classe"], df[c], normalize="index").reindex(D.CLASSES_PT)
        if c in D.ORDINAIS_CATEGORICAS:
            tab = tab[[k for k in ["no", "Sometimes", "Frequently", "Always"] if k in tab.columns]]
        else:
            tab = tab[tab.sum().sort_values(ascending=False).index]
        esquerda = np.zeros(len(tab))
        cores = estilo.CATEGORICA[: tab.shape[1]]
        for k, col in enumerate(tab.columns):
            ax.barh(tab.index, tab[col], left=esquerda, color=cores[k], label=str(col), height=0.7,
                    edgecolor="white", linewidth=1)
            esquerda += tab[col].to_numpy()
        ax.set_title(f"{D.ROTULO_CURTO.get(c, c)} ({c})", fontsize=10)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xlabel("% das pessoas da classe")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7.5)
        ax.grid(False)
    axes[0, 0].invert_yaxis()  # eixo y compartilhado: inverte todos (Abaixo do peso no topo)
    axes[0, 0].set_ylabel("Classe")
    axes[1, 0].set_ylabel("Classe")
    fig.suptitle("Variáveis qualitativas por classe (barras 100%) — note Gênero em Obesidade II e III",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    _rodape(fig)
    return fig


# --------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------
def tabela_outliers(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for c in D.NUMERICAS + ["IMC"]:
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (s - s.mean()) / s.std()
        linhas.append({
            "variavel": c, "q1": q1, "q3": q3, "iqr": iqr, "limite_inf": lo, "limite_sup": hi,
            "n_outliers_iqr": int(((s < lo) | (s > hi)).sum()), "n_outliers_z3": int((z.abs() > 3).sum()),
            "min": s.min(), "max": s.max(),
        })
    return pd.DataFrame(linhas)


def fig_outliers(df: pd.DataFrame):
    variaveis = D.NUMERICAS + ["IMC"]
    z = (df[variaveis] - df[variaveis].mean()) / df[variaveis].std()
    z = z.rename(columns=D.ROTULO_CURTO)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.boxplot(data=z, orient="h", color="#eef5fd", linecolor=estilo.AZUL, fliersize=3,
                flierprops=dict(markerfacecolor=estilo.LARANJA, markeredgecolor="none"), ax=ax)
    ax.axvline(3, color=estilo.TEXTO_MUDO, linestyle=":", linewidth=1)
    ax.axvline(-3, color=estilo.TEXTO_MUDO, linestyle=":", linewidth=1)
    ax.set_xlabel("Valor padronizado (z-score) — pontilhado em |z| = 3")
    ax.set_ylabel("Variável")
    ax.set_title("Outliers (pontos laranja = fora de 1,5×IQR) nas variáveis numéricas padronizadas")
    _rodape(fig)
    return fig


# --------------------------------------------------------------------------
def executar(dataset: str, out: Path) -> dict:
    estilo.aplicar_estilo_mpl()
    pasta = out / "eda"
    pasta.mkdir(parents=True, exist_ok=True)
    bruto = D.carregar_bruto(dataset)
    df, log = D.limpar(bruto)

    D.tabela_estrutural(bruto).to_csv(pasta / "estrutura.csv", index=False)
    log.to_csv(pasta / "tratamentos.csv", index=False)
    df[D.NUMERICAS + ["IMC"]].describe().T.to_csv(pasta / "descritivas.csv")
    tabela_outliers(df).to_csv(pasta / "outliers.csv", index=False)
    df.groupby("classe", observed=True)[D.NUMERICAS + ["IMC"]].median().to_csv(pasta / "medianas_por_classe.csv")

    figuras = {
        "01_classes": fig_classes(df),
        "02_numericas": fig_numericas(df),
        "03_categoricas": fig_categoricas(df),
        "04_inconsistencia_sintetica": fig_sintetico(bruto),
        "05_correlacao": fig_correlacao(df),
        "06_numericas_por_classe": fig_numericas_por_classe(df),
        "07_imc_por_classe": fig_imc_classe(df),
        "08_altura_peso": fig_altura_peso(df),
        "09_categoricas_por_classe": fig_categoricas_por_classe(df),
        "10_outliers": fig_outliers(df),
    }
    for nome, fig in figuras.items():
        fig.savefig(pasta / f"{nome}.png")
        plt.close(fig)
    return {"linhas_brutas": len(bruto), "linhas_limpas": len(df), "figuras": list(figuras)}


def main() -> None:
    matplotlib.use("Agg")  # só na linha de comando: salva PNGs sem abrir janelas
    parser = argparse.ArgumentParser(description="EDA da Parte Comum")
    parser.add_argument("--dataset", default="obesity", choices=list(D.CAMINHOS))
    parser.add_argument("--out", default="./out/parte_comum/")
    args = parser.parse_args()
    r = executar(args.dataset, Path(args.out))
    print(f"{r['linhas_brutas']} linhas brutas -> {r['linhas_limpas']} após limpeza")
    print(f"{len(r['figuras'])} figuras salvas em {Path(args.out) / 'eda'}")


if __name__ == "__main__":
    main()
