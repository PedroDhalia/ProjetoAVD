"""Agregações usadas tanto pelo script de EDA (figuras estáticas) quanto pelo
dashboard (figuras interativas). Manter os cálculos aqui garante que o
relatório e o dashboard mostrem exatamente os mesmos números.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.parte_especifica.dados import ROTULOS_ANTECEDENCIA

PERFIS_RISCO = [
    "> 90 dias e nenhum pedido especial",
    "≤ 90 dias e nenhum pedido especial",
    "> 90 dias e ≥ 1 pedido especial",
    "≤ 90 dias e ≥ 1 pedido especial",
]


# --------------------------------------------------------------------------
# Taxas de cancelamento
# --------------------------------------------------------------------------
def taxa_por(df: pd.DataFrame, col: str, ordem: list | None = None, min_n: int = 0) -> pd.DataFrame:
    """Reservas, cancelamentos e taxa de cancelamento por categoria."""
    g = (
        df.groupby(col, observed=True)["is_canceled"]
        .agg(reservas="size", canceladas="sum")
        .reset_index()
    )
    g["taxa"] = g["canceladas"] / g["reservas"]
    g["participacao_reservas"] = g["reservas"] / g["reservas"].sum()
    g = g[g["reservas"] >= min_n]
    if ordem is not None:
        g[col] = pd.Categorical(g[col], categories=ordem, ordered=True)
        g = g.sort_values(col)
    return g.reset_index(drop=True)


def taxa_por_duas(df: pd.DataFrame, linhas: str, colunas: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matriz de taxa de cancelamento e matriz de contagens (linhas x colunas)."""
    taxa = df.pivot_table(index=linhas, columns=colunas, values="is_canceled", aggfunc="mean", observed=True)
    n = df.pivot_table(index=linhas, columns=colunas, values="is_canceled", aggfunc="size", observed=True)
    return taxa, n


def taxa_antecedencia_por_pedidos(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa por faixa de antecedência, separando reservas com e sem pedido especial."""
    d = df.assign(
        pedidos=np.where(df["total_of_special_requests"] > 0, "Com ≥ 1 pedido especial", "Nenhum pedido especial")
    )
    g = (
        d.groupby(["faixa_antecedencia", "pedidos"], observed=True)["is_canceled"]
        .agg(reservas="size", taxa="mean")
        .reset_index()
    )
    g["faixa_antecedencia"] = pd.Categorical(g["faixa_antecedencia"], ROTULOS_ANTECEDENCIA, ordered=True)
    return g.sort_values(["pedidos", "faixa_antecedencia"]).reset_index(drop=True)


def perfil_risco(df: pd.DataFrame) -> pd.DataFrame:
    """Divide as reservas em 4 perfis (antecedência x pedidos especiais) e compara
    a participação de cada perfil no total de reservas e no total de cancelamentos."""
    longa = df["lead_time"] > 90
    sem_pedido = df["total_of_special_requests"] == 0
    perfil = np.select(
        [longa & sem_pedido, ~longa & sem_pedido, longa & ~sem_pedido],
        PERFIS_RISCO[:3],
        default=PERFIS_RISCO[3],
    )
    g = (
        df.assign(perfil=perfil)
        .groupby("perfil")["is_canceled"]
        .agg(reservas="size", canceladas="sum")
        .reindex(PERFIS_RISCO)
        .fillna(0)
        .reset_index()
    )
    g["taxa"] = g["canceladas"] / g["reservas"].replace(0, np.nan)
    g["pct_reservas"] = g["reservas"] / g["reservas"].sum()
    g["pct_cancelamentos"] = g["canceladas"] / max(g["canceladas"].sum(), 1)
    return g


def serie_mensal(df: pd.DataFrame) -> pd.DataFrame:
    """Reservas e taxa de cancelamento por mês de chegada e hotel."""
    g = (
        df.groupby(["ano_mes", "hotel"])["is_canceled"]
        .agg(reservas="size", canceladas="sum")
        .reset_index()
    )
    g["taxa"] = g["canceladas"] / g["reservas"]
    return g


def sazonalidade(df: pd.DataFrame) -> pd.DataFrame:
    """Média por mês do calendário (juntando os anos) e hotel."""
    g = (
        df.groupby(["mes_nome", "hotel"], observed=True)
        .agg(reservas=("is_canceled", "size"), taxa=("is_canceled", "mean"), adr_mediana=("adr", "median"))
        .reset_index()
    )
    n_anos = df.groupby("mes_nome", observed=True)["arrival_date_year"].nunique()
    g["reservas_por_ano"] = g["reservas"] / g["mes_nome"].map(n_anos).astype(float)
    return g


# --------------------------------------------------------------------------
# Distribuições
# --------------------------------------------------------------------------
def histograma_por_status(df: pd.DataFrame, col: str, largura: float, maximo: float | None = None) -> pd.DataFrame:
    """Histograma pré-calculado (em % de cada grupo) para não enviar 119 mil pontos ao navegador."""
    maximo = maximo if maximo is not None else float(df[col].max())
    bordas = np.arange(0, maximo + largura, largura)
    linhas = []
    for status, sub in df.groupby("status"):
        cont, _ = np.histogram(sub[col].clip(upper=maximo), bins=bordas)
        linhas.append(
            pd.DataFrame({"inicio": bordas[:-1], "fim": bordas[1:], "contagem": cont,
                          "pct": 100 * cont / max(len(sub), 1), "status": status})
        )
    return pd.concat(linhas, ignore_index=True)


def estatisticas_caixa(df: pd.DataFrame, valor: str, grupos: list[str]) -> pd.DataFrame:
    """Quartis e cercas de Tukey (1,5 x IQR) por grupo, para desenhar boxplots
    sem enviar todos os pontos."""
    def _stats(s: pd.Series) -> pd.Series:
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        dentro = s[(s >= lo) & (s <= hi)]
        return pd.Series(
            {"q1": q1, "mediana": med, "q3": q3,
             "bigode_inf": dentro.min() if len(dentro) else q1,
             "bigode_sup": dentro.max() if len(dentro) else q3,
             "n": len(s), "n_outliers": int(((s < lo) | (s > hi)).sum()), "media": s.mean()}
        )

    return df.groupby(grupos, observed=True)[valor].apply(_stats).unstack().reset_index()


def correlacao(df: pd.DataFrame, metodo: str = "spearman") -> pd.DataFrame:
    cols = [
        "is_canceled", "lead_time", "adr", "noites_total", "hospedes_total", "total_of_special_requests",
        "required_car_parking_spaces", "booking_changes", "previous_cancellations",
        "previous_bookings_not_canceled", "is_repeated_guest", "days_in_waiting_list",
        "tem_agente", "tem_empresa",
    ]
    return df[cols].corr(method=metodo)


# --------------------------------------------------------------------------
# Qualidade
# --------------------------------------------------------------------------
def tabela_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Regra de Tukey: fora de [Q1 - 1,5 IQR, Q3 + 1,5 IQR]."""
    linhas = []
    for c in cols:
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        linhas.append(
            {"variavel": c, "q1": q1, "q3": q3, "limite_inf": lo, "limite_sup": hi,
             "min": s.min(), "max": s.max(), "n_outliers": n, "pct_outliers": 100 * n / len(s)}
        )
    return pd.DataFrame(linhas)


def sensibilidade_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    """Compara indicadores com e sem as linhas repetidas."""
    sem = df[df["duplicada"] == 0]
    linhas = []
    for nome, base in [("Com repetidas (base usada)", df), ("Sem repetidas", sem)]:
        faixa = base.groupby("faixa_antecedencia", observed=True)["is_canceled"].mean()
        ped = base.groupby(base["total_of_special_requests"] > 0)["is_canceled"].mean()
        linhas.append(
            {
                "base": nome,
                "reservas": len(base),
                "taxa_geral": base["is_canceled"].mean(),
                "taxa_0_7_dias": faixa.iloc[0],
                "taxa_mais_1_ano": faixa.iloc[-1],
                "taxa_sem_pedido": ped.loc[False],
                "taxa_com_pedido": ped.loc[True],
                "taxa_nao_reembolsavel": base.loc[base["deposit_type"] == "Não reembolsável", "is_canceled"].mean(),
            }
        )
    return pd.DataFrame(linhas)


def kpis(df: pd.DataFrame) -> dict:
    canc = df["is_canceled"] == 1
    receita_total = df["receita_estimada"].sum()
    return {
        "reservas": len(df),
        "taxa_cancelamento": canc.mean() if len(df) else np.nan,
        "canceladas": int(canc.sum()),
        "receita_cancelada": df.loc[canc, "receita_estimada"].sum(),
        "pct_receita_cancelada": df.loc[canc, "receita_estimada"].sum() / receita_total if receita_total else np.nan,
        "antecedencia_mediana": df["lead_time"].median() if len(df) else np.nan,
        "adr_mediana": df["adr"].median() if len(df) else np.nan,
    }


# --------------------------------------------------------------------------
# Medidas de associação (explorador de variáveis)
# --------------------------------------------------------------------------
def associacao(df: pd.DataFrame, x: str, y: str, numericas: list[str]) -> tuple[str, float, str]:
    """Escolhe a medida de associação conforme o tipo das duas variáveis.

    * numérica x numérica: correlação de Spearman (ρ, de −1 a 1);
    * categórica x numérica: razão de correlação η² (0 a 1) = variância entre grupos / variância total;
    * categórica x categórica: V de Cramér (0 a 1), baseado no qui-quadrado.
    O alvo binário is_canceled é tratado como categórico.
    """
    d = df[[x, y]].dropna()
    if len(d) < 3 or x == y:
        return "—", float("nan"), "Selecione duas variáveis diferentes."
    x_num = x in numericas and x != "is_canceled"
    y_num = y in numericas and y != "is_canceled"
    if x_num and y_num:
        v = d[x].corr(d[y], method="spearman")
        forca = abs(v)
        nome = "Spearman ρ"
    elif x_num or y_num:
        cat, num = (y, x) if x_num else (x, y)
        g = d.groupby(cat, observed=True)[num]
        ss_entre = (g.size() * (g.mean() - d[num].mean()) ** 2).sum()
        ss_total = ((d[num] - d[num].mean()) ** 2).sum()
        v = ss_entre / ss_total if ss_total > 0 else float("nan")
        forca = np.sqrt(v) if pd.notna(v) else 0
        nome = "η² (razão de correlação)"
    else:
        ct = pd.crosstab(d[x], d[y]).to_numpy().astype(float)
        n = ct.sum()
        esperado = ct.sum(axis=1, keepdims=True) * ct.sum(axis=0, keepdims=True) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            qui2 = np.nansum((ct - esperado) ** 2 / esperado)
        k = min(ct.shape) - 1
        v = np.sqrt(qui2 / (n * k)) if k > 0 else float("nan")
        forca = v
        nome = "V de Cramér"
    if pd.isna(forca):
        texto = "indefinida"
    elif forca < 0.1:
        texto = "associação desprezível"
    elif forca < 0.3:
        texto = "associação fraca"
    elif forca < 0.5:
        texto = "associação moderada"
    else:
        texto = "associação forte"
    return nome, float(v), texto
