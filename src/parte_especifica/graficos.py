"""Figuras interativas (plotly) do dashboard.

Todas usam as agregações de analise.py: os dados chegam pré-agregados ao
navegador (nunca as 119 mil linhas), o que mantém o dashboard rápido.
Regras visuais: azul = não cancelada, laranja = cancelada/risco, cinza = contexto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import estilo
from src.parte_especifica import analise, dados

PCT = ".0%"
FONTE = "Fonte: Hotel Booking Demand (Antonio, Almeida & Nunes, 2019)."


def _layout(fig: go.Figure, titulo: str, subtitulo: str | None = None, altura: int = 420, **kw) -> go.Figure:
    texto = f"<b>{titulo}</b>" + (f"<br><span style='font-size:12px'>{subtitulo}</span>" if subtitulo else "")
    base = dict(
        title=dict(text=texto, x=0, xanchor="left", y=0.97, yanchor="top"),
        height=altura, margin=dict(l=10, r=10, t=105 if subtitulo else 70, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
        hoverlabel=dict(font_size=12),
    )
    base["separators"] = ",."  # vírgula decimal, ponto de milhar (pt-BR)
    base.update(kw)
    fig.update_layout(**base)
    return fig


def _vazio(msg: str = "Nenhuma reserva com os filtros atuais.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.update_layout(height=250)


def _fmt_n(n) -> str:
    return f"{int(n):,}".replace(",", ".")


# ==========================================================================
# Qualidade
# ==========================================================================
def fig_faltantes(bruto: pd.DataFrame) -> go.Figure:
    t = dados.diagnostico_faltantes(bruto).sort_values("pct")
    mec = {"company": "Estrutural (não se aplica)", "agent": "Estrutural (não se aplica)", "meal": "Rótulo equivalente",
           "country": "MAR", "distribution_channel": "MCAR", "children": "MCAR", "market_segment": "MCAR"}
    cores = {"Estrutural (não se aplica)": estilo.AZUL, "MAR": estilo.LARANJA, "MCAR": estilo.CINZA_CONTEXTO,
             "Rótulo equivalente": "#9ec5f4"}
    t["mecanismo"] = t["coluna"].map(mec)
    fig = go.Figure()
    for m, cor in cores.items():
        sub = t[t["mecanismo"] == m]
        fig.add_bar(
            y=sub["coluna"], x=sub["pct"], orientation="h", name=m, marker_color=cor,
            text=[f"{p:.2f}% ({_fmt_n(n)})" for p, n in zip(sub["pct"], sub["total"])], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.3f}% das linhas<br>mecanismo: " + m + "<extra></extra>",
        )
    fig.update_xaxes(title="% de linhas com valor ausente (NaN ou 'Undefined')", range=[0, 118])
    fig.update_yaxes(title="Variável", categoryorder="array", categoryarray=t["coluna"].tolist())
    return _layout(fig, "Valores ausentes por variável e mecanismo provável", altura=380, barmode="overlay")


def fig_outliers(df: pd.DataFrame, col: str) -> go.Figure:
    st = analise.estatisticas_caixa(df, col, ["hotel"])
    fig = go.Figure()
    for r in st.itertuples():
        cor = estilo.COR_HOTEL[r.hotel]
        fig.add_trace(go.Box(
            name=r.hotel, q1=[r.q1], median=[r.mediana], q3=[r.q3], lowerfence=[r.bigode_inf],
            upperfence=[r.bigode_sup], mean=[r.media], marker_color=cor, orientation="h", boxmean=True,
            hoverinfo="x", showlegend=False,
        ))
        extremos = df.loc[(df["hotel"] == r.hotel) & (df[col] > r.bigode_sup), col]
        if len(extremos):
            amostra = extremos.sample(min(len(extremos), 1500), random_state=1)
            fig.add_trace(go.Scatter(
                x=amostra, y=[r.hotel] * len(amostra), mode="markers", showlegend=False,
                marker=dict(color=cor, size=5, opacity=0.35), hovertemplate="%{x}<extra>outlier</extra>",
            ))
    rot = dados.ROTULOS.get(col, col)
    fig.update_xaxes(title=rot)
    fig.update_yaxes(title="Hotel")
    n_out = int(st["n_outliers"].sum())
    return _layout(fig, f"{rot}: distribuição e outliers por hotel",
                   f"{_fmt_n(n_out)} valores fora de 1,5×IQR ({100 * n_out / max(len(df), 1):.1f}%). Pontos = até 1.500 "
                   "outliers sorteados por hotel. Losango tracejado = média.", altura=330)


# ==========================================================================
# Exploratórias
# ==========================================================================
def fig_e1_antecedencia(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vazio()
    h = analise.histograma_por_status(df, "lead_time", largura=10, maximo=500)
    fig = go.Figure()
    for status in ["Não cancelada", "Cancelada"]:
        sub = h[h["status"] == status]
        if sub.empty:
            continue
        cor = estilo.COR_STATUS[status]
        fig.add_scatter(
            x=sub["inicio"], y=sub["pct"], mode="lines", line=dict(shape="hv", width=2.5, color=cor), name=status,
            customdata=np.c_[sub["fim"], sub["contagem"]],
            hovertemplate=status + "<br>%{x:.0f}–%{customdata[0]:.0f} dias: %{y:.1f}% (%{customdata[1]:,} reservas)<extra></extra>",
        )
        med = df.loc[df["status"] == status, "lead_time"].median()
        fig.add_vline(x=med, line=dict(color=cor, dash="dash", width=1.5))
        fig.add_annotation(x=med, y=1, yref="paper", text=f"mediana {status.lower()}: {med:.0f} dias", showarrow=False,
                           xanchor="left", xshift=4, yshift=-8 if status == "Cancelada" else -28, font=dict(size=11))
    fig.update_xaxes(title="Antecedência (dias entre reservar e chegar; > 500 agrupado no fim)", range=[0, 500])
    fig.update_yaxes(title="% das reservas do grupo", ticksuffix="%")
    return _layout(fig, "Distribuição da antecedência: canceladas x não canceladas",
                   "Reservas canceladas são feitas muito antes da chegada")


def fig_e2_sazonalidade(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vazio()
    s = analise.sazonalidade(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Reservas por mês (média dos anos)", "Taxa de cancelamento"))
    for hotel, cor in estilo.COR_HOTEL.items():
        sub = s[s["hotel"] == hotel].sort_values("mes_nome")
        if sub.empty:
            continue
        x = sub["mes_nome"].astype(str)
        fig.add_scatter(x=x, y=sub["reservas_por_ano"], mode="lines+markers", name=hotel, legendgroup=hotel,
                        line=dict(color=cor, width=2.5), marker=dict(size=8),
                        hovertemplate=hotel + "<br>%{x}: %{y:,.0f} reservas/mês<extra></extra>", row=1, col=1)
        fig.add_scatter(x=x, y=sub["taxa"], mode="lines+markers", name=hotel, legendgroup=hotel, showlegend=False,
                        line=dict(color=cor, width=2.5), marker=dict(size=8),
                        hovertemplate=hotel + "<br>%{x}: %{y:.1%} canceladas<extra></extra>", row=2, col=1)
    fig.update_yaxes(title="Reservas", row=1, col=1)
    fig.update_yaxes(title="Taxa", tickformat=PCT, rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title="Mês de chegada", row=2, col=1)
    return _layout(fig, "Sazonalidade: volume de reservas e cancelamento por mês", altura=560)


NOMES_CORR = {
    "is_canceled": "Cancelada", "lead_time": "Antecedência", "adr": "Diária (ADR)", "noites_total": "Noites",
    "hospedes_total": "Hóspedes", "total_of_special_requests": "Pedidos especiais",
    "required_car_parking_spaces": "Vagas estacion.", "booking_changes": "Alterações",
    "previous_cancellations": "Cancel. anteriores", "previous_bookings_not_canceled": "Reservas ant. mantidas",
    "is_repeated_guest": "Recorrente", "days_in_waiting_list": "Lista de espera", "tem_agente": "Via agente",
    "tem_empresa": "Via empresa",
}


def fig_e3_correlacao(df: pd.DataFrame) -> go.Figure:
    if len(df) < 10:
        return _vazio()
    c = analise.correlacao(df).rename(index=NOMES_CORR, columns=NOMES_CORR)
    z = c.to_numpy().copy()
    z[np.triu_indices_from(z, k=1)] = np.nan
    texto = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(z)))
    escala = [[i / (len(estilo.DIVERGENTE) - 1), cor] for i, cor in enumerate(estilo.DIVERGENTE)]
    fig = go.Figure(go.Heatmap(
        z=z, x=c.columns, y=c.index, zmin=-1, zmax=1, colorscale=escala, text=texto, texttemplate="%{text}",
        textfont=dict(size=10), colorbar=dict(title="ρ"), hoverongaps=False,
        hovertemplate="%{y} × %{x}<br>Spearman ρ = %{z:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title="Variável")
    fig.update_xaxes(tickangle=-40, title="Variável")
    return _layout(fig, "Correlação de Spearman entre as variáveis numéricas",
                   "Vermelho = sobe junto; azul = uma sobe e a outra desce. Primeira linha: relação com o cancelamento.",
                   altura=620)


def fig_e4_adr(df: pd.DataFrame) -> go.Figure:
    d = df[df["adr"] > 0]
    if d.empty:
        return _vazio()
    st = analise.estatisticas_caixa(d, "adr", ["mes_nome", "hotel"])
    fig = go.Figure()
    for hotel, cor in estilo.COR_HOTEL.items():
        sub = st[st["hotel"] == hotel].sort_values("mes_nome")
        if sub.empty:
            continue
        fig.add_trace(go.Box(
            x=sub["mes_nome"].astype(str), q1=sub["q1"], median=sub["mediana"], q3=sub["q3"],
            lowerfence=sub["bigode_inf"], upperfence=sub["bigode_sup"], name=hotel, marker_color=cor,
            hovertemplate=hotel + " · %{x}<extra></extra>",
        ))
    fig.update_layout(boxmode="group")
    fig.update_xaxes(title="Mês de chegada")
    fig.update_yaxes(title="Diária média — ADR (€)", tickprefix="€ ")
    return _layout(fig, "Diária média (ADR) por mês e hotel",
                   "Caixa = Q1–Q3, linha = mediana, bigodes = 1,5×IQR. Cortesias (ADR = 0) excluídas.", altura=440)


def fig_e5_segmento_deposito(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vazio()
    taxa, n = analise.taxa_por_duas(df, "market_segment", "deposit_type")
    ordem = df["market_segment"].value_counts().index
    cols = [c for c in ["Sem depósito", "Reembolsável", "Não reembolsável"] if c in taxa.columns]
    taxa, n = taxa.reindex(index=ordem, columns=cols), n.reindex(index=ordem, columns=cols)
    texto = np.where(taxa.isna(), "—", taxa.map(lambda v: f"{v:.0%}") + "<br>n=" + n.fillna(0).astype(int).map(_fmt_n))
    escala = [[i / (len(estilo.SEQ_LARANJA) - 1), c] for i, c in enumerate(estilo.SEQ_LARANJA)]
    fig = go.Figure(go.Heatmap(
        z=taxa.to_numpy(dtype=float), x=cols, y=taxa.index, zmin=0, zmax=1, colorscale=escala, text=texto,
        texttemplate="%{text}", textfont=dict(size=11), colorbar=dict(title="Taxa", tickformat=PCT), hoverongaps=False,
        customdata=n.fillna(0).to_numpy(), hovertemplate="%{y} · %{x}<br>%{z:.1%} canceladas<br>%{customdata:,} reservas<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title="Segmento (por volume)")
    fig.update_xaxes(title="Tipo de depósito", side="bottom")
    return _layout(fig, "Taxa de cancelamento por segmento e tipo de depósito",
                   "'Não reembolsável' cancela ~99%: lotes de quartos de grupos/operadoras liberados em bloco",
                   altura=470)


def fig_e6_compromisso(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vazio()
    paineis = [("pedidos_especiais_cat", "Pedidos especiais", dados.ORDEM_PEDIDOS, None),
               ("is_repeated_guest", "Hóspede recorrente", [0, 1], {0: "Não", 1: "Sim"}),
               ("tem_agente", "Reservou via agente", [0, 1], {0: "Não", 1: "Sim"})]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=[p[1] for p in paineis],
                        horizontal_spacing=0.05)
    geral = df["is_canceled"].mean()
    for i, (col, nome, ordem, mapa) in enumerate(paineis, start=1):
        t = analise.taxa_por(df, col, ordem=ordem)
        x = t[col].astype(str) if mapa is None else t[col].map(mapa)
        fig.add_bar(x=x, y=t["taxa"], marker_color=estilo.AZUL, showlegend=False,
                    text=[f"{v:.0%}" for v in t["taxa"]], textposition="outside", customdata=t["reservas"],
                    hovertemplate=nome + ": %{x}<br>%{y:.1%} canceladas<br>%{customdata:,} reservas<extra></extra>",
                    row=1, col=i)
        fig.add_hline(y=geral, line=dict(color=estilo.TEXTO_MUDO, dash="dash", width=1), row=1, col=i)
        fig.update_xaxes(title=nome, type="category", row=1, col=i)
    fig.update_yaxes(title="Taxa de cancelamento", tickformat=PCT, range=[0, 0.65], row=1, col=1)
    return _layout(fig, "Taxa de cancelamento por sinais de relacionamento do hóspede",
                   f"Tracejado = média geral dos filtros atuais ({geral:.0%})", altura=400)


# ==========================================================================
# Explanatórias
# ==========================================================================
def fig_x1(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vazio()
    t = analise.taxa_por(df, "faixa_antecedencia", ordem=dados.ROTULOS_ANTECEDENCIA)
    geral = df["is_canceled"].mean()
    cores = [estilo.LARANJA if v > geral else estilo.CINZA_CONTEXTO for v in t["taxa"]]
    fig = go.Figure(go.Bar(
        x=t["faixa_antecedencia"].astype(str), y=t["taxa"], marker_color=cores,
        text=[f"<b>{v:.0%}</b>" for v in t["taxa"]], textposition="outside", textfont=dict(size=17),
        customdata=t["reservas"], hovertemplate="%{x}<br>%{y:.1%} canceladas<br>%{customdata:,} reservas<extra></extra>",
    ))
    fig.add_hline(y=geral, line=dict(dash="dash", width=1.5, color=estilo.TEXTO_SECUNDARIO),
                  annotation_text=f"média geral: {geral:.0%}", annotation_position="top left")
    fig.update_yaxes(title="Taxa de cancelamento", tickformat=PCT, range=[0, max(0.8, t["taxa"].max() * 1.15)])
    fig.update_xaxes(title="Antecedência da reserva (dias entre reservar e chegar)")
    return _layout(fig, "Quanto mais cedo o hóspede reserva, maior a chance de cancelar",
                   "Laranja = faixas acima da média. Passe o mouse para ver o número de reservas.", altura=460)


def fig_x2(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vazio()
    t = analise.taxa_antecedencia_por_pedidos(df)
    fig = go.Figure()
    for grupo, cor, larg in [("Com ≥ 1 pedido especial", estilo.TEXTO_MUDO, 2.5),
                             ("Nenhum pedido especial", estilo.LARANJA, 5)]:
        sub = t[t["pedidos"] == grupo].sort_values("faixa_antecedencia")
        if sub.empty:
            continue
        x = sub["faixa_antecedencia"].astype(str)
        fig.add_scatter(
            x=x, y=sub["taxa"], mode="lines+markers+text", name=grupo, line=dict(color=cor, width=larg),
            marker=dict(size=11 if larg > 3 else 8), text=[f"{v:.0%}" for v in sub["taxa"]],
            textposition="top center" if larg > 3 else "bottom center", customdata=sub["reservas"],
            hovertemplate=grupo + "<br>%{x}: %{y:.1%} canceladas<br>%{customdata:,} reservas<extra></extra>",
        )
        fig.add_annotation(x=x.iloc[-1], y=sub["taxa"].iloc[-1], text=f"<b>{grupo}</b>" if larg > 3 else grupo,
                           showarrow=False, xanchor="left", xshift=14)
    fig.update_yaxes(title="Taxa de cancelamento", tickformat=PCT, range=[0, 1])
    fig.update_xaxes(title="Antecedência da reserva", range=[-0.3, 6.4])
    return _layout(fig, "Um pedido especial corta o cancelamento pela metade — em qualquer antecedência",
                   "Pedido especial = berço, andar alto, cama de casal etc.: sinal de que o hóspede planejou a estadia",
                   altura=480, showlegend=True)


def fig_x3(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vazio()
    p = analise.perfil_risco(df)
    cores = [estilo.LARANJA, "#d9d8d2", "#bdbcb4", "#a5a49c"]
    barras = ["Reservas", "Cancelamentos"]
    fig = go.Figure()
    for k, r in enumerate(p.itertuples()):
        vals = [r.pct_reservas, r.pct_cancelamentos]
        fig.add_bar(
            y=barras, x=vals, orientation="h", name=f"{r.perfil} (cancela {r.taxa:.0%})", marker_color=cores[k],
            marker_line=dict(color="white", width=2),
            text=[f"<b>{v:.0%}</b>" if k == 0 else f"{v:.0%}" for v in vals], textposition="inside",
            insidetextanchor="middle", textfont=dict(size=18 if k == 0 else 12, color="white" if k == 0 else "#0b0b0b"),
            hovertemplate=r.perfil + "<br>%{x:.1%} do total de %{y}<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="% do total", tickformat=PCT, range=[0, 1])
    fig.update_yaxes(autorange="reversed", title="")
    p0 = p.iloc[0]
    return _layout(fig, f"O perfil de maior risco é {p0.pct_reservas:.0%} das reservas, mas {p0.pct_cancelamentos:.0%} dos cancelamentos",
                   "Perfil de risco = mais de 90 dias de antecedência e nenhum pedido especial", altura=400,
                   legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0, traceorder="normal"))


# ==========================================================================
# Modelo
# ==========================================================================
def fig_importancia(imp: pd.DataFrame, n: int = 12) -> go.Figure:
    d = imp.head(n).iloc[::-1]
    cores = [estilo.LARANJA if v in ("lead_time", "total_of_special_requests") else estilo.AZUL for v in d["variavel"]]
    fig = go.Figure(go.Bar(
        x=d["queda_auc"], y=d["rotulo"], orientation="h", marker_color=cores,
        error_x=dict(type="data", array=d["desvio"], color=estilo.TEXTO_MUDO, thickness=1),
        hovertemplate="%{y}<br>queda da AUC: %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="Queda média da AUC ao embaralhar a variável")
    fig.update_yaxes(title="")
    return _layout(fig, "O que o modelo usa para prever o cancelamento",
                   "Laranja = variáveis da hipótese (antecedência e pedidos especiais)", altura=440)


def fig_roc(roc: pd.DataFrame, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=estilo.TEXTO_MUDO),
                    name="Aleatório (AUC 0,50)", hoverinfo="skip")
    fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=estilo.LARANJA, width=3),
                    name=f"Modelo (AUC {auc:.2f})",
                    hovertemplate="falsos alarmes: %{x:.0%}<br>cancelamentos detectados: %{y:.0%}<extra></extra>")
    fig.update_xaxes(title="Taxa de falsos alarmes (não cancelou, mas foi apontada)", tickformat=PCT, range=[0, 1])
    fig.update_yaxes(title="Cancelamentos detectados (recall)", tickformat=PCT, range=[0, 1])
    return _layout(fig, "Curva ROC no teste (chegadas de 2017)", altura=400)


def fig_matriz_modelo(cm: pd.DataFrame) -> go.Figure:
    z = cm.to_numpy()
    tot = z.sum()
    texto = [[f"{_fmt_n(v)}<br>({v / tot:.0%})" for v in linha] for linha in z]
    escala = [[i / (len(estilo.SEQ_AZUL) - 1), c] for i, c in enumerate(estilo.SEQ_AZUL)]
    fig = go.Figure(go.Heatmap(z=z, x=list(cm.columns), y=list(cm.index), colorscale=escala, text=texto,
                               texttemplate="%{text}", textfont=dict(size=14), showscale=False,
                               hovertemplate="%{y} · %{x}: %{z:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Realidade")
    fig.update_xaxes(title="Previsão do modelo (limiar 0,5)")
    return _layout(fig, "Matriz de confusão (teste 2017)", altura=400)


def fig_curva_simulador(lead: np.ndarray, proba: np.ndarray, lead_atual: int, proba_atual: float, base: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=lead, y=proba, mode="lines", line=dict(color=estilo.LARANJA, width=3),
                    name="Risco estimado", hovertemplate="%{x} dias: %{y:.0%}<extra></extra>")
    fig.add_hline(y=base, line=dict(dash="dash", color=estilo.TEXTO_MUDO, width=1),
                  annotation_text=f"taxa média: {base:.0%}", annotation_position="bottom right")
    fig.add_scatter(x=[lead_atual], y=[proba_atual], mode="markers", marker=dict(size=16, color=estilo.LARANJA,
                    line=dict(color="white", width=3)), name="Esta reserva",
                    hovertemplate="Esta reserva<br>%{x} dias: %{y:.0%}<extra></extra>")
    fig.update_xaxes(title="Antecedência (dias)")
    fig.update_yaxes(title="Probabilidade estimada de cancelar", tickformat=PCT, range=[0, 1])
    return _layout(fig, "Como o risco desta reserva mudaria com outra antecedência",
                   "Todas as outras características mantidas iguais", altura=380, showlegend=False)


# ==========================================================================
# Explorador
# ==========================================================================
NUMERICAS_EXPLORADOR = [
    "lead_time", "adr", "noites_total", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "hospedes_total", "total_of_special_requests", "booking_changes", "previous_cancellations",
    "previous_bookings_not_canceled", "days_in_waiting_list", "required_car_parking_spaces",
    "arrival_date_week_number", "arrival_date_day_of_month", "receita_estimada",
]
CATEGORICAS_EXPLORADOR = [
    "status", "hotel", "mes_nome", "arrival_date_year", "faixa_antecedencia", "pedidos_especiais_cat", "market_segment",
    "distribution_channel", "deposit_type", "customer_type", "meal", "pais_grupo", "reserved_room_type",
    "assigned_room_type", "is_repeated_guest", "tem_agente", "tem_empresa", "quarto_diferente", "reservation_status",
]


def _ordem_categorias(df: pd.DataFrame, col: str) -> list:
    s = df[col]
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if c in set(s.unique())]
    if pd.api.types.is_numeric_dtype(s):
        return sorted(s.unique())
    return s.value_counts().index.tolist()


def fig_explorador(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    if df.empty:
        return _vazio()
    rx, ry = dados.ROTULOS.get(x, x), dados.ROTULOS.get(y, y)
    x_num, y_num = x in NUMERICAS_EXPLORADOR, y in NUMERICAS_EXPLORADOR
    escala_azul = [[i / (len(estilo.SEQ_AZUL) - 1), c] for i, c in enumerate(estilo.SEQ_AZUL)]

    if x_num and y_num:
        lim_x, lim_y = df[x].quantile(0.995), df[y].quantile(0.995)
        d = df[(df[x] <= lim_x) & (df[y] <= lim_y)]
        nbx = min(40, max(5, d[x].nunique()))
        nby = min(40, max(5, d[y].nunique()))
        H, ex, ey = np.histogram2d(d[x], d[y], bins=[nbx, nby])
        cx, cy = (ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2
        Hm = np.where(H.T > 0, H.T, np.nan)
        fig = go.Figure(go.Heatmap(x=cx, y=cy, z=np.log10(Hm), colorscale=escala_azul, customdata=H.T,
                                   colorbar=dict(title="reservas", tickvals=[0, 1, 2, 3, 4],
                                                 ticktext=["1", "10", "100", "1 mil", "10 mil"]),
                                   hovertemplate=rx + " ≈ %{x:.1f}<br>" + ry + " ≈ %{y:.1f}<br>%{customdata:,.0f} reservas<extra></extra>"))
        med = d.groupby(pd.cut(d[x], ex, include_lowest=True), observed=True)[y].median()
        fig.add_scatter(x=[i.mid for i in med.index], y=med.values, mode="lines+markers", name="Mediana de Y",
                        line=dict(color=estilo.LARANJA, width=2.5), marker=dict(size=5))
        fig.update_xaxes(title=rx)
        fig.update_yaxes(title=ry)
        return _layout(fig, f"{ry} × {rx}", "Cor = nº de reservas (escala log). Linha laranja = mediana de Y em cada faixa de X. "
                       "0,5% mais extremos de cada eixo omitidos.", altura=480)

    if y == "is_canceled" or x == "is_canceled":
        outra = x if y == "is_canceled" else y
        if outra in NUMERICAS_EXPLORADOR:
            faixas = pd.qcut(df[outra], q=10, duplicates="drop")
            t = df.groupby(faixas, observed=True)["is_canceled"].agg(["mean", "size"]).reset_index()
            rotulos = [f"{i.left:.0f}–{i.right:.0f}" for i in t[outra]]
        else:
            t = df.groupby(outra, observed=True)["is_canceled"].agg(["mean", "size"]).reindex(
                _ordem_categorias(df, outra)).reset_index()
            rotulos = t[outra].astype(str)
        geral = df["is_canceled"].mean()
        fig = go.Figure(go.Bar(x=rotulos, y=t["mean"], customdata=t["size"],
                               marker_color=[estilo.LARANJA if v > geral else estilo.AZUL for v in t["mean"]],
                               text=[f"{v:.0%}" for v in t["mean"]], textposition="outside",
                               hovertemplate="%{x}<br>%{y:.1%} canceladas<br>%{customdata:,} reservas<extra></extra>"))
        fig.add_hline(y=geral, line=dict(dash="dash", color=estilo.TEXTO_MUDO), annotation_text=f"média {geral:.0%}")
        fig.update_xaxes(title=dados.ROTULOS.get(outra, outra) + (" (faixas por decil)" if outra in NUMERICAS_EXPLORADOR else ""),
                         type="category")
        fig.update_yaxes(title="Taxa de cancelamento", tickformat=PCT, range=[0, 1.1])
        return _layout(fig, f"Taxa de cancelamento por {dados.ROTULOS.get(outra, outra).lower()}",
                       "Laranja = acima da média", altura=460)

    if x_num != y_num:  # uma categórica e uma numérica -> boxplots
        cat, num = (y, x) if x_num else (x, y)
        ordem = _ordem_categorias(df, cat)[:25]
        st = analise.estatisticas_caixa(df[df[cat].isin(ordem)], num, [cat])
        st[cat] = pd.Categorical(st[cat], categories=ordem, ordered=True)
        st = st.sort_values(cat)
        fig = go.Figure(go.Box(
            x=st[cat].astype(str), q1=st["q1"], median=st["mediana"], q3=st["q3"], lowerfence=st["bigode_inf"],
            upperfence=st["bigode_sup"], mean=st["media"], marker_color=estilo.AZUL, boxmean=True, name=num,
        ))
        fig.update_xaxes(title=dados.ROTULOS.get(cat, cat), type="category")
        fig.update_yaxes(title=dados.ROTULOS.get(num, num))
        return _layout(fig, f"{dados.ROTULOS.get(num, num)} por {dados.ROTULOS.get(cat, cat).lower()}",
                       "Boxplot (sem outliers desenhados): caixa = Q1–Q3, linha = mediana, tracejado = média",
                       altura=480, showlegend=False)

    # categórica x categórica -> heatmap de % por linha
    ox, oy = _ordem_categorias(df, x)[:25], _ordem_categorias(df, y)[:25]
    ct = pd.crosstab(df[y], df[x]).reindex(index=oy, columns=ox).fillna(0)
    pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
    escala = [[i / (len(estilo.SEQ_AZUL) - 1), c] for i, c in enumerate(estilo.SEQ_AZUL)]
    fig = go.Figure(go.Heatmap(z=pct.to_numpy(), x=[str(c) for c in ox], y=[str(c) for c in oy], colorscale=escala,
                               zmin=0, zmax=1, text=pct.map(lambda v: f"{v:.0%}" if pd.notna(v) and v >= 0.005 else ""),
                               texttemplate="%{text}", customdata=ct.to_numpy(), colorbar=dict(title="% da linha", tickformat=PCT),
                               hovertemplate=ry + ": %{y}<br>" + rx + ": %{x}<br>%{z:.1%} da linha (%{customdata:,.0f} reservas)<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=ry, type="category")
    fig.update_xaxes(title=rx, type="category")
    return _layout(fig, f"{ry} × {rx}", "Cada linha soma 100%: como as reservas de cada categoria de Y se distribuem em X",
                   altura=max(420, 30 * len(oy) + 160))
