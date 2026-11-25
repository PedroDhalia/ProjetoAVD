"""Peças compartilhadas pelas páginas do dashboard: dados em cache, filtros

globais, renderização das notas analíticas e exportação do relatório."""



from __future__ import annotations



import datetime as dt

import html

from pathlib import Path



import numpy as np

import pandas as pd

import streamlit as st



from src.parte_especifica import analise, dados, graficos, modelo, notas



RAIZ = Path(__file__).resolve().parents[2]

PASTA_OUT = RAIZ / "out" / "parte_especifica" / "tabelas"

INTEGRANTES = ["Pedro de Melo Dhalia", "Matheos de Oliveira Guerra", "Miguel José França de Andrade"]



def br(texto: str) -> str:
    """Troca separadores do formato en-US pelo pt-BR: 1,234.5 -> 1.234,5."""
    return texto.translate(str.maketrans({",": ".", ".": ","}))


def pct(v: float, casas: int = 1) -> str:

    return br(f"{v:.{casas}%}")





def num(v: float, casas: int = 0) -> str:

    return br(f"{v:,.{casas}f}")





CHAVES_FILTRO = {

    "hotel": ["Hotel urbano", "Hotel resort"],

    "ano": [2015, 2016, 2017],

    "meses": ("Jan", "Dez"),

    "segmento": [],

    "cliente": [],

    "sem_repetidas": False,

}





# --------------------------------------------------------------------------

# Dados em cache (carregados uma vez por processo e compartilhados)

# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando e tratando as 119 mil reservas…")

def carregar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """(bruto, limpo, log de tratamentos). Não modificar os DataFrames devolvidos."""

    bruto = dados.carregar_bruto()

    limpo, log = dados.limpar(bruto)

    return bruto, limpo, log





@st.cache_resource(show_spinner="Treinando o modelo de risco (só na primeira vez, ~10 s)…")

def obter_modelo() -> modelo.ResultadoModelo:

    _, df, _ = carregar()

    caminho = PASTA_OUT / "modelo_importancia.csv"

    res = modelo.treinar(df, calcular_importancia=not caminho.exists())

    if caminho.exists():

        res.importancia = pd.read_csv(caminho)

    return res





# --------------------------------------------------------------------------

# Filtros globais

# --------------------------------------------------------------------------

# Padrão usado: o valor "oficial" de cada filtro fica em st.session_state["flt_*"]

# (sobrevive à troca de página). Os widgets usam chaves próprias ("w_*"), nascem

# com o valor oficial e o atualizam via on_change. Assim os filtros valem já na

# primeira execução de qualquer página.

def iniciar_estado() -> None:

    for chave, padrao in CHAVES_FILTRO.items():

        if "flt_" + chave not in st.session_state:

            st.session_state["flt_" + chave] = padrao





def _sincronizar(chave: str) -> None:

    st.session_state["flt_" + chave] = st.session_state["w_" + chave]





def _limpar_filtros() -> None:

    for chave, padrao in CHAVES_FILTRO.items():

        st.session_state["flt_" + chave] = padrao

        st.session_state.pop("w_" + chave, None)  # o widget renasce com o valor padrão





def _filtro(chave: str):

    return st.session_state.get("flt_" + chave, CHAVES_FILTRO[chave])





def _meses() -> tuple[int, int]:

    v = _filtro("meses")

    v = [v] if isinstance(v, str) else list(v)

    idx = [dados.MESES_PT.index(m) + 1 for m in v if m in dados.MESES_PT] or [1, 12]

    return min(idx), max(idx)





def aplicar_filtros(df: pd.DataFrame, ignorar_categorias: bool = False) -> pd.DataFrame:

    m = np.ones(len(df), dtype=bool)

    if _filtro("hotel"):

        m &= df["hotel"].isin(_filtro("hotel")).to_numpy()

    if _filtro("ano"):

        m &= df["arrival_date_year"].isin(_filtro("ano")).to_numpy()

    i, f = _meses()

    m &= df["mes_num"].between(i, f).to_numpy()

    if _filtro("sem_repetidas"):

        m &= (df["duplicada"] == 0).to_numpy()

    if not ignorar_categorias:

        if _filtro("segmento"):

            m &= df["market_segment"].isin(_filtro("segmento")).to_numpy()

        if _filtro("cliente"):

            m &= df["customer_type"].isin(_filtro("cliente")).to_numpy()

    return df.loc[m]





def descrever_filtros() -> str:

    i, f = _meses()

    partes = [

        "Hotel: " + (", ".join(_filtro("hotel")) or "todos"),

        "Anos: " + (", ".join(map(str, sorted(_filtro("ano")))) or "todos"),

        f"Meses: {dados.MESES_PT[i - 1]}–{dados.MESES_PT[f - 1]}",

        "Segmentos: " + (", ".join(_filtro("segmento")) or "todos"),

        "Tipo de cliente: " + (", ".join(_filtro("cliente")) or "todos"),

        "Linhas repetidas: " + ("removidas" if _filtro("sem_repetidas") else "mantidas"),

    ]

    return " · ".join(partes)





def barra_lateral_filtros() -> pd.DataFrame:

    """Desenha os filtros na barra lateral e devolve os dados filtrados.



    Os filtros são INTERLIGADOS: as contagens exibidas ao lado de cada segmento

    e tipo de cliente se atualizam conforme hotel, ano, meses e repetidas.

    """

    iniciar_estado()

    _, df, _ = carregar()

    seg_opcoes = df["market_segment"].value_counts().index.tolist()

    cli_opcoes = df["customer_type"].value_counts().index.tolist()

    with st.sidebar:

        st.subheader("Filtros", anchor=False)

        st.pills("Hotel", CHAVES_FILTRO["hotel"], selection_mode="multi", key="w_hotel", default=_filtro("hotel"),

                 on_change=_sincronizar, args=("hotel",), help="Nenhum selecionado = todos")

        st.pills("Ano de chegada", CHAVES_FILTRO["ano"], selection_mode="multi", key="w_ano", default=_filtro("ano"),

                 on_change=_sincronizar, args=("ano",), help="Os dados vão de jul/2015 a ago/2017")

        i, f = _meses()

        st.select_slider("Meses de chegada", dados.MESES_PT, value=(dados.MESES_PT[i - 1], dados.MESES_PT[f - 1]),

                         key="w_meses", on_change=_sincronizar, args=("meses",))

        st.toggle("Remover linhas repetidas", key="w_sem_repetidas", value=_filtro("sem_repetidas"),

                  on_change=_sincronizar, args=("sem_repetidas",),

                  help="Análise de sensibilidade: 26,8% das linhas são idênticas a outra (quartos de uma mesma "

                       "reserva de grupo). Ligue para ver se as conclusões se mantêm sem elas.")



        base = aplicar_filtros(df, ignorar_categorias=True)

        cont_seg = base["market_segment"].value_counts()

        cont_cli = base["customer_type"].value_counts()

        st.multiselect("Segmento de mercado", seg_opcoes, key="w_segmento", default=_filtro("segmento"),

                       on_change=_sincronizar, args=("segmento",), placeholder="Todos",

                       format_func=lambda v: f"{v} ({cont_seg.get(v, 0):,})".replace(",", "."),

                       help="Entre parênteses: reservas disponíveis com os demais filtros.")

        st.multiselect("Tipo de cliente", cli_opcoes, key="w_cliente", default=_filtro("cliente"),

                       on_change=_sincronizar, args=("cliente",), placeholder="Todos",

                       format_func=lambda v: f"{v} ({cont_cli.get(v, 0):,})".replace(",", "."),

                       help="Entre parênteses: reservas disponíveis com os demais filtros.")

        filtrado = aplicar_filtros(df)

        st.caption(f"**{len(filtrado):,}** de {len(df):,} reservas selecionadas".replace(",", "."))

        st.button("Limpar filtros", icon=":material/restart_alt:", on_click=_limpar_filtros, width="stretch")



        st.divider()

        st.subheader("Exportar", anchor=False)

        st.download_button(

            "Dados filtrados (CSV)", data=lambda: _csv(filtrado), file_name="reservas_filtradas.csv",

            mime="text/csv", icon=":material/table_chart:", width="stretch", on_click="ignore",

        )

        st.download_button(

            "Relatório da seleção (HTML)", data=lambda: relatorio_html(filtrado, descrever_filtros()),

            file_name="relatorio_cancelamentos.html", mime="text/html", icon=":material/description:",

            width="stretch", on_click="ignore",

            help="Relatório com indicadores, gráficos interativos e notas analíticas para os filtros atuais.",

        )

    return filtrado





def _csv(df: pd.DataFrame) -> bytes:

    colunas = [c for c in df.columns if c not in ("data_chegada", "ano_mes")]

    return df[colunas].to_csv(index=False).encode("utf-8-sig")





# --------------------------------------------------------------------------

# Componentes visuais

# --------------------------------------------------------------------------

def nota(chave: str, expandida: bool = True) -> None:

    n = notas.NOTAS[chave]

    with st.container(border=True):

        st.markdown(":material/lightbulb: **Nota analítica**")

        st.markdown(f"**O que mostra.** {n['o_que']}")

        st.markdown(f"**Insight.** {n['insight']}")

        st.markdown(f"**Por que este gráfico.** {n['por_que']}")





def grafico_com_nota(fig, chave: str, alt: str, proporcao=(2.3, 1)) -> None:

    col_g, col_n = st.columns(proporcao, gap="medium")

    with col_g:

        st.plotly_chart(fig, alt=alt, key=f"graf_{chave}", config={"displaylogo": False})

    with col_n:

        nota(chave)





def linha_kpis(df: pd.DataFrame, referencia: pd.DataFrame | None = None) -> None:

    k = analise.kpis(df)

    ref = analise.kpis(referencia) if referencia is not None and len(referencia) != len(df) else None



    def delta(chave, fmt, mult=1.0):

        if ref is None or pd.isna(k[chave]) or pd.isna(ref[chave]):

            return None

        return br(fmt.format((k[chave] - ref[chave]) * mult)) + " p.p."



    with st.container(horizontal=True):

        st.metric("Reservas", num(k["reservas"]), border=True,

                  help="Número de reservas (linhas) com os filtros atuais.")

        st.metric("Taxa de cancelamento", pct(k["taxa_cancelamento"]) if k["reservas"] else "—",

                  delta=delta("taxa_cancelamento", "{:+.1f}", 100), delta_color="inverse", border=True,

                  delta_description="vs. base completa" if ref else None,

                  help="Reservas canceladas (inclui não comparecimento) / total de reservas.")

        st.metric("Receita em reservas canceladas", f"€ {num(k['receita_cancelada'] / 1e6, 1)} mi",

                  border=True, help="Soma de diária média × noites das reservas canceladas (estimativa).")

        st.metric("Antecedência mediana", f"{k['antecedencia_mediana']:.0f} dias" if k["reservas"] else "—",

                  border=True, help="Metade das reservas é feita com menos dias de antecedência que isso.")





# --------------------------------------------------------------------------

# Relatório exportável

# --------------------------------------------------------------------------

def relatorio_html(df: pd.DataFrame, filtros: str) -> str:

    k = analise.kpis(df)

    figuras = [

        ("x1", graficos.fig_x1(df)), ("x2", graficos.fig_x2(df)), ("x3", graficos.fig_x3(df)),

        ("e1", graficos.fig_e1_antecedencia(df)), ("e2", graficos.fig_e2_sazonalidade(df)),

        ("e3", graficos.fig_e3_correlacao(df)), ("e4", graficos.fig_e4_adr(df)),

        ("e5", graficos.fig_e5_segmento_deposito(df)), ("e6", graficos.fig_e6_compromisso(df)),

    ]

    blocos = []

    for i, (chave, fig) in enumerate(figuras):

        n = notas.NOTAS[chave]

        fig.update_layout(template="plotly_white")

        blocos.append(

            f"<section><h2>{html.escape(n['titulo'])}</h2>"

            + fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)

            + f"<p><b>O que mostra.</b> {html.escape(n['o_que'])}</p>"

            + f"<p><b>Insight.</b> {html.escape(n['insight'])}</p>"

            + f"<p><b>Por que este gráfico.</b> {html.escape(n['por_que'])}</p></section>"

        )

    recs = "".join(f"<li>{html.escape(r)}</li>" for r in notas.RECOMENDACOES)

    taxa = f"{k['taxa_cancelamento']:.1%}" if k["reservas"] else "—"

    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">

<title>Relatório — cancelamentos de reservas</title>

<style>

body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#0b0b0b;line-height:1.5}}

h1{{margin-bottom:.2rem}} .sub{{color:#52514e}} .kpis{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}}

.kpi{{border:1px solid #e1e0d9;border-radius:8px;padding:.6rem 1rem;min-width:180px}} .kpi b{{font-size:1.5rem;display:block}}

section{{border-top:1px solid #e1e0d9;margin-top:2rem;padding-top:1rem}} .filtros{{background:#f9f9f7;padding:.6rem 1rem;border-radius:8px;font-size:.9rem}}

</style></head><body>

<h1>Cancelamento de reservas — relatório da seleção</h1>

<p class="sub">Gerado pelo dashboard em {dt.datetime.now():%d/%m/%Y %H:%M} · Projeto AVD 2026.2 · {", ".join(INTEGRANTES)}</p>

<p class="filtros"><b>Filtros:</b> {html.escape(filtros)}</p>

<p><b>Pergunta de pesquisa.</b> {html.escape(notas.PERGUNTA)}</p>

<p><b>Grande ideia.</b> {html.escape(notas.GRANDE_IDEIA)}</p>

<div class="kpis">

<div class="kpi">Reservas<b>{k['reservas']:,}</b></div>

<div class="kpi">Taxa de cancelamento<b>{taxa}</b></div>

<div class="kpi">Receita em canceladas<b>€ {k['receita_cancelada'] / 1e6:,.1f} mi</b></div>

<div class="kpi">Antecedência mediana<b>{k['antecedencia_mediana']:.0f} dias</b></div>

</div>

<h2>Recomendações para a gestão</h2><ul>{recs}</ul>

{''.join(blocos)}

<p class="sub">Fonte: Hotel Booking Demand (Antonio, Almeida &amp; Nunes, 2019, Data in Brief 22). Os números das notas

referem-se à base completa; os gráficos refletem os filtros acima.</p>

</body></html>"""

