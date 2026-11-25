import streamlit as st

from src.parte_especifica import app_comum, notas

bruto, df, _ = app_comum.carregar()

st.title("Por que os hóspedes cancelam?", anchor=False)
st.caption("Cancelamento de reservas em dois hotéis de Portugal — um hotel urbano em Lisboa e um resort no Algarve — "
           "com chegadas de julho de 2015 a agosto de 2017.")

app_comum.linha_kpis(df)

col1, col2 = st.columns([1.4, 1], gap="large")
with col1:
    with st.container(border=True):
        st.markdown(":material/help: **Pergunta de pesquisa**")
        st.markdown(f"### {notas.PERGUNTA}")
        st.markdown(f"**Hipótese.** {notas.HIPOTESE}")
    with st.container(border=True):
        st.markdown(":material/insights: **Grande ideia (resposta curta)**")
        st.markdown(notas.GRANDE_IDEIA)
        st.page_link("app_pages/historia.py", label="Ver a história completa e as recomendações",
                     icon=":material/arrow_forward:")
with col2:
    with st.container(border=True):
        st.markdown(":material/groups: **Público-alvo**")
        st.markdown(notas.PUBLICO)
    with st.container(border=True):
        st.markdown(":material/table_chart: **Os dados**")
        st.markdown(
            f"""
- **{len(bruto):,} reservas × {bruto.shape[1]} variáveis** (quantitativas e qualitativas) — Kaggle
  *Hotel Booking Demand*, publicado por Antonio, Almeida & Nunes (2019, *Data in Brief*).
- Tipo de problema: **classificação** — a variável-alvo é `is_canceled` (1 = cancelada).
- Após o tratamento: **{len(df):,} reservas** (180 registros sem nenhum hóspede removidos; nenhum dado
  faltante descartado).
""".replace(",", ".")
        )

st.subheader("Como navegar", anchor=False)
with st.container(horizontal=True):
    for pagina, rotulo, icone, texto in [
        ("app_pages/historia.py", "História: o que fazer", ":material/auto_stories:",
         "As 3 visualizações explanatórias para a gestão e as recomendações."),
        ("app_pages/exploracao.py", "Exploração", ":material/query_stats:",
         "As 6 visualizações exploratórias que deram origem à história, com filtros."),
        ("app_pages/explorador.py", "Explorador de variáveis", ":material/scatter_plot:",
         "Cruze quaisquer duas variáveis e veja a força da relação."),
        ("app_pages/simulador.py", "Simulador de risco", ":material/online_prediction:",
         "O risco é previsível? Teste uma reserva hipotética."),
        ("app_pages/qualidade.py", "Qualidade dos dados", ":material/fact_check:",
         "Faltantes e seus mecanismos, outliers, repetições e tratamentos."),
        ("app_pages/parte_comum.py", "Parte Comum: K-Means", ":material/bubble_chart:",
         "Obesity: cotovelo, animação, matrizes de confusão e discussão."),
    ]:
        with st.container(border=True, width=300):
            st.page_link(pagina, label=rotulo, icon=icone)
            st.caption(texto)
