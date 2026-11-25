import streamlit as st

from src.parte_especifica import app_comum, graficos, notas

df = app_comum.barra_lateral_filtros()
_, base, _ = app_comum.carregar()

st.title("O risco de cancelamento já está escrito na reserva", anchor=False)
st.caption("Visualizações explanatórias para a gestão do hotel. Todos os gráficos respondem aos filtros da barra lateral.")

with st.container(border=True):
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        st.markdown(":material/insights: **Grande ideia**")
        st.markdown(f"#### {notas.GRANDE_IDEIA}")
    with c2:
        st.markdown(":material/groups: **Para quem**")
        st.markdown(notas.PUBLICO)

app_comum.linha_kpis(df, referencia=base)

if df.empty:
    st.warning("Nenhuma reserva com os filtros atuais. Ajuste os filtros na barra lateral.", icon=":material/warning:")
    st.stop()

st.header("1 · Antecedência: o principal sinal de risco", anchor=False)
comparar = st.toggle("Comparar lado a lado: com e sem as linhas repetidas", key="hist_comparar",
                     help="Teste de robustez: as conclusões se mantêm sem os quartos repetidos de reservas de grupo?")
if comparar:
    sem_rep = df[df["duplicada"] == 0]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(graficos.fig_x1(df), key="x1_com", alt="Taxa de cancelamento por antecedência, base com repetidas")
        st.caption(f"Com linhas repetidas: {len(df):,} reservas".replace(",", "."))
    with c2:
        st.plotly_chart(graficos.fig_x1(sem_rep), key="x1_sem", alt="Taxa de cancelamento por antecedência, sem repetidas")
        st.caption(f"Sem linhas repetidas: {len(sem_rep):,} reservas — a escada continua subindo".replace(",", "."))
    app_comum.nota("x1")
else:
    app_comum.grafico_com_nota(graficos.fig_x1(df), "x1",
                               alt="Taxa de cancelamento sobe com a antecedência: de 10% até 7 dias a 68% acima de 1 ano")

st.header("2 · Pedidos especiais: o sinal de compromisso", anchor=False)
app_comum.grafico_com_nota(graficos.fig_x2(df), "x2",
                           alt="Em toda faixa de antecedência, reservas com pedido especial cancelam metade ou menos")

st.header("3 · Onde agir: o perfil que concentra os cancelamentos", anchor=False)
app_comum.grafico_com_nota(graficos.fig_x3(df), "x3",
                           alt="Perfil com mais de 90 dias e nenhum pedido especial: 27% das reservas e 47% dos cancelamentos")

st.header("Recomendações para a gestão", anchor=False)
for i, rec in enumerate(notas.RECOMENDACOES, start=1):
    st.markdown(f"**{i}.** {rec}")

with st.expander("Raciocínio do storytelling: da exploração à explicação", icon=":material/route:"):
    for titulo, texto in notas.RACIOCINIO:
        st.markdown(f"**{titulo}**")
        st.markdown(texto)
