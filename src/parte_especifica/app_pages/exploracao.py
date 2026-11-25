import streamlit as st

from src.parte_especifica import app_comum, graficos

df = app_comum.barra_lateral_filtros()
_, base, _ = app_comum.carregar()

st.title("Exploração dos dados", anchor=False)
st.caption("As seis visualizações exploratórias que levaram à história. Use os filtros da barra lateral para comparar "
           "hotéis, anos, temporadas, segmentos e tipos de cliente — todos os gráficos se atualizam juntos.")

app_comum.linha_kpis(df, referencia=base)

if df.empty:
    st.warning("Nenhuma reserva com os filtros atuais. Ajuste os filtros na barra lateral.", icon=":material/warning:")
    st.stop()

secao = st.segmented_control(
    "Seção", ["Distribuições", "Tempo e preço", "Relações"], default="Distribuições", key="expl_secao",
    persist_state="session", label_visibility="collapsed", required=True,
)

if secao == "Distribuições":
    st.subheader("E1 · Antecedência das reservas", anchor=False)
    app_comum.grafico_com_nota(graficos.fig_e1_antecedencia(df), "e1",
                               alt="Distribuição da antecedência: canceladas têm mediana de 113 dias, não canceladas 45")
    st.subheader("E6 · Sinais de relacionamento do hóspede", anchor=False)
    app_comum.grafico_com_nota(graficos.fig_e6_compromisso(df), "e6",
                               alt="Pedidos especiais, hóspede recorrente e reserva sem agente reduzem o cancelamento")
elif secao == "Tempo e preço":
    st.subheader("E2 · Sazonalidade", anchor=False)
    app_comum.grafico_com_nota(graficos.fig_e2_sazonalidade(df), "e2",
                               alt="Volume e taxa de cancelamento por mês; hotel urbano cancela mais o ano todo")
    st.subheader("E4 · Diária média por mês", anchor=False)
    app_comum.grafico_com_nota(graficos.fig_e4_adr(df), "e4",
                               alt="Diária média do resort vai de cerca de 48 euros em janeiro a 188 em agosto")
else:
    st.subheader("E3 · Correlações", anchor=False)
    app_comum.grafico_com_nota(graficos.fig_e3_correlacao(df), "e3",
                               alt="Antecedência é a variável mais associada ao cancelamento; pedidos especiais a mais protetora",
                               proporcao=(2.6, 1))
    st.subheader("E5 · Segmento × tipo de depósito", anchor=False)
    app_comum.grafico_com_nota(graficos.fig_e5_segmento_deposito(df), "e5",
                               alt="Reservas não reembolsáveis cancelam cerca de 99%, concentradas em grupos e operadoras")
