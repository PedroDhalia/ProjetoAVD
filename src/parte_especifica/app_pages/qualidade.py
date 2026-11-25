import streamlit as st

from src.parte_especifica import analise, app_comum, dados, graficos

bruto, df, log = app_comum.carregar()

st.title("Qualidade dos dados", anchor=False)
st.caption("Estrutura, valores ausentes (e por que estão ausentes), inconsistências, outliers e linhas repetidas. "
           "Esta página usa a base completa, sem filtros.")

secao = st.segmented_control(
    "Seção", ["Estrutura", "Faltantes", "Outliers", "Linhas repetidas", "Tratamentos"], default="Estrutura",
    key="qual_secao", persist_state="session", label_visibility="collapsed", required=True,
)

if secao == "Estrutura":
    estrutura = dados.tabela_estrutural(bruto)
    tipos = estrutura["tipo_estatistico"]
    with st.container(horizontal=True):
        st.metric("Linhas (reservas)", f"{len(bruto):,}".replace(",", "."), border=True)
        st.metric("Colunas originais", bruto.shape[1], border=True)
        st.metric("Quantitativas", int(tipos.str.startswith("Quantitativa").sum()), border=True,
                  help="Contagens, antecedência, diária e componentes numéricos da data.")
        st.metric("Qualitativas", int(tipos.str.startswith("Qualitativa").sum()), border=True,
                  help="Hotel, segmento, canal, depósito, país, tipos de quarto, IDs, alvo etc. (+1 coluna de data)")
    st.dataframe(
        estrutura, hide_index=True, alt="Dicionário de variáveis com tipo e faltantes",
        column_config={
            "coluna": "Coluna", "descricao": "Descrição", "tipo_estatistico": "Tipo estatístico",
            "dtype_pandas": "Tipo no pandas", "valores_unicos": "Valores únicos", "faltantes": "Faltantes (NaN)",
        },
    )
    st.info(
        "`reservation_status` e `reservation_status_date` registram o DESFECHO da reserva (check-out, cancelada, "
        "não compareceu). Elas são um vazamento do alvo e não entram em nenhuma análise de causa nem no modelo.",
        icon=":material/warning:", title="Atenção ao vazamento",
    )

elif secao == "Faltantes":
    col_g, col_n = st.columns([2.2, 1], gap="medium")
    with col_g:
        st.plotly_chart(graficos.fig_faltantes(bruto), key="graf_faltantes", alt="Faltantes por variável e mecanismo")
    with col_n:
        app_comum.nota("q1_faltantes")
    st.subheader("Evidências do mecanismo de ausência", anchor=False)
    st.caption("Comparamos as reservas COM e SEM o valor ausente. Se diferem em variáveis observadas, a ausência não é "
               "aleatória (MAR); se a ausência significa 'não se aplica', é estrutural.")
    evid = dados.evidencias_mecanismo(bruto)
    cols = st.columns(4)
    for col, chave in zip(cols, ["country", "agent", "company", "children"]):
        with col:
            st.markdown(f"**{chave}**")
            st.dataframe(
                evid[chave], hide_index=True, alt=f"Comparação ausente x presente para {chave}",
                column_config={
                    "taxa_cancelamento": st.column_config.NumberColumn("Cancel.", format="percent"),
                    "pct_resort": st.column_config.NumberColumn("% resort", format="percent"),
                },
            )
    st.markdown(f"Taxa de cancelamento geral: **{bruto['is_canceled'].mean():.1%}**.")

elif secao == "Outliers":
    var = st.segmented_control(
        "Variável", ["lead_time", "adr", "noites_total", "days_in_waiting_list", "adults", "booking_changes"],
        default="adr", key="qual_var", format_func=lambda c: dados.ROTULOS.get(c, c), required=True,
    )
    col_g, col_n = st.columns([2.2, 1], gap="medium")
    with col_g:
        st.plotly_chart(graficos.fig_outliers(df, var), key="graf_outliers", alt=f"Boxplot e outliers de {var}")
        st.caption("Dados após o tratamento (os erros de digitação já foram corrigidos).")
    with col_n:
        app_comum.nota("q2_outliers")
    tab = analise.tabela_outliers(df, ["lead_time", "adr", "noites_total", "adults", "children", "babies",
                                       "days_in_waiting_list", "previous_cancellations", "booking_changes",
                                       "total_of_special_requests"])
    st.dataframe(tab, hide_index=True, alt="Outliers pela regra de Tukey por variável",
                 column_config={"pct_outliers": st.column_config.NumberColumn("% outliers", format="%.2f%%")})
    st.markdown(
        "**Erros corrigidos:** diária de 5.400 € (a 2ª maior é 510 €) e de −6,38 €; 2 reservas com 9–10 bebês e 1 com "
        "10 crianças acompanhadas de 1–2 adultos. **Mantidos:** antecedências de até 737 dias (operadoras bloqueiam "
        "quartos com até 2 anos), diárias altas no verão do resort, grupos de até 55 adultos e estadias longas. "
        "**Atenção:** em variáveis discretas com IQR = 0 (adultos, alterações, cancelamentos anteriores) a regra de "
        "Tukey marca como outlier tudo o que difere do valor mais comum — ali ela não indica erro."
    )

elif secao == "Linhas repetidas":
    sens = analise.sensibilidade_duplicatas(df)
    st.markdown(
        f"**{int(df['duplicada'].sum()):,} linhas ({df['duplicada'].mean():.1%})** são idênticas a outra em todas as "
        "colunas. Como o dataset não tem número de reserva, não dá para saber se são cópias ou quartos diferentes de "
        "uma mesma reserva de grupo. A concentração em 'Não reembolsável' (98% das linhas desse tipo se repetem) e em "
        "Grupos indica a segunda opção: **cada linha é um quarto**. Por isso mantivemos as linhas e oferecemos o "
        "botão **Remover linhas repetidas** na barra lateral das páginas de análise.".replace(",", ".")
    )
    pct = ["taxa_geral", "taxa_0_7_dias", "taxa_mais_1_ano", "taxa_sem_pedido", "taxa_com_pedido",
           "taxa_nao_reembolsavel"]
    st.dataframe(
        sens, hide_index=True, alt="Indicadores com e sem linhas repetidas",
        column_config={c: st.column_config.NumberColumn(c.replace("_", " ").replace("taxa ", "taxa: "), format="percent")
                       for c in pct} | {"base": "Base", "reservas": "Reservas"},
    )
    st.success(
        "Sem as repetidas a taxa geral cai de 37% para 28% (os lotes de grupos cancelados pesam menos), mas TODAS as "
        "conclusões se mantêm: o cancelamento continua subindo com a antecedência (8% → 41%) e quem faz pedido especial "
        "continua cancelando menos (22% vs. 33%).", icon=":material/check_circle:", title="Conclusões robustas",
    )

else:
    st.dataframe(
        log, hide_index=True, alt="Log de tratamentos aplicados aos dados",
        column_config={"etapa": "Etapa", "colunas": "Colunas", "linhas_afetadas": "Linhas afetadas",
                       "diagnostico": st.column_config.TextColumn("Diagnóstico", width="large"),
                       "tratamento": st.column_config.TextColumn("Tratamento", width="large")},
        row_height=110,
    )
    st.caption("Nenhum dado faltante foi simplesmente removido. As únicas linhas excluídas (180) são registros "
               "impossíveis — reservas sem nenhum hóspede.")
