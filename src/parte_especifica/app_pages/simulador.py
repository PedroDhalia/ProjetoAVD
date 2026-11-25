import numpy as np
import pandas as pd
import streamlit as st

from src.parte_especifica import app_comum, dados, graficos, modelo

_, df, _ = app_comum.carregar()
res = app_comum.obter_modelo()
m = res.metricas

st.title("O risco é previsível? Simulador de cancelamento", anchor=False)
st.caption("Modelo de classificação (Gradient Boosting, scikit-learn) treinado só com informações conhecidas no "
           "momento da reserva. Treino: chegadas de jul/2015 a dez/2016 · Teste: chegadas de 2017.")

with st.container(horizontal=True):
    st.metric("AUC (teste 2017)", app_comum.num(m["auc"], 2), border=True,
              help="Probabilidade de o modelo dar risco maior a uma reserva cancelada do que a uma não cancelada.")
    st.metric("Acurácia", app_comum.pct(m["acuracia"]), delta=app_comum.br(f"{100 * (m['acuracia'] - m['acuracia_baseline']):+.1f}") + " p.p.",
              delta_description="vs. prever 'nunca cancela'", border=True)
    st.metric("Precisão", app_comum.pct(m["precisao"]), border=True,
              help="Das reservas apontadas como risco, quantas de fato cancelaram.")
    st.metric("Recall", app_comum.pct(m["recall"]), border=True,
              help="Dos cancelamentos reais, quantos o modelo apontou.")

aba = st.segmented_control("Seção", ["Simular uma reserva", "Desempenho do modelo"], default="Simular uma reserva",
                           key="sim_aba", persist_state="session", label_visibility="collapsed", required=True)

if aba == "Simular uma reserva":
    tipica = modelo.reserva_tipica(df)
    cats = res.categorias
    col_in, col_out = st.columns([1, 1.5], gap="large")
    with col_in:
        with st.container(border=True):
            st.markdown(":material/edit_note: **Características da reserva**")
            hotel = st.segmented_control("Hotel", cats["hotel"], default="Hotel urbano", key="sim_hotel", required=True)
            lead = st.slider("Antecedência (dias)", 0, 500, 120, key="sim_lead")
            pedidos = st.segmented_control("Pedidos especiais", [0, 1, 2, 3, 4, 5], default=0, key="sim_pedidos",
                                           required=True)
            deposito = st.segmented_control("Depósito", cats["deposit_type"], default="Sem depósito", key="sim_dep",
                                            required=True)
            c1, c2 = st.columns(2)
            with c1:
                segmento = st.selectbox("Segmento", cats["market_segment"],
                                        index=cats["market_segment"].index("Agência online"), key="sim_seg")
                cliente = st.selectbox("Tipo de cliente", cats["customer_type"],
                                       index=cats["customer_type"].index("Individual"), key="sim_cli")
                pais = st.selectbox("País de origem", cats["pais_grupo"],
                                    index=cats["pais_grupo"].index("Portugal"), key="sim_pais")
            with c2:
                mes = st.selectbox("Mês de chegada", list(range(1, 13)), index=7, key="sim_mes",
                                   format_func=lambda i: dados.MESES_PT[i - 1])
                adr = st.number_input("Diária (€)", 0, 500, 100, step=5, key="sim_adr")
                noites = st.number_input("Noites", 1, 30, 3, key="sim_noites")
            c3, c4 = st.columns(2)
            with c3:
                recorrente = st.toggle("Hóspede recorrente", key="sim_rec")
            with c4:
                cancel_ant = st.number_input("Cancelamentos anteriores", 0, 10, 0, key="sim_cancant")

    canal = {"Agência online": "Agência/operadora", "Agência/operadora offline": "Agência/operadora",
             "Grupos": "Agência/operadora", "Direto": "Direto", "Corporativo": "Corporativo",
             "Cortesia": "Direto", "Aviação": "Corporativo"}.get(segmento, tipica["distribution_channel"])
    reserva = tipica | {
        "hotel": hotel, "lead_time": lead, "total_of_special_requests": pedidos, "deposit_type": deposito,
        "market_segment": segmento, "distribution_channel": canal, "customer_type": cliente, "pais_grupo": pais,
        # Divide as noites em fim de semana / semana na proporção do calendário (2 de cada 7).
        "mes_num": mes, "adr": adr, "stays_in_weekend_nights": round(noites * 2 / 7),
        "stays_in_week_nights": noites - round(noites * 2 / 7),
        "is_repeated_guest": int(recorrente), "previous_cancellations": cancel_ant,
        "tem_agente": int(segmento in ("Agência online", "Agência/operadora offline", "Grupos")),
    }
    linha = pd.DataFrame([reserva])
    proba = float(modelo.prever(res, linha)[0])
    base = m["taxa_cancelamento_treino"]

    grade = pd.DataFrame([reserva] * 101)
    grade["lead_time"] = np.linspace(0, 500, 101).round()
    curva = modelo.prever(res, grade)

    with col_out:
        if proba >= 0.6:
            faixa, icone, cor, acao = ("Alto", ":material/error:", "red",
                                       "Pedir depósito ou reconfirmação próximo à data; considerar no overbooking.")
        elif proba >= 0.35:
            faixa, icone, cor, acao = ("Médio", ":material/warning:", "orange",
                                       "Contato de pré-chegada para estimular pedidos especiais e confirmar a estadia.")
        else:
            faixa, icone, cor, acao = ("Baixo", ":material/check_circle:", "green",
                                       "Nenhuma ação especial: reserva com perfil de baixo risco.")
        with st.container(border=True):
            c1, c2 = st.columns([1, 1.6], vertical_alignment="center")
            with c1:
                st.metric("Risco estimado de cancelar", f"{proba:.0%}",
                          delta=f"{100 * (proba - base):+.0f} p.p.", delta_color="inverse",
                          delta_description="vs. média do treino")
            with c2:
                st.markdown(f"#### :{cor}[{icone} Risco {faixa.lower()}]")
                st.markdown(f"**Ação sugerida:** {acao}")
        st.plotly_chart(graficos.fig_curva_simulador(grade["lead_time"].to_numpy(), curva, lead, proba, base),
                        key="graf_sim", alt="Risco estimado da reserva em função da antecedência")
        st.caption("Faixas de risco: baixo < 35% ≤ médio < 60% ≤ alto. As demais características (adultos, refeição, "
                   "tipo de quarto etc.) ficam no valor mais comum da base. Estimativa estatística, não uma certeza: "
                   "use para priorizar ações, não para recusar hóspedes.")

else:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.plotly_chart(graficos.fig_roc(res.roc, m["auc"]), key="graf_roc", alt="Curva ROC do modelo, AUC 0,84")
    with c2:
        st.plotly_chart(graficos.fig_matriz_modelo(res.matriz_confusao), key="graf_cm",
                        alt="Matriz de confusão do modelo no teste de 2017")
    app_comum.nota("m2")
    app_comum.grafico_com_nota(graficos.fig_importancia(res.importancia), "m1",
                               alt="Segmento, país, depósito, pedidos especiais e antecedência são as variáveis mais importantes")
    with st.expander("Variáveis excluídas do modelo e por quê", icon=":material/block:"):
        st.markdown(
            "- `reservation_status`, `reservation_status_date`: são o próprio desfecho (vazamento direto).\n"
            "- `assigned_room_type`: definido no check-in — quem cancelou nunca recebe quarto (vazamento).\n"
            "- `booking_changes`: acontece ao longo do tempo, depois da reserva.\n"
            "- `required_car_parking_spaces`: 0% de cancelamento entre quem pediu vaga — indício de registro na "
            "chegada (vazamento).\n"
            "- `arrival_date_year`, IDs de `agent` e `company`: não generalizam para o futuro.\n\n"
            f"Treino: {res.n_treino:,} reservas · Teste: {res.n_teste:,} reservas. A divisão por data (e não "
            "aleatória) simula o uso real e evita que quartos repetidos de um mesmo grupo caiam ao mesmo tempo no "
            "treino e no teste.".replace(",", ".")
        )
