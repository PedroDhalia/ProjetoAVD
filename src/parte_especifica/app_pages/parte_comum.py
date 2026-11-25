import streamlit as st

from src.parte_comum import interativo as I
from src.parte_especifica import app_comum


@st.cache_resource(show_spinner="Rodando o K-Means para k = 1 a 10 (só na primeira vez, ~30 s)…")
def _calcular():
    return I.calcular_tudo()


st.title("Parte Comum — K-Means no dataset Obesity", anchor=False)
st.caption("2.087 pessoas (após remover 24 duplicatas) · 15 variáveis padronizadas (z-score) · 7 classes de nível de "
           "obesidade. O K-Means nunca vê as classes: elas só são usadas depois, para avaliar os grupos.")

r = _calcular()
p1, p2 = r["passadas"]["livre"], r["passadas"]["fixo"]

with st.container(horizontal=True):
    st.metric("k sugerido pelo cotovelo", r["k_livre"], border=True,
              help="Ponto da curva de inércia mais distante da reta entre k = 1 e k = 10 (estável em 5 sementes).")
    st.metric("Acurácia — Passada 1", app_comum.pct(p1["acuracia"]), border=True, help=f"k = {p1['k']}")
    st.metric("Acurácia — Passada 2", app_comum.pct(p2["acuracia"]), border=True, help=f"k = {p2['k']} (nº real de classes)")
    st.metric("Linha de base", app_comum.pct(r["baseline"]), border=True,
              help="Acurácia de sempre 'chutar' a classe mais comum.")

secao = st.segmented_control(
    "Seção", ["Cotovelo", "Animação k = 2…10", "Passada 1 × Passada 2", "Matrizes e métricas", "Discussão crítica"],
    default="Cotovelo", key="pc_secao", persist_state="session", label_visibility="collapsed", required=True,
)

if secao == "Cotovelo":
    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1:
        st.plotly_chart(I.fig_cotovelo(r["curva"], r["k_livre"]), key="pc_cotovelo",
                        alt="Curva do cotovelo suave, com k = 4 escolhido")
    with c2:
        with st.container(border=True):
            st.markdown(":material/lightbulb: **Como o k foi escolhido**")
            st.markdown(
                "A curva de inércia cai de forma **suave**, sem um cotovelo nítido — sinal de que os dados não têm grupos "
                "bem separados. Para não escolher 'no olho', usamos o ponto mais distante da reta entre o primeiro e o "
                f"último ponto (*kneedle*): **k = {r['k_livre']}**, o mesmo em todas as sementes testadas. Depois dele, o "
                "ganho marginal cai de ~8% para ~6% por grupo extra."
            )
            st.markdown("**Padronização.** Sem o z-score, o peso (variância ≈ 686 kg²) dominaria a distância euclidiana "
                        "e as outras 14 variáveis seriam praticamente ignoradas.")
        st.dataframe(r["estabilidade"], hide_index=True, alt="Cotovelo escolhido em diferentes sementes",
                     column_config={"semente": "Semente", "k_cotovelo_1a10": "k (curva 1–10)",
                                    "k_cotovelo_2a10": "k (curva 2–10)"})
    st.dataframe(
        r["curva"], hide_index=True, alt="Inércia, silhueta e queda percentual por k",
        column_config={"k": "k", "inercia": st.column_config.NumberColumn("Inércia", format="%.0f"),
                       "silhueta": st.column_config.NumberColumn("Silhueta", format="%.3f"),
                       "queda_pct": st.column_config.NumberColumn("Queda vs k−1 (%)", format="%.1f"),
                       "distancia_corda": st.column_config.NumberColumn("Distância à reta", format="%.3f")},
    )

elif secao == "Animação k = 2…10":
    st.plotly_chart(I.fig_animacao(r), key="pc_anim", alt="Animação dos grupos do K-Means de k igual a 2 a 10")
    st.caption("Use ▶ Animar ou arraste o controle de k. Passe o mouse nos pontos para ver a classe real de cada pessoa.")
    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1:
        st.plotly_chart(I.fig_acuracia_por_k(r["por_k"], r["k_livre"], r["baseline"]), key="pc_acc_k",
                        alt="Acurácia cresce com k; ARI fica baixo")
    with c2:
        with st.container(border=True):
            st.markdown(":material/lightbulb: **O que observar**")
            st.markdown(
                "- Com **k = 2** o K-Means separa basicamente **leves (59 kg) × pesados (97 kg)**; com **k = 3** os "
                "pesados se dividem por **gênero** (99% homens × 96% mulheres).\n"
                "- A partir de **k = 4** surgem grupos de **atributos raros**: os 44 fumantes (k = 4) e, em k = 7, "
                "quem monitora calorias.\n"
                "- A acurácia mapeada cresce com k em boa parte porque mais grupos = mais 'vagas' para classes; o **ARI**, "
                "que desconta o acaso, sobe pouco."
            )

elif secao == "Passada 1 × Passada 2":
    y, coords, v = r["prep"]["y"], r["coords"], r["variancia_pca"]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.plotly_chart(I.fig_pca_passada(coords, p1["rotulos"], p1["centros_pca"],
                                          f"Passada 1 — k livre = {p1['k']}", v), key="pc_p1", alt="Grupos da passada 1")
    with c2:
        st.plotly_chart(I.fig_pca_passada(coords, p2["rotulos"], p2["centros_pca"],
                                          f"Passada 2 — k fixo = {p2['k']}", v), key="pc_p2", alt="Grupos da passada 2")
    c3, c4 = st.columns([1, 1], gap="medium")
    with c3:
        st.plotly_chart(I.fig_pca_classes(coords, y, v), key="pc_classes", alt="Classes reais na projeção PCA")
    with c4:
        with st.container(border=True):
            st.markdown(":material/compare_arrows: **Lado a lado**")
            st.markdown(
                f"| | Passada 1 | Passada 2 |\n|---|---|---|\n"
                f"| k | {p1['k']} | {p2['k']} |\n"
                f"| Acurácia | {p1['acuracia']:.1%} | {p2['acuracia']:.1%} |\n"
                f"| Precisão macro | {p1['precisao_macro']:.2f} | {p2['precisao_macro']:.2f} |\n"
                f"| Recall macro | {p1['recall_macro']:.2f} | {p2['recall_macro']:.2f} |\n"
                f"| ARI | {p1['ari']:.3f} | {p2['ari']:.3f} |\n"
                f"| Classes nunca previstas | {len(p1['classes_nunca_previstas'])} | {len(p2['classes_nunca_previstas'])} |"
            )
            st.markdown(
                "**Passada 1** separa **jovens leves**, **mulheres pesadas** (→ Obesidade III), **homens pesados** "
                "(→ Obesidade II) e **fumantes**. **Passada 2** refina, mas gasta dois grupos com atributos raros "
                "(fumantes e quem monitora calorias)."
            )
            st.caption("A projeção PCA explica só ~29% da variância: grupos que parecem sobrepostos aqui podem estar "
                       "separados em outras dimensões.")

elif secao == "Matrizes e métricas":
    escolha = st.segmented_control("Passada", ["Passada 1 (k livre)", "Passada 2 (k fixo)", "Lado a lado"],
                                   default="Lado a lado", key="pc_mat", required=True)
    alvos = {"Passada 1 (k livre)": [("Passada 1", p1)], "Passada 2 (k fixo)": [("Passada 2", p2)],
             "Lado a lado": [("Passada 1", p1), ("Passada 2", p2)]}[escolha]
    cols = st.columns(len(alvos), gap="medium")
    for col, (nome, av) in zip(cols, alvos):
        with col:
            st.plotly_chart(I.fig_contingencia(av["contingencia"], av["mapa"], f"{nome} (k = {av['k']}): contingência"),
                            key=f"pc_ct_{nome}", alt=f"Tabela de contingência da {nome}")
            st.plotly_chart(I.fig_matriz(av["matriz"], f"{nome}: matriz de confusão — acurácia {av['acuracia']:.1%}"),
                            key=f"pc_cm_{nome}", alt=f"Matriz de confusão da {nome}")
            st.dataframe(
                I.tabela_metricas(av), hide_index=True, alt=f"Precisão e recall por classe na {nome}",
                column_config={
                    "classe": "Classe", "suporte_real": "Reais", "previstos": "Previstos", "acertos": "Acertos",
                    "precisao": st.column_config.ProgressColumn("Precisão", min_value=0, max_value=1, format="%.2f"),
                    "recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.2f"),
                    "f1": st.column_config.NumberColumn("F1", format="%.2f"),
                },
            )
            st.caption("Precisão vazia = classe nunca prevista (0/0, indefinida). Métricas calculadas direto da matriz "
                       "e conferidas com o scikit-learn.")

else:
    st.subheader("Qual passada gerou grupos mais coerentes?", anchor=False)
    st.markdown(
        f"- **Com as classes reais**, a Passada 2 vence em tudo: acurácia {p2['acuracia']:.1%} vs {p1['acuracia']:.1%}, "
        f"ARI {p2['ari']:.2f} vs {p1['ari']:.2f}, e prevê 6 classes em vez de 3. Era esperado: com 7 grupos há mais "
        "'vagas' para classes.\n"
        "- **Internamente**, nenhuma é boa: silhueta ≈ 0,12–0,15 e cotovelo sem joelho claro. Os 4 grupos do cotovelo "
        "fazem sentido como **perfis** (jovens leves, mulheres pesadas, homens pesados, fumantes), mas não são os níveis "
        "de obesidade.\n"
        "- Os dados têm estrutura mais forte em **gênero + peso** do que nas 7 classes — em parte por um artefato do "
        "SMOTE: Obesidade II tem 295 homens e 2 mulheres; Obesidade III, 323 mulheres e 1 homem."
    )
    st.subheader("A granularidade das classes faz sentido?", anchor=False)
    c1, c2 = st.columns(2, gap="medium")
    for col, nome, av in [(c1, "Passada 1", p1), (c2, "Passada 2", p2)]:
        with col:
            st.markdown(f"**{nome} (k = {av['k']})** reavaliada contra classes mais grossas")
            st.dataframe(av["granularidade"], hide_index=True, alt=f"Granularidade das classes na {nome}",
                         column_config={"agrupamento_de_classes": "Agrupamento", "n_classes": "Nº classes",
                                        "acuracia_mapeada": st.column_config.NumberColumn("Acurácia", format="percent"),
                                        "ari": st.column_config.NumberColumn("ARI", format="%.3f"),
                                        "nmi": st.column_config.NumberColumn("NMI", format="%.3f"),
                                        "acuracia_chute_majoritario": st.column_config.NumberColumn("Linha de base", format="percent")})
    st.markdown(
        "- Juntar **Sobrepeso I + II** melhora a acurácia sem piorar o ARI: os dados **não distinguem** esses dois níveis "
        "(IMC 22,8–28,8 vs 25,7–30,4 — faixas sobrepostas; a OMS nem faz essa divisão).\n"
        "- Com as 4 classes da OMS a acurácia sobe para 55–58%, mas a linha de base também sobe (46,6%) e o ARI não "
        "melhora: juntar classes facilita a acurácia 'de graça'."
    )
    st.subheader("O que define a classe?", anchor=False)
    st.dataframe(r["sensibilidade"], hide_index=True, alt="K-Means com diferentes conjuntos de variáveis",
                 column_config={"conjunto": "Variáveis usadas (k = 7)", "n_variaveis": "Nº",
                                "acuracia_mapeada": st.column_config.NumberColumn("Acurácia", format="percent"),
                                "ari": st.column_config.NumberColumn("ARI", format="%.3f"),
                                "silhueta": st.column_config.NumberColumn("Silhueta", format="%.3f")})
    st.markdown(
        "Só **altura e peso** dão a mesma acurácia das 15 variáveis e uma silhueta 2,6× maior: a classe é, na prática, o "
        "**IMC discretizado** (correlação de Spearman 0,99). O K-Means mede semelhança global entre pessoas, por isso "
        "encontra outras estruturas reais (gênero, fumo) que não são a classe. Lembrete: a 'acurácia' aqui mede "
        "**coincidência** entre grupos e classes, não acerto de previsão."
    )
