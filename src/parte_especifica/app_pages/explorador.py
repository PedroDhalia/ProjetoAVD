import streamlit as st

from src.parte_especifica import analise, app_comum, dados, graficos

df = app_comum.barra_lateral_filtros()

st.title("Explorador de variáveis", anchor=False)
st.caption("Escolha duas variáveis quaisquer. O tipo de gráfico e a medida de associação são escolhidos "
           "automaticamente de acordo com o tipo de cada uma (numérica ou categórica).")

opcoes = ["is_canceled"] + graficos.NUMERICAS_EXPLORADOR + graficos.CATEGORICAS_EXPLORADOR


def _rotulo(c: str) -> str:
    tipo = "numérica" if c in graficos.NUMERICAS_EXPLORADOR else ("alvo" if c == "is_canceled" else "categórica")
    return f"{dados.ROTULOS.get(c, c)} · {c} ({tipo})"


def _trocar():
    st.session_state["exp_x"], st.session_state["exp_y"] = st.session_state["exp_y"], st.session_state["exp_x"]


if "exp_x" not in st.session_state:
    st.session_state["exp_x"] = "lead_time"
    st.session_state["exp_y"] = "is_canceled"

with st.container(horizontal=True, vertical_alignment="bottom"):
    st.selectbox("Variável X", opcoes, key="exp_x", format_func=_rotulo, persist_state="session", width=420)
    st.button("Trocar", icon=":material/swap_horiz:", on_click=_trocar, help="Inverte X e Y")
    st.selectbox("Variável Y", opcoes, key="exp_y", format_func=_rotulo, persist_state="session", width=420)

x, y = st.session_state["exp_x"], st.session_state["exp_y"]
if df.empty:
    st.warning("Nenhuma reserva com os filtros atuais.", icon=":material/warning:")
    st.stop()
if x == y:
    st.info("Escolha duas variáveis diferentes.", icon=":material/info:")
    st.stop()

nome, valor, forca = analise.associacao(df, x, y, graficos.NUMERICAS_EXPLORADOR)
col_g, col_m = st.columns([2.6, 1], gap="medium")
with col_g:
    st.plotly_chart(graficos.fig_explorador(df, x, y), key="graf_explorador", alt=f"Relação entre {x} e {y}")
with col_m:
    st.metric(nome, app_comum.br(f"{valor:+.3f}" if "Spearman" in nome else f"{valor:.3f}"), border=True,
              help="Spearman: −1 a 1 (sinal = direção). η² e V de Cramér: 0 a 1 (só intensidade).")
    st.markdown(f"**Leitura:** {forca}.")
    with st.container(border=True):
        st.markdown(":material/school: **Como interpretar**")
        st.markdown(
            "- **Spearman ρ** (numérica × numérica): mede se uma cresce quando a outra cresce, mesmo sem relação linear.\n"
            "- **η²** (categórica × numérica): fração da variação da numérica explicada pelas categorias.\n"
            "- **V de Cramér** (categórica × categórica): 0 = independentes, 1 = uma determina a outra.\n"
            "- Faixas usadas: < 0,1 desprezível · < 0,3 fraca · < 0,5 moderada · ≥ 0,5 forte "
            "(para η² comparamos √η²)."
        )
    st.caption(f"{len(df):,} reservas consideradas.".replace(",", "."))

app_comum.nota("explorador")
