"""Dashboard do Projeto Final de AVD 2026.2.

Execução (a partir da raiz do repositório):
    streamlit run src/parte_especifica/dashboard.py

Parte Específica: cancelamento de reservas em dois hotéis portugueses
(Hotel Booking Demand). Parte Comum: K-Means no dataset Obesity (UCI 544).
"""

import sys
from pathlib import Path

import streamlit as st

# Permite "from src..." quando o Streamlit roda este arquivo diretamente.
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.parte_especifica import app_comum  # noqa: E402

st.set_page_config(
    page_title="Cancelamentos de reservas · AVD 2026.2",
    page_icon=":material/hotel:",
    layout="wide",
)
app_comum.iniciar_estado()

PAGINAS = Path(__file__).parent / "app_pages"

pagina = st.navigation(
    {
        "Parte Específica — Hotéis": [
            st.Page(PAGINAS / "inicio.py", title="Visão geral", icon=":material/home:", default=True,
                    help="Pergunta de pesquisa, público-alvo, principais números e como navegar pelo dashboard."),
            st.Page(PAGINAS / "historia.py", title="História: o que fazer", icon=":material/auto_stories:",
                    help="Visualizações explanatórias para a gestão: onde está o risco de cancelamento e que ações tomar."),
            st.Page(PAGINAS / "exploracao.py", title="Exploração", icon=":material/query_stats:",
                    help="As seis visualizações exploratórias com notas analíticas. Todas respondem aos filtros da barra lateral."),
            st.Page(PAGINAS / "explorador.py", title="Explorador de variáveis", icon=":material/scatter_plot:",
                    help="Escolha quaisquer duas variáveis: o gráfico e a medida de associação se ajustam aos tipos delas."),
            st.Page(PAGINAS / "simulador.py", title="Simulador de risco", icon=":material/online_prediction:",
                    help="Modelo de classificação: desempenho, variáveis mais importantes e simulação do risco de uma reserva."),
            st.Page(PAGINAS / "qualidade.py", title="Qualidade dos dados", icon=":material/fact_check:",
                    help="Estrutura, valores ausentes e seus mecanismos, tratamentos aplicados, outliers e linhas repetidas."),
        ],
        "Parte Comum — Obesity": [
            st.Page(PAGINAS / "parte_comum.py", title="K-Means: duas passadas", icon=":material/bubble_chart:",
                    help="Cotovelo, animação dos agrupamentos de k = 2 a 10, comparação lado a lado das passadas, "
                         "matrizes de confusão e discussão crítica."),
        ],
    },
    position="sidebar",
)

pagina.run()

# Créditos no fim da barra lateral (depois dos filtros de cada página).
with st.sidebar:
    st.divider()
    st.caption("Projeto Final · Análise e Visualização de Dados · 2026.2")
    st.caption(" · ".join(app_comum.INTEGRANTES))
