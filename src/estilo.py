"""Paleta de cores e estilo visual compartilhados pelo projeto.

As cores seguem uma regra única em todos os gráficos (matplotlib e plotly):

* Azul  = reserva NÃO cancelada / referência.
* Laranja = reserva cancelada / risco. Nos gráficos explanatórios o laranja é
  o atributo pré-atentivo de destaque e o restante fica em cinza.
* Hotéis: violeta (urbano) e verde-água (resort).
* Classes de obesidade: rampa ordinal (azul = abaixo do peso, cinza = normal,
  laranja claro -> escuro = sobrepeso -> obesidade III).

As paletas categóricas foram checadas para daltonismo (protanopia e
deuteranopia) com o validador da skill de visualização usada no projeto.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Tinta e superfícies -----------------------------------------------------
TINTA = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
TEXTO_MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
SUPERFICIE = "#fcfcfb"

# --- Cores com significado fixo ---------------------------------------------
AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA_CONTEXTO = "#c9c8c1"
VERMELHO = "#e34948"

COR_STATUS = {"Não cancelada": AZUL, "Cancelada": LARANJA}
COR_HOTEL = {"Hotel urbano": "#6250d6", "Hotel resort": "#199e70"}

# Paleta categórica em ordem fixa (nunca reciclada).
CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Só para a animação do K-Means com k até 10.
CATEGORICA_ESTENDIDA = CATEGORICA + ["#104281", "#8a6d3b"]

# Rampa sequencial (uma só matiz) para taxas de cancelamento.
SEQ_LARANJA = [
    "#fdf3ec", "#fbe0d0", "#f8c7ab", "#f4aa82", "#ee8c5b",
    "#e0703f", "#c4521f", "#983b14", "#7a2f10", "#66250b",
]
SEQ_AZUL = [
    "#eef5fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#6da7ec",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281",
]
# Divergente azul <-> vermelho com ponto neutro cinza (correlações).
DIVERGENTE = ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32a29"]

# Classes do dataset Obesity em ordem de IMC.
CORES_CLASSE_OBESIDADE = [
    "#2a78d6",  # Abaixo do peso
    "#898781",  # Peso normal (ponto neutro)
    "#ee9a6e",  # Sobrepeso I
    "#e0703f",  # Sobrepeso II
    "#c4521f",  # Obesidade I
    "#983b14",  # Obesidade II
    "#66250b",  # Obesidade III
]


def cmap_sequencial(cores: list[str] = SEQ_LARANJA, nome: str = "seq"):
    """Colormap contínuo do matplotlib a partir de uma lista de cores."""
    return mpl.colors.LinearSegmentedColormap.from_list(nome, cores)


def cmap_divergente():
    return mpl.colors.LinearSegmentedColormap.from_list("div", DIVERGENTE)


def aplicar_estilo_mpl() -> None:
    """Estilo limpo para os gráficos estáticos (relatório)."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": EIXO,
            "axes.labelcolor": TEXTO_SECUNDARIO,
            "axes.titlecolor": TINTA,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.7,
            "xtick.color": TEXTO_SECUNDARIO,
            "ytick.color": TEXTO_SECUNDARIO,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "axes.prop_cycle": mpl.cycler(color=CATEGORICA),
        }
    )


# Layout padrão dos gráficos plotly (o dashboard aplica o tema do Streamlit por cima).
LAYOUT_PLOTLY = dict(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13),
    margin=dict(l=10, r=10, t=60, b=10),
    hoverlabel=dict(font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
)
