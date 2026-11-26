"""Carga, limpeza e engenharia de atributos do dataset Hotel Booking Demand.

Fonte: Antonio, Almeida & Nunes (2019), "Hotel booking demand datasets",
Data in Brief 22. Versão Kaggle (jessemostipak/hotel-booking-demand).

Cada linha é uma reserva com chegada prevista entre 01/07/2015 e 31/08/2017
em dois hotéis de Portugal: um hotel urbano (Lisboa) e um resort (Algarve).

Convenção: os nomes ORIGINAIS das colunas são mantidos (para bater com a
documentação do dataset). Os valores categóricos são traduzidos para português
e as colunas novas têm nomes em português, sem acentos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_CSV = RAIZ / "data" / "parte_especifica" / "hotel_bookings.csv"

# --------------------------------------------------------------------------
# Traduções e rótulos
# --------------------------------------------------------------------------
MESES_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MESES_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

TRADUCOES = {
    "hotel": {"City Hotel": "Hotel urbano", "Resort Hotel": "Hotel resort"},
    "meal": {
        "BB": "Café da manhã (BB)",
        "HB": "Meia pensão (HB)",
        "FB": "Pensão completa (FB)",
        "SC": "Sem refeição (SC)",
        "Undefined": "Sem refeição (SC)",  # documentação: Undefined e SC são a mesma coisa
    },
    "market_segment": {
        "Online TA": "Agência online",
        "Offline TA/TO": "Agência/operadora offline",
        "Groups": "Grupos",
        "Direct": "Direto",
        "Corporate": "Corporativo",
        "Complementary": "Cortesia",
        "Aviation": "Aviação",
    },
    "distribution_channel": {
        "TA/TO": "Agência/operadora",
        "Direct": "Direto",
        "Corporate": "Corporativo",
        "GDS": "GDS",
    },
    "deposit_type": {
        "No Deposit": "Sem depósito",
        "Non Refund": "Não reembolsável",
        "Refundable": "Reembolsável",
    },
    "customer_type": {
        "Transient": "Individual",
        "Transient-Party": "Individual vinculado",
        "Contract": "Contrato",
        "Group": "Grupo",
    },
    "reservation_status": {"Check-Out": "Check-out", "Canceled": "Cancelada", "No-Show": "Não compareceu"},
}

PAISES_PT = {
    "PRT": "Portugal", "GBR": "Reino Unido", "FRA": "França", "ESP": "Espanha", "DEU": "Alemanha",
    "ITA": "Itália", "IRL": "Irlanda", "BEL": "Bélgica", "BRA": "Brasil", "NLD": "Holanda",
    "USA": "EUA", "CHE": "Suíça", "CN": "China", "AUT": "Áustria", "SWE": "Suécia", "CHN": "China",
    "POL": "Polônia", "ISR": "Israel", "RUS": "Rússia", "NOR": "Noruega", "Desconhecido": "Desconhecido",
}

# Rótulos legíveis para eixos e seletores do dashboard.
ROTULOS = {
    "hotel": "Hotel",
    "is_canceled": "Cancelada (0/1)",
    "lead_time": "Antecedência (dias)",
    "arrival_date_year": "Ano de chegada",
    "arrival_date_month": "Mês de chegada",
    "arrival_date_week_number": "Semana do ano",
    "arrival_date_day_of_month": "Dia do mês",
    "stays_in_weekend_nights": "Noites de fim de semana",
    "stays_in_week_nights": "Noites de semana",
    "adults": "Adultos",
    "children": "Crianças",
    "babies": "Bebês",
    "meal": "Regime de refeição",
    "country": "País de origem",
    "market_segment": "Segmento de mercado",
    "distribution_channel": "Canal de distribuição",
    "is_repeated_guest": "Hóspede recorrente (0/1)",
    "previous_cancellations": "Cancelamentos anteriores",
    "previous_bookings_not_canceled": "Reservas anteriores não canceladas",
    "reserved_room_type": "Tipo de quarto reservado",
    "assigned_room_type": "Tipo de quarto atribuído",
    "booking_changes": "Alterações na reserva",
    "deposit_type": "Tipo de depósito",
    "agent": "ID do agente",
    "company": "ID da empresa",
    "days_in_waiting_list": "Dias em lista de espera",
    "customer_type": "Tipo de cliente",
    "adr": "Diária média - ADR (€)",
    "required_car_parking_spaces": "Vagas de estacionamento",
    "total_of_special_requests": "Pedidos especiais",
    "reservation_status": "Status final",
    "reservation_status_date": "Data do status",
    # Colunas derivadas
    "status": "Situação da reserva",
    "mes_num": "Mês (número)",
    "mes_nome": "Mês de chegada",
    "data_chegada": "Data de chegada",
    "noites_total": "Total de noites",
    "hospedes_total": "Total de hóspedes",
    "faixa_antecedencia": "Faixa de antecedência",
    "pedidos_especiais_cat": "Pedidos especiais",
    "tem_agente": "Reserva via agente (0/1)",
    "tem_empresa": "Reserva via empresa (0/1)",
    "quarto_diferente": "Quarto atribuído ≠ reservado (0/1)",
    "receita_estimada": "Receita estimada (€)",
    "pais_grupo": "País de origem (top 10)",
    "duplicada": "Linha repetida (0/1)",
}

FAIXAS_ANTECEDENCIA = [-1, 7, 30, 90, 180, 365, np.inf]
ROTULOS_ANTECEDENCIA = ["0–7 dias", "8–30 dias", "31–90 dias", "91–180 dias", "181–365 dias", "> 1 ano"]
ORDEM_PEDIDOS = ["0", "1", "2", "3+"]

COLUNAS_NUMERICAS = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
    "previous_cancellations", "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests",
    "noites_total", "hospedes_total",
]
COLUNAS_CATEGORICAS = [
    "hotel", "meal", "pais_grupo", "market_segment", "distribution_channel", "reserved_room_type",
    "assigned_room_type", "deposit_type", "customer_type", "mes_nome", "faixa_antecedencia",
    "pedidos_especiais_cat", "status", "reservation_status",
]


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------
def carregar_bruto(caminho: Path | str = CAMINHO_CSV) -> pd.DataFrame:
    """Lê o CSV original. Os 'NULL' de agent/company viram NaN automaticamente."""
    return pd.read_csv(caminho, na_values=["NULL"], keep_default_na=True)


# --------------------------------------------------------------------------
# Diagnóstico de faltantes (antes do tratamento)
# --------------------------------------------------------------------------
def diagnostico_faltantes(bruto: pd.DataFrame) -> pd.DataFrame:
    """Conta faltantes explícitos (NaN) e implícitos ('Undefined') por coluna."""
    linhas = []
    for col in bruto.columns:
        n_nan = int(bruto[col].isna().sum())
        n_undef = int((bruto[col] == "Undefined").sum()) if bruto[col].dtype != "float64" else 0
        if n_nan or n_undef:
            linhas.append(
                {
                    "coluna": col,
                    "faltantes_nan": n_nan,
                    "faltantes_undefined": n_undef,
                    "total": n_nan + n_undef,
                    "pct": 100 * (n_nan + n_undef) / len(bruto),
                }
            )
    return pd.DataFrame(linhas).sort_values("pct", ascending=False).reset_index(drop=True)


def evidencias_mecanismo(bruto: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tabelas que sustentam a discussão do mecanismo de ausência.

    A ideia é comparar o grupo com o valor ausente contra o grupo com o valor
    presente. Se forem parecidos -> compatível com MCAR. Se diferirem em
    variáveis observadas -> MAR. Se a ausência for "não se aplica" -> estrutural.
    """
    taxa = bruto["is_canceled"].mean()
    out: dict[str, pd.DataFrame] = {}
    for col in ["country", "agent", "company", "children"]:
        m = bruto[col].isna()
        out[col] = pd.DataFrame(
            {
                "grupo": ["ausente", "presente"],
                "linhas": [int(m.sum()), int((~m).sum())],
                "taxa_cancelamento": [bruto.loc[m, "is_canceled"].mean(), bruto.loc[~m, "is_canceled"].mean()],
                "pct_resort": [
                    (bruto.loc[m, "hotel"] == "Resort Hotel").mean(),
                    (bruto.loc[~m, "hotel"] == "Resort Hotel").mean(),
                ],
                "segmento_mais_comum": [
                    bruto.loc[m, "market_segment"].mode().iat[0] if m.any() else "-",
                    bruto.loc[~m, "market_segment"].mode().iat[0],
                ],
            }
        )
    out["_taxa_geral"] = pd.DataFrame({"taxa_cancelamento_geral": [taxa]})
    # Segmentos das reservas sem agente / com empresa (ausência estrutural)
    out["agent_por_segmento"] = (
        bruto.assign(sem_agente=bruto["agent"].isna())
        .groupby("market_segment")["sem_agente"].agg(["mean", "size"])
        .rename(columns={"mean": "pct_sem_agente", "size": "linhas"})
        .sort_values("pct_sem_agente", ascending=False)
    )
    out["company_por_segmento"] = (
        bruto.assign(com_empresa=bruto["company"].notna())
        .groupby("market_segment")["com_empresa"].agg(["mean", "size"])
        .rename(columns={"mean": "pct_com_empresa", "size": "linhas"})
        .sort_values("pct_com_empresa", ascending=False)
    )
    return out


# --------------------------------------------------------------------------
# Limpeza
# --------------------------------------------------------------------------
def limpar(bruto: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica todos os tratamentos e devolve (dados_limpos, log_de_tratamentos).

    Nenhum dado FALTANTE é removido: todos são imputados ou recodificados com
    justificativa. As únicas linhas removidas são registros impossíveis
    (reserva sem nenhum hóspede), que são inconsistências e não ausências.
    """
    df = bruto.copy()
    log: list[dict] = []

    def registrar(etapa, colunas, linhas, diagnostico, tratamento):
        log.append(
            {"etapa": etapa, "colunas": colunas, "linhas_afetadas": int(linhas),
             "diagnostico": diagnostico, "tratamento": tratamento}
        )

    # Marca as linhas repetidas ANTES de qualquer alteração (análise de sensibilidade).
    df["duplicada"] = df.duplicated(keep="first").astype(int)
    registrar(
        "Linhas repetidas", "todas", df["duplicada"].sum(),
        "26,8% das linhas são idênticas a outra. O dataset não tem ID de reserva; linhas "
        "iguais são, em sua maioria, quartos de uma mesma reserva de grupo (mesma data, "
        "agente e condições), concentradas em 'Não reembolsável' e 'Grupos'.",
        "Mantidas (cada linha é um quarto reservado, o que importa para a ocupação). "
        "Marcadas na coluna 'duplicada' para análise de sensibilidade no dashboard.",
    )

    # 1) children: 4 NaN
    n = df["children"].isna().sum()
    df["children"] = df["children"].fillna(0)
    registrar(
        "Faltante", "children", n,
        "4 reservas do hotel urbano em ago/2015, todas canceladas. Volume ínfimo (0,003%), "
        "sem padrão relacionado ao valor ausente -> compatível com MCAR (falha de registro).",
        "Imputado com a moda (0 crianças, 93% das reservas).",
    )

    # 2) country: 488 NaN
    n = df["country"].isna().sum()
    df["country"] = df["country"].fillna("Desconhecido")
    registrar(
        "Faltante", "country", n,
        "95% no hotel resort, sobretudo segmentos Corporativo e Direto, e taxa de "
        "cancelamento de 13,7% (vs. 37% geral). A ausência depende de variáveis observadas "
        "-> MAR. Excluir essas linhas enviesaria a amostra (removeria reservas que não cancelam).",
        "Mantido como categoria própria 'Desconhecido'.",
    )

    # 3) agent: NaN estrutural
    n = df["agent"].isna().sum()
    df["tem_agente"] = df["agent"].notna().astype(int)
    df["agent"] = df["agent"].fillna(0).astype(int)
    registrar(
        "Faltante estrutural", "agent", n,
        "13,7% sem agente: reservas Diretas, Corporativas e de Grupos, que por definição não "
        "passam por agência ('não se aplica'). A ausência é informativa: 24,7% de "
        "cancelamento sem agente vs. 39,0% com agente.",
        "Criada a variável 'tem_agente' (0/1); ID ausente recodificado como 0 = 'sem agente'.",
    )

    # 4) company: NaN estrutural (94%)
    n = df["company"].isna().sum()
    df["tem_empresa"] = df["company"].notna().astype(int)
    df["company"] = df["company"].fillna(0).astype(int)
    registrar(
        "Faltante estrutural", "company", n,
        "94,3% sem empresa: só reservas pagas por empresas (Corporativo, Grupos) têm ID. "
        "Ausência = 'não se aplica'. Também informativa: 17,5% de cancelamento com empresa "
        "vs. 38,2% sem.",
        "Criada 'tem_empresa' (0/1). O ID (mais de 300 valores, 94% vazio) não é usado nas análises.",
    )

    # 5) meal Undefined == SC
    n = (df["meal"] == "Undefined").sum()
    registrar(
        "Faltante implícito", "meal", n,
        "'Undefined' e 'SC' significam a mesma coisa segundo a documentação do dataset "
        "(nenhum pacote de refeição).",
        "'Undefined' unificado com 'SC' (Sem refeição).",
    )

    # 6) market_segment / distribution_channel Undefined
    m_seg = df["market_segment"] == "Undefined"
    m_can = df["distribution_channel"] == "Undefined"
    moda_seg_urbano = df.loc[(df["hotel"] == "City Hotel") & ~m_seg, "market_segment"].mode().iat[0]
    df.loc[m_seg, "market_segment"] = moda_seg_urbano
    mapa_canal = {"Direct": "Direct", "Online TA": "TA/TO", "Offline TA/TO": "TA/TO", "Groups": "TA/TO",
                  "Corporate": "Corporate", "Complementary": "Direct", "Aviation": "Corporate"}
    df.loc[m_can, "distribution_channel"] = df.loc[m_can, "market_segment"].map(mapa_canal)
    registrar(
        "Faltante implícito", "market_segment, distribution_channel", int(m_seg.sum() + m_can.sum()),
        "2 segmentos e 5 canais 'Undefined' (0,004%), todos no hotel urbano -> MCAR.",
        "Segmento imputado pela moda do hotel urbano (Agência online); canal deduzido do "
        "segmento (ex.: Agência online -> Agência/operadora).",
    )

    # 7) Inconsistência: reservas sem hóspedes
    sem_hospede = (df["adults"] + df["children"] + df["babies"]) == 0
    n = sem_hospede.sum()
    df = df.loc[~sem_hospede].copy()
    registrar(
        "Inconsistência", "adults, children, babies", n,
        "Reserva com 0 adultos, 0 crianças e 0 bebês é impossível (0,15% das linhas). "
        "Não é dado faltante: é registro inválido.",
        "Linhas removidas.",
    )

    # 8) Inconsistência: ADR negativo ou absurdo
    m_adr = (df["adr"] < 0) | (df["adr"] > 1000)
    n = m_adr.sum()
    df.loc[m_adr, "adr"] = np.nan
    df["mes_num"] = df["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MESES_EN)})
    mediana = df.groupby(["hotel", "mes_num", "reserved_room_type"])["adr"].transform("median")
    df["adr"] = df["adr"].fillna(mediana)
    registrar(
        "Outlier / erro", "adr", n,
        "Uma diária de -6,38 € e uma de 5.400 € (a segunda maior é 510 €). São erros de "
        "digitação, não diárias reais.",
        "Valor substituído pela mediana do mesmo hotel, mês e tipo de quarto. A reserva é mantida.",
    )

    # 9) Inconsistência: contagens implausíveis de crianças/bebês
    m_cri = (df["babies"] >= 9) | (df["children"] >= 10)
    n = m_cri.sum()
    df.loc[df["babies"] >= 9, "babies"] = 0
    df.loc[df["children"] >= 10, "children"] = 0
    registrar(
        "Outlier / erro", "children, babies", n,
        "3 reservas com 9 ou 10 bebês/crianças acompanhados de 1 ou 2 adultos; o próximo "
        "maior valor é 3. Provável erro de digitação.",
        "Substituído pela moda (0).",
    )

    registrar(
        "Outlier mantido", "lead_time, adr, adults, days_in_waiting_list, stays", 0,
        "Antecedência de até 737 dias, diárias de até 510 €, grupos de até 55 adultos e "
        "estadias longas são raros, mas plausíveis (operadoras bloqueiam quartos com "
        "1-2 anos; alta temporada e suítes elevam a diária).",
        "Mantidos. Nos gráficos usamos faixas (antecedência) e medianas/quartis, que são "
        "robustos a valores extremos.",
    )

    # --- Tipos ---------------------------------------------------------------
    df["children"] = df["children"].astype(int)

    # --- Traduções -------------------------------------------------------------
    df["reservation_status_en"] = df["reservation_status"]
    for col, mapa in TRADUCOES.items():
        df[col] = df[col].map(mapa).fillna(df[col])

    # --- Engenharia de atributos ---------------------------------------------------
    df["mes_nome"] = pd.Categorical(
        df["mes_num"].map(lambda m: MESES_PT[m - 1]), categories=MESES_PT, ordered=True
    )
    df["data_chegada"] = pd.to_datetime(
        dict(year=df["arrival_date_year"], month=df["mes_num"], day=df["arrival_date_day_of_month"])
    )
    df["ano_mes"] = df["data_chegada"].dt.to_period("M").dt.to_timestamp()
    df["noites_total"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["hospedes_total"] = df["adults"] + df["children"] + df["babies"]
    df["faixa_antecedencia"] = pd.cut(
        df["lead_time"], FAIXAS_ANTECEDENCIA, labels=ROTULOS_ANTECEDENCIA
    )
    df["pedidos_especiais_cat"] = pd.Categorical(
        df["total_of_special_requests"].clip(upper=3).astype(int).astype(str).replace({"3": "3+"}),
        categories=ORDEM_PEDIDOS, ordered=True,
    )
    df["status"] = np.where(df["is_canceled"] == 1, "Cancelada", "Não cancelada")
    df["quarto_diferente"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    df["receita_estimada"] = df["adr"] * df["noites_total"]
    top_paises = df["country"].value_counts().head(10).index
    df["pais_grupo"] = np.where(df["country"].isin(top_paises), df["country"], "Outros")
    df["pais_grupo"] = df["pais_grupo"].map(lambda c: PAISES_PT.get(c, c))

    df = df.reset_index(drop=True)
    return df, pd.DataFrame(log)


def carregar_limpo(caminho: Path | str = CAMINHO_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    return limpar(carregar_bruto(caminho))


def tabela_estrutural(df: pd.DataFrame) -> pd.DataFrame:
    """Pilar estrutural: tipo de cada variável original."""
    tipos = {
        "hotel": "Qualitativa nominal", "is_canceled": "Qualitativa binária (ALVO)",
        "lead_time": "Quantitativa discreta", "arrival_date_year": "Quantitativa discreta (tempo)",
        "arrival_date_month": "Qualitativa ordinal (tempo)", "arrival_date_week_number": "Quantitativa discreta (tempo)",
        "arrival_date_day_of_month": "Quantitativa discreta (tempo)",
        "stays_in_weekend_nights": "Quantitativa discreta", "stays_in_week_nights": "Quantitativa discreta",
        "adults": "Quantitativa discreta", "children": "Quantitativa discreta", "babies": "Quantitativa discreta",
        "meal": "Qualitativa nominal", "country": "Qualitativa nominal (177 países)",
        "market_segment": "Qualitativa nominal", "distribution_channel": "Qualitativa nominal",
        "is_repeated_guest": "Qualitativa binária", "previous_cancellations": "Quantitativa discreta",
        "previous_bookings_not_canceled": "Quantitativa discreta", "reserved_room_type": "Qualitativa nominal (código)",
        "assigned_room_type": "Qualitativa nominal (código)", "booking_changes": "Quantitativa discreta",
        "deposit_type": "Qualitativa nominal", "agent": "Qualitativa nominal (ID)", "company": "Qualitativa nominal (ID)",
        "days_in_waiting_list": "Quantitativa discreta", "customer_type": "Qualitativa nominal",
        "adr": "Quantitativa contínua (€)", "required_car_parking_spaces": "Quantitativa discreta",
        "total_of_special_requests": "Quantitativa discreta", "reservation_status": "Qualitativa nominal (vazamento)",
        "reservation_status_date": "Data (vazamento)",
    }
    linhas = []
    for col, tipo in tipos.items():
        s = df[col]
        linhas.append(
            {"coluna": col, "descricao": ROTULOS.get(col, col), "tipo_estatistico": tipo,
             "dtype_pandas": str(s.dtype), "valores_unicos": int(s.nunique()),
             "faltantes": int(s.isna().sum())}
        )
    return pd.DataFrame(linhas)
