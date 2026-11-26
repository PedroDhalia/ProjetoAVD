"""EDA da Parte Específica (Hotel Booking Demand) — figuras estáticas e tabelas.

Uso:
    python -m src.parte_especifica.eda --out ./out/parte_especifica/

Gera em <out>:
    tabelas/*.csv  -> estrutura, faltantes, tratamentos, outliers, sensibilidade, modelo
    figuras/*.png  -> as mesmas visualizações do dashboard, em versão estática para o relatório
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src import estilo
from src.parte_especifica import analise, dados, modelo

PCT = mtick.PercentFormatter(1.0, decimals=0)


def _salvar(fig, pasta: Path, nome: str) -> Path:
    caminho = pasta / f"{nome}.png"
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def _rodape(fig, texto="Fonte: Hotel Booking Demand (Antonio, Almeida & Nunes, 2019). Chegadas de jul/2015 a ago/2017."):
    fig.text(0.0, -0.02, texto, fontsize=8, color=estilo.TEXTO_MUDO, ha="left")


# ==========================================================================
# Qualidade dos dados
# ==========================================================================
def fig_faltantes(bruto: pd.DataFrame):
    t = dados.diagnostico_faltantes(bruto)
    mecanismo = {
        "company": "Estrutural (não se aplica)", "agent": "Estrutural (não se aplica)",
        "meal": "Rótulo equivalente", "country": "MAR", "distribution_channel": "MCAR",
        "children": "MCAR", "market_segment": "MCAR",
    }
    cores = {"Estrutural (não se aplica)": estilo.AZUL, "MAR": estilo.LARANJA,
             "MCAR": estilo.CINZA_CONTEXTO, "Rótulo equivalente": "#9ec5f4"}
    t["mecanismo"] = t["coluna"].map(mecanismo)
    t = t.sort_values("pct")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    barras = ax.barh(t["coluna"], t["pct"], color=t["mecanismo"].map(cores), height=0.6)
    for b, (_, r) in zip(barras, t.iterrows()):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                f"{r['pct']:.2f}% ({int(r['total']):,} linhas)".replace(",", "."), va="center", fontsize=8.5,
                color=estilo.TEXTO_SECUNDARIO)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% de linhas com valor ausente (NaN ou 'Undefined')")
    ax.set_ylabel("Variável")
    ax.set_title("Valores ausentes por variável e mecanismo provável de ausência")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cores.values()]
    ax.legend(handles, cores.keys(), title="Mecanismo", loc="lower right")
    ax.grid(axis="y", visible=False)
    _rodape(fig)
    return fig


def fig_outliers(df: pd.DataFrame):
    variaveis = [("lead_time", "Antecedência (dias)"), ("adr", "Diária média - ADR (€)"),
                 ("noites_total", "Total de noites"), ("days_in_waiting_list", "Dias em lista de espera")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))
    for ax, (col, rot) in zip(axes.flat, variaveis):
        s = df[col]
        ax.boxplot(
            [df.loc[df["hotel"] == h, col] for h in ["Hotel urbano", "Hotel resort"]],
            orientation="horizontal", widths=0.55, patch_artist=True,
            boxprops=dict(facecolor="#eef5fd", edgecolor=estilo.AZUL),
            medianprops=dict(color=estilo.LARANJA, linewidth=2),
            whiskerprops=dict(color=estilo.AZUL), capprops=dict(color=estilo.AZUL),
            flierprops=dict(marker="o", markersize=2.5, markerfacecolor=estilo.TEXTO_MUDO,
                            markeredgecolor="none", alpha=0.35),
        )
        ax.set_yticks([1, 2], ["Urbano", "Resort"])
        q1, q3 = s.quantile([0.25, 0.75])
        n_out = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
        ax.set_title(f"{rot}", fontsize=10.5)
        ax.set_xlabel(f"{rot} — {n_out:,} outliers pela regra de Tukey ({100 * n_out / len(s):.1f}%)".replace(",", "."),
                      fontsize=8.5)
    fig.suptitle("Outliers das principais variáveis numéricas (boxplot por hotel)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _rodape(fig)
    return fig


# ==========================================================================
# Exploratórias
# ==========================================================================
def fig_e1_antecedencia(df: pd.DataFrame):
    h = analise.histograma_por_status(df, "lead_time", largura=10, maximo=500)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for status in ["Não cancelada", "Cancelada"]:
        sub = h[h["status"] == status]
        ax.step(sub["inicio"], sub["pct"], where="post", color=estilo.COR_STATUS[status], linewidth=2, label=status)
        ax.fill_between(sub["inicio"], sub["pct"], step="post", color=estilo.COR_STATUS[status], alpha=0.08)
    for status, ls in [("Não cancelada", "--"), ("Cancelada", "--")]:
        med = df.loc[df["status"] == status, "lead_time"].median()
        ax.axvline(med, color=estilo.COR_STATUS[status], linestyle=ls, linewidth=1)
        ax.text(med + 4, ax.get_ylim()[1] * (0.92 if status == "Cancelada" else 0.80),
                f"mediana {status.lower()}: {med:.0f} dias", color=estilo.TINTA, fontsize=8.5)
    ax.set_xlim(0, 500)
    ax.set_xlabel("Antecedência da reserva (dias entre a reserva e a chegada; valores > 500 agrupados no fim)")
    ax.set_ylabel("% das reservas do grupo")
    ax.set_title("E1. Distribuição da antecedência: reservas canceladas x não canceladas")
    ax.legend(title="Situação", loc="upper right")
    _rodape(fig)
    return fig


def fig_e2_sazonalidade(df: pd.DataFrame):
    s = analise.sazonalidade(df)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for hotel, cor in estilo.COR_HOTEL.items():
        sub = s[s["hotel"] == hotel].sort_values("mes_nome")
        x = np.arange(len(sub))
        ax1.plot(x, sub["reservas_por_ano"], marker="o", color=cor, linewidth=2, markersize=5, label=hotel)
        ax2.plot(x, sub["taxa"], marker="o", color=cor, linewidth=2, markersize=5, label=hotel)
    ax1.set_ylabel("Reservas por mês\n(média dos anos)")
    ax1.set_title("E2. Sazonalidade: volume de reservas e taxa de cancelamento por mês de chegada")
    ax1.legend(title="Hotel", loc="upper left")
    ax2.set_ylabel("Taxa de cancelamento")
    ax2.yaxis.set_major_formatter(PCT)
    ax2.set_xticks(range(12), dados.MESES_PT)
    ax2.set_xlabel("Mês de chegada")
    ax2.set_ylim(0, 0.55)
    fig.tight_layout()
    _rodape(fig)
    return fig


def fig_e3_correlacao(df: pd.DataFrame):
    c = analise.correlacao(df)
    nomes = {
        "is_canceled": "Cancelada", "lead_time": "Antecedência", "adr": "Diária (ADR)", "noites_total": "Noites",
        "hospedes_total": "Hóspedes", "total_of_special_requests": "Pedidos especiais",
        "required_car_parking_spaces": "Vagas estacion.", "booking_changes": "Alterações",
        "previous_cancellations": "Cancel. anteriores", "previous_bookings_not_canceled": "Reservas ant. mantidas",
        "is_repeated_guest": "Recorrente", "days_in_waiting_list": "Lista de espera",
        "tem_agente": "Via agente", "tem_empresa": "Via empresa",
    }
    c = c.rename(index=nomes, columns=nomes)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    mascara = np.triu(np.ones_like(c, dtype=bool), k=1)
    dados_plot = np.ma.array(c.to_numpy(), mask=mascara)
    im = ax.imshow(dados_plot, cmap=estilo.cmap_divergente(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)), c.index)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(i + 1):
            v = c.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.5 else estilo.TINTA)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Correlação de Spearman (−1 a +1)")
    ax.set_title("E3. Correlação entre as variáveis numéricas (Spearman)")
    _rodape(fig)
    return fig


def fig_e4_adr(df: pd.DataFrame):
    st = analise.estatisticas_caixa(df[df["adr"] > 0], "adr", ["mes_nome", "hotel"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    desloc = {"Hotel urbano": -0.2, "Hotel resort": 0.2}
    for hotel, cor in estilo.COR_HOTEL.items():
        sub = st[st["hotel"] == hotel].sort_values("mes_nome")
        stats = [
            {"med": r.mediana, "q1": r.q1, "q3": r.q3, "whislo": r.bigode_inf, "whishi": r.bigode_sup, "fliers": []}
            for r in sub.itertuples()
        ]
        pos = np.arange(12) + desloc[hotel]
        ax.bxp(stats, positions=pos, widths=0.35, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=cor, alpha=0.25, edgecolor=cor), medianprops=dict(color=cor, linewidth=2),
               whiskerprops=dict(color=cor), capprops=dict(color=cor))
        ax.plot([], [], color=cor, linewidth=6, alpha=0.5, label=hotel)
    ax.set_xticks(range(12), dados.MESES_PT)
    ax.set_xlabel("Mês de chegada")
    ax.set_ylabel("Diária média - ADR (€)")
    ax.set_title("E4. Diária média (ADR) por mês e hotel — o resort tem sazonalidade de preço muito mais forte")
    ax.legend(title="Hotel", loc="upper left")
    _rodape(fig, "Boxplot: caixa = Q1–Q3, linha = mediana, bigodes = 1,5×IQR (outliers omitidos). ADR = 0 (cortesias) excluído.")
    return fig


def fig_e5_segmento_deposito(df: pd.DataFrame):
    taxa, n = analise.taxa_por_duas(df, "market_segment", "deposit_type")
    ordem = analise.taxa_por(df, "market_segment").sort_values("reservas", ascending=False)["market_segment"]
    taxa, n = taxa.loc[ordem], n.loc[ordem]
    colunas = ["Sem depósito", "Reembolsável", "Não reembolsável"]
    taxa, n = taxa[colunas], n[colunas]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(taxa.to_numpy(dtype=float), cmap=estilo.cmap_sequencial(), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(colunas)), colunas)
    ax.set_yticks(range(len(taxa)), taxa.index)
    ax.grid(False)
    for i in range(taxa.shape[0]):
        for j in range(taxa.shape[1]):
            v, cnt = taxa.iat[i, j], n.iat[i, j]
            if pd.isna(v):
                ax.text(j, i, "sem reservas", ha="center", va="center", fontsize=8, color=estilo.TEXTO_MUDO)
            else:
                ax.text(j, i, f"{v:.0%}\nn={int(cnt):,}".replace(",", "."), ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 else estilo.TINTA)
    cb = fig.colorbar(im, ax=ax, format=PCT)
    cb.set_label("Taxa de cancelamento")
    ax.set_xlabel("Tipo de depósito")
    ax.set_ylabel("Segmento de mercado (ordenado por volume)")
    ax.set_title("E5. Taxa de cancelamento por segmento e tipo de depósito")
    _rodape(fig, "Atenção: 'Não reembolsável' cancela 99% — são lotes de quartos de grupos/operadoras (98% linhas repetidas).")
    return fig


def fig_e6_compromisso(df: pd.DataFrame):
    """Taxa de cancelamento por sinais de compromisso do hóspede."""
    paineis = [
        ("pedidos_especiais_cat", "Pedidos especiais", dados.ORDEM_PEDIDOS),
        ("is_repeated_guest", "Hóspede recorrente", [0, 1]),
        ("tem_agente", "Reservou via agente", [0, 1]),
    ]
    geral = df["is_canceled"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, (col, titulo, ordem) in zip(axes, paineis):
        t = analise.taxa_por(df, col, ordem=ordem)
        rot = t[col].astype(str).replace({"0": "Não", "1": "Sim"}) if col != "pedidos_especiais_cat" else t[col].astype(str)
        ax.bar(rot, t["taxa"], color=estilo.AZUL, width=0.6)
        ax.axhline(geral, color=estilo.TEXTO_MUDO, linestyle="--", linewidth=1)
        for x, (v, nn) in enumerate(zip(t["taxa"], t["reservas"])):
            ax.text(x, v + 0.01, f"{v:.0%}\nn={nn:,}".replace(",", "."), ha="center", fontsize=8)
        ax.set_title(titulo, fontsize=10.5)
        ax.set_xlabel(titulo)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Taxa de cancelamento")
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylim(0, 0.62)
    axes[2].text(1.45, geral, f"média\n{geral:.0%}", va="center", fontsize=8, color=estilo.TEXTO_MUDO)
    fig.suptitle("E6. Taxa de cancelamento por sinais de relacionamento do hóspede", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _rodape(fig)
    return fig


# ==========================================================================
# Explanatórias (público: gestão do hotel)
# ==========================================================================
def fig_x1_antecedencia(df: pd.DataFrame):
    t = analise.taxa_por(df, "faixa_antecedencia", ordem=dados.ROTULOS_ANTECEDENCIA)
    geral = df["is_canceled"].mean()
    cores = [estilo.LARANJA if v > geral else estilo.CINZA_CONTEXTO for v in t["taxa"]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(t["faixa_antecedencia"].astype(str), t["taxa"], color=cores, width=0.65)
    ax.axhline(geral, color=estilo.TINTA, linestyle="--", linewidth=1)
    ax.text(-0.45, geral + 0.015, f"média geral: {geral:.0%}", fontsize=9, color=estilo.TINTA)
    for x, (v, nn) in enumerate(zip(t["taxa"], t["reservas"])):
        ax.text(x, v + 0.012, f"{v:.0%}", ha="center", fontsize=12, fontweight="bold",
                color=estilo.TINTA)
        ax.text(x, 0.012, f"{nn:,} reservas".replace(",", "."), ha="center", fontsize=7.5,
                color="white" if cores[x] == estilo.LARANJA else estilo.TEXTO_SECUNDARIO)
    ax.set_ylim(0, 0.78)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Taxa de cancelamento")
    ax.set_xlabel("Antecedência da reserva (dias entre reservar e chegar)")
    fig.suptitle("Quanto mais cedo o hóspede reserva, maior a chance de cancelar", x=0.125, y=1.0,
                 ha="left", fontsize=13, fontweight="bold", color=estilo.TINTA)
    ax.set_title("A partir de 31 dias a taxa já supera a média; acima de 6 meses, mais da metade cancela",
                 fontsize=9.5, fontweight="normal", color=estilo.TEXTO_SECUNDARIO)
    ax.grid(axis="x", visible=False)
    _rodape(fig)
    return fig


def fig_x2_pedidos(df: pd.DataFrame):
    t = analise.taxa_antecedencia_por_pedidos(df)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    estilos = {"Nenhum pedido especial": (estilo.LARANJA, 3), "Com ≥ 1 pedido especial": (estilo.TEXTO_MUDO, 2)}
    for grupo, (cor, lw) in estilos.items():
        sub = t[t["pedidos"] == grupo].sort_values("faixa_antecedencia")
        x = np.arange(len(sub))
        ax.plot(x, sub["taxa"], color=cor, linewidth=lw, marker="o", markersize=7, label=grupo)
        ax.text(x[-1] + 0.12, sub["taxa"].iloc[-1], grupo, color=estilo.TINTA, fontsize=9, va="center",
                fontweight="bold" if cor == estilo.LARANJA else "normal")
        for xi, v in zip(x, sub["taxa"]):
            ax.text(xi, v + (0.035 if cor == estilo.LARANJA else -0.06), f"{v:.0%}", ha="center", fontsize=8.5,
                    color=estilo.TINTA)
    ax.set_xticks(range(len(dados.ROTULOS_ANTECEDENCIA)), dados.ROTULOS_ANTECEDENCIA)
    ax.set_xlim(-0.3, 6.6)
    ax.set_ylim(0, 0.95)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Taxa de cancelamento")
    ax.set_xlabel("Antecedência da reserva")
    ax.legend(loc="upper left", title="Pedidos especiais na reserva")
    ax.set_title("Um simples pedido especial corta o cancelamento pela metade (ou mais) em toda antecedência")
    ax.grid(axis="x", visible=False)
    _rodape(fig, "Pedido especial = berço, andar alto, cama de casal etc. Sinal de que o hóspede já planejou a estadia.")
    return fig


def fig_x3_perfis(df: pd.DataFrame):
    p = analise.perfil_risco(df)
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    barras = ["Participação nas\nRESERVAS", "Participação nos\nCANCELAMENTOS"]
    cores = [estilo.LARANJA, "#d9d8d2", "#bdbcb4", "#a5a49c"]
    for yi, coluna in enumerate(["pct_reservas", "pct_cancelamentos"]):
        esquerda = 0.0
        for k, (perfil, v) in enumerate(zip(p["perfil"], p[coluna])):
            ax.barh(yi, v, left=esquerda, color=cores[k], height=0.55, edgecolor="white", linewidth=2)
            if v > 0.07:
                ax.text(esquerda + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=11 if k == 0 else 9,
                        fontweight="bold" if k == 0 else "normal", color="white" if k == 0 else estilo.TINTA)
            esquerda += v
    ax.set_yticks([0, 1], barras)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("% do total")
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cores]
    ax.legend(handles, [f"{a} (cancela {t:.0%})" for a, t in zip(p["perfil"], p["taxa"])],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
    ax.set_title("O perfil de maior risco é 27% das reservas, mas responde por 47% dos cancelamentos")
    _rodape(fig, "")
    return fig


# ==========================================================================
# Modelo
# ==========================================================================
def fig_modelo_importancia(res: modelo.ResultadoModelo):
    imp = res.importancia.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cores = [estilo.LARANJA if v in ("lead_time", "total_of_special_requests") else estilo.AZUL for v in imp["variavel"]]
    ax.barh(imp["rotulo"], imp["queda_auc"], xerr=imp["desvio"], color=cores, height=0.6,
            error_kw=dict(ecolor=estilo.TEXTO_MUDO, lw=1))
    ax.set_xlabel("Queda média da AUC ao embaralhar a variável (importância por permutação)")
    ax.set_ylabel("Variável")
    ax.set_title("Importância das variáveis no modelo de risco de cancelamento")
    ax.grid(axis="y", visible=False)
    _rodape(fig, "Laranja = variáveis da hipótese (antecedência e pedidos especiais). Teste: chegadas de 2017.")
    return fig


def fig_modelo_matriz(res: modelo.ResultadoModelo):
    cm = res.matriz_confusao
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.imshow(cm.to_numpy(), cmap=estilo.cmap_sequencial(estilo.SEQ_AZUL))
    ax.set_xticks([0, 1], cm.columns)
    ax.set_yticks([0, 1], cm.index)
    ax.grid(False)
    tot = cm.to_numpy().sum()
    for i in range(2):
        for j in range(2):
            v = cm.iat[i, j]
            ax.text(j, i, f"{v:,}\n({v / tot:.0%})".replace(",", "."), ha="center", va="center",
                    color="white" if v > tot * 0.3 else estilo.TINTA, fontsize=10)
    ax.set_title("Matriz de confusão do modelo (teste 2017, limiar 0,5)")
    ax.set_xlabel("Previsão")
    ax.set_ylabel("Realidade")
    return fig


# ==========================================================================
def executar(out: Path) -> dict:
    estilo.aplicar_estilo_mpl()
    pasta_fig = out / "figuras"
    pasta_tab = out / "tabelas"
    pasta_fig.mkdir(parents=True, exist_ok=True)
    pasta_tab.mkdir(parents=True, exist_ok=True)

    bruto = dados.carregar_bruto()
    df, log = dados.limpar(bruto)

    # ---- Tabelas --------------------------------------------------------------
    dados.tabela_estrutural(bruto).to_csv(pasta_tab / "estrutura.csv", index=False)
    dados.diagnostico_faltantes(bruto).to_csv(pasta_tab / "faltantes.csv", index=False)
    for nome, t in dados.evidencias_mecanismo(bruto).items():
        t.to_csv(pasta_tab / f"mecanismo_{nome.strip('_')}.csv")
    log.to_csv(pasta_tab / "tratamentos.csv", index=False)
    cols_out = ["lead_time", "adr", "noites_total", "adults", "children", "babies", "days_in_waiting_list",
                "previous_cancellations", "booking_changes", "total_of_special_requests"]
    analise.tabela_outliers(df, cols_out).to_csv(pasta_tab / "outliers_iqr.csv", index=False)
    analise.sensibilidade_duplicatas(df).to_csv(pasta_tab / "sensibilidade_duplicatas.csv", index=False)
    df[dados.COLUNAS_NUMERICAS].describe().T.to_csv(pasta_tab / "descritivas_numericas.csv")
    pd.concat(
        [df[c].value_counts(normalize=True).rename("proporcao").to_frame().assign(variavel=c)
         for c in ["hotel", "meal", "market_segment", "distribution_channel", "deposit_type", "customer_type",
                   "pais_grupo", "reserved_room_type"]]
    ).to_csv(pasta_tab / "frequencias_categoricas.csv")
    analise.perfil_risco(df).to_csv(pasta_tab / "perfil_risco.csv", index=False)
    analise.taxa_por(df, "faixa_antecedencia", ordem=dados.ROTULOS_ANTECEDENCIA).to_csv(
        pasta_tab / "taxa_por_antecedencia.csv", index=False)
    analise.taxa_antecedencia_por_pedidos(df).to_csv(pasta_tab / "taxa_antecedencia_pedidos.csv", index=False)

    # ---- Figuras ----------------------------------------------------------------
    figuras = {
        "q1_faltantes": fig_faltantes(bruto),
        "q2_outliers": fig_outliers(df),
        "e1_antecedencia": fig_e1_antecedencia(df),
        "e2_sazonalidade": fig_e2_sazonalidade(df),
        "e3_correlacao": fig_e3_correlacao(df),
        "e4_adr": fig_e4_adr(df),
        "e5_segmento_deposito": fig_e5_segmento_deposito(df),
        "e6_compromisso": fig_e6_compromisso(df),
        "x1_antecedencia": fig_x1_antecedencia(df),
        "x2_pedidos": fig_x2_pedidos(df),
        "x3_perfis": fig_x3_perfis(df),
    }
    for nome, fig in figuras.items():
        _salvar(fig, pasta_fig, nome)

    # ---- Modelo ------------------------------------------------------------------
    res = modelo.treinar(df, calcular_importancia=True)
    _salvar(fig_modelo_importancia(res), pasta_fig, "m1_importancia")
    _salvar(fig_modelo_matriz(res), pasta_fig, "m2_matriz_confusao")
    res.importancia.to_csv(pasta_tab / "modelo_importancia.csv", index=False)
    res.matriz_confusao.to_csv(pasta_tab / "modelo_matriz_confusao.csv")
    metricas = {k: float(v) for k, v in res.metricas.items()} | {"n_treino": res.n_treino, "n_teste": res.n_teste}
    (pasta_tab / "modelo_metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")

    resumo = analise.kpis(df) | {"linhas_brutas": len(bruto), "colunas_brutas": bruto.shape[1]}
    (pasta_tab / "resumo.json").write_text(json.dumps({k: float(v) for k, v in resumo.items()}, indent=2),
                                           encoding="utf-8")
    return {"figuras": sorted(p.name for p in pasta_fig.glob("*.png")), "metricas_modelo": metricas, "resumo": resumo}


def main() -> None:
    matplotlib.use("Agg")  # só na linha de comando: salva PNGs sem abrir janelas
    parser = argparse.ArgumentParser(description="EDA da Parte Específica (Hotel Booking Demand)")
    parser.add_argument("--out", default="./out/parte_especifica/", help="pasta de saída")
    args = parser.parse_args()
    r = executar(Path(args.out))
    print(f"{len(r['figuras'])} figuras geradas em {args.out}figuras")
    print("Métricas do modelo (teste 2017):", {k: round(v, 3) for k, v in r["metricas_modelo"].items()})


if __name__ == "__main__":
    main()
