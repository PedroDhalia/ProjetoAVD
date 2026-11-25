"""Cálculos e figuras plotly da Parte Comum para o dashboard (página K-Means).

Reaproveita exatamente as mesmas funções dos scripts (kmeans.py e metricas.py),
então os números batem com o relatório e com os notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import adjusted_rand_score

from src import estilo
from src.parte_comum import dados as D
from src.parte_comum import kmeans as K
from src.parte_comum import metricas as M

CORES_CLASSE = dict(zip(D.CLASSES_PT, estilo.CORES_CLASSE_OBESIDADE))
KS_ANIMACAO = list(range(2, 11))


def calcular_tudo() -> dict:
    prep = D.preparar("obesity")
    Xs, y = prep["Xs"], prep["y"]
    curva = K.curva_cotovelo(Xs)
    k_livre = K.escolher_k(curva)
    coords, pca = K.projecao_pca(Xs)

    animacao, por_k, modelos = {}, [], {}
    for k in KS_ANIMACAO:
        km = K.ajustar(Xs, k)
        modelos[k] = km
        centros = pca.transform(km.cluster_centers_)
        ordem = np.argsort(centros[:, 0])  # renumera pela posição no eixo PC1 -> cores estáveis na animação
        novo = np.empty_like(ordem)
        novo[ordem] = np.arange(k)
        animacao[k] = {"rotulos": novo[km.labels_], "centros": centros[ordem]}
        av = M.avaliar_passada(km.labels_, y)
        por_k.append({"k": k, "acuracia": av["acuracia"], "ari": av["ari"],
                      "classes_previstas": 7 - len(av["classes_nunca_previstas"])})

    passadas = {}
    for nome, k in [("livre", k_livre), ("fixo", len(np.unique(y)))]:
        km = modelos[k]
        av = M.avaliar_passada(km.labels_, y)
        passadas[nome] = av | {"rotulos": km.labels_, "centros_pca": pca.transform(km.cluster_centers_),
                               "granularidade": M.granularidade(km.labels_, y)}
    return {
        "prep": prep, "curva": curva, "k_livre": k_livre, "coords": coords, "variancia_pca": pca.explained_variance_ratio_,
        "animacao": animacao, "por_k": pd.DataFrame(por_k), "passadas": passadas,
        "sensibilidade": K.sensibilidade_features(prep["df"], k=7),
        "estabilidade": K.estabilidade_cotovelo(Xs),
        "baseline": pd.Series(y).value_counts(normalize=True).iloc[0],
    }


def _layout(fig, titulo, subtitulo=None, altura=420, **kw):
    texto = f"<b>{titulo}</b>" + (f"<br><span style='font-size:12px'>{subtitulo}</span>" if subtitulo else "")
    base = dict(title=dict(text=texto, x=0, xanchor="left"), height=altura,
                margin=dict(l=10, r=10, t=105 if subtitulo else 70, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    base["separators"] = ",."  # vírgula decimal, ponto de milhar (pt-BR)
    base.update(kw)
    fig.update_layout(**base)
    return fig


def fig_cotovelo(curva: pd.DataFrame, k: int) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[curva["k"].iat[0], curva["k"].iat[-1]], y=[curva["inercia"].iat[0], curva["inercia"].iat[-1]],
                    mode="lines", line=dict(dash="dot", color=estilo.TEXTO_MUDO), name="Reta 1º–último ponto",
                    hoverinfo="skip")
    fig.add_scatter(x=curva["k"], y=curva["inercia"], mode="lines+markers", line=dict(color=estilo.AZUL, width=3),
                    marker=dict(size=9), name="Inércia", customdata=np.c_[curva["queda_pct"].fillna(0), curva["silhueta"].fillna(0)],
                    hovertemplate="k = %{x}<br>inércia = %{y:,.0f}<br>queda vs k−1: %{customdata[0]:.1f}%"
                                  "<br>silhueta: %{customdata[1]:.3f}<extra></extra>")
    linha = curva[curva["k"] == k].iloc[0]
    fig.add_scatter(x=[k], y=[linha["inercia"]], mode="markers", name=f"Cotovelo: k = {k}",
                    marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color=estilo.LARANJA, width=3)))
    fig.add_vline(x=7, line=dict(dash="dash", color=estilo.TEXTO_MUDO), annotation_text="7 classes reais",
                  annotation_position="top right")
    fig.update_xaxes(title="Número de grupos (k)", dtick=1)
    fig.update_yaxes(title="Inércia (soma das distâncias² aos centróides)")
    return _layout(fig, "Método do Cotovelo — Passada 1",
                   f"Curva suave; o ponto mais distante da reta (critério kneedle) é k = {k}", altura=430)


def fig_animacao(dados_calc: dict) -> go.Figure:
    coords, y, anim = dados_calc["coords"], dados_calc["prep"]["y"], dados_calc["animacao"]
    v = dados_calc["variancia_pca"]
    texto_hover = [f"Classe real: {c}" for c in y]
    kmax = max(KS_ANIMACAO)

    def tracos(k):
        rot, centros = anim[k]["rotulos"], anim[k]["centros"]
        lista = []
        for g in range(kmax):
            m = rot == g
            lista.append(go.Scattergl(
                x=coords[m, 0], y=coords[m, 1], mode="markers", name=f"Grupo {g + 1}",
                marker=dict(size=5, color=estilo.CATEGORICA_ESTENDIDA[g], opacity=0.75),
                text=np.array(texto_hover)[m], hovertemplate="Grupo " + str(g + 1) + "<br>%{text}<extra></extra>",
                showlegend=g < k,
            ))
        lista.append(go.Scatter(x=centros[:, 0], y=centros[:, 1], mode="markers", name="Centróides",
                                marker=dict(symbol="x", size=14, color=estilo.TINTA, line=dict(color="white", width=2)),
                                hoverinfo="skip"))
        return lista

    k0 = KS_ANIMACAO[0]
    fig = go.Figure(data=tracos(k0), frames=[go.Frame(data=tracos(k), name=str(k)) for k in KS_ANIMACAO])
    passos = [dict(method="animate", label=str(k), args=[[str(k)], dict(mode="immediate", frame=dict(duration=600, redraw=True),
                                                                          transition=dict(duration=400))])
              for k in KS_ANIMACAO]
    fig.update_layout(
        sliders=[dict(active=0, currentvalue=dict(prefix="k = ", font=dict(size=16)), pad=dict(t=40), steps=passos)],
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.08, xanchor="left", yanchor="top", showactive=False,
                          buttons=[dict(label="▶ Animar", method="animate",
                                        args=[None, dict(frame=dict(duration=1100, redraw=True), transition=dict(duration=500),
                                                         fromcurrent=True)]),
                                   dict(label="❚❚ Pausar", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
    )
    fig.update_xaxes(title=f"Componente principal 1 ({v[0]:.0%} da variância)")
    fig.update_yaxes(title=f"Componente principal 2 ({v[1]:.0%})")
    return _layout(fig, "Como os grupos se reorganizam de k = 2 a k = 10",
                   "Projeção PCA das 15 variáveis padronizadas. Grupos numerados da esquerda para a direita (PC1) "
                   "para manter as cores estáveis entre os quadros.", altura=640,
                   legend=dict(orientation="v", x=1.01, y=1, xanchor="left"))


def fig_acuracia_por_k(por_k: pd.DataFrame, k_livre: int, baseline: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=por_k["k"], y=por_k["acuracia"], mode="lines+markers", name="Acurácia (cluster → classe majoritária)",
                    line=dict(color=estilo.AZUL, width=3), marker=dict(size=8),
                    customdata=por_k["classes_previstas"],
                    hovertemplate="k = %{x}<br>acurácia %{y:.1%}<br>%{customdata} de 7 classes previstas<extra></extra>")
    fig.add_scatter(x=por_k["k"], y=por_k["ari"], mode="lines+markers", name="ARI (0 = acaso, 1 = idêntico)",
                    line=dict(color=estilo.TEXTO_MUDO, width=2, dash="dot"), marker=dict(size=7),
                    hovertemplate="k = %{x}<br>ARI %{y:.3f}<extra></extra>")
    for k, rot in [(k_livre, "Passada 1"), (7, "Passada 2")]:
        v = por_k.loc[por_k["k"] == k, "acuracia"].iat[0]
        fig.add_scatter(x=[k], y=[v], mode="markers+text", marker=dict(size=16, color=estilo.LARANJA), text=[f"{rot}: {v:.1%}"],
                        textposition="top center", showlegend=False, hoverinfo="skip")
    fig.add_hline(y=baseline, line=dict(dash="dash", color=estilo.TEXTO_MUDO),
                  annotation_text=f"chutar a classe mais comum: {baseline:.1%}", annotation_position="bottom right")
    fig.update_xaxes(title="Número de grupos (k)", dtick=1)
    fig.update_yaxes(title="Valor (0 a 1)", tickformat=".0%", range=[0, 0.7])
    return _layout(fig, "Concordância grupos × classes reais para cada k",
                   "A acurácia sobe com k quase 'de graça' (mais grupos = mais vagas para classes); o ARI corrige isso",
                   altura=420)


def fig_pca_passada(coords, rotulos, centros, titulo, variancia) -> go.Figure:
    fig = go.Figure()
    for g in range(len(np.unique(rotulos))):
        m = rotulos == g
        fig.add_scattergl(x=coords[m, 0], y=coords[m, 1], mode="markers", name=f"Grupo {g}",
                          marker=dict(size=5, color=estilo.CATEGORICA_ESTENDIDA[g], opacity=0.7))
    fig.add_scatter(x=centros[:, 0], y=centros[:, 1], mode="markers", name="Centróides", hoverinfo="skip",
                    marker=dict(symbol="x", size=14, color=estilo.TINTA, line=dict(color="white", width=2)))
    fig.update_xaxes(title=f"PC1 ({variancia[0]:.0%})")
    fig.update_yaxes(title=f"PC2 ({variancia[1]:.0%})")
    return _layout(fig, titulo, altura=430)


def fig_pca_classes(coords, y, variancia) -> go.Figure:
    fig = go.Figure()
    for c in D.CLASSES_PT:
        m = y == c
        fig.add_scattergl(x=coords[m, 0], y=coords[m, 1], mode="markers", name=c,
                          marker=dict(size=5, color=CORES_CLASSE[c], opacity=0.75))
    fig.update_xaxes(title=f"PC1 ({variancia[0]:.0%})")
    fig.update_yaxes(title=f"PC2 ({variancia[1]:.0%})")
    return _layout(fig, "Classes reais na mesma projeção (o K-Means não viu estes rótulos)", altura=430)


def fig_contingencia(ct: pd.DataFrame, mapa: dict, titulo: str) -> go.Figure:
    escala = [[i / (len(estilo.SEQ_AZUL) - 1), c] for i, c in enumerate(estilo.SEQ_AZUL)]
    x = [f"G{c} → {mapa[int(c)]}" for c in ct.columns]
    fig = go.Figure(go.Heatmap(z=ct.to_numpy(), x=x, y=ct.index, colorscale=escala, text=ct.to_numpy(),
                               texttemplate="%{text}", colorbar=dict(title="Pessoas"),
                               hovertemplate="%{y} no %{x}: %{z} pessoas<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Classe real")
    fig.update_xaxes(title="Grupo → classe majoritária", tickangle=-30)
    return _layout(fig, titulo, "Tabela retangular: 7 classes × k grupos", altura=430)


def fig_matriz(cm: pd.DataFrame, titulo: str) -> go.Figure:
    norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
    escala = [[i / (len(estilo.SEQ_AZUL) - 1), c] for i, c in enumerate(estilo.SEQ_AZUL)]
    texto = [[f"{v}<br>{p:.0%}" if v else "0" for v, p in zip(lv, lp)] for lv, lp in zip(cm.to_numpy(), norm.to_numpy())]
    fig = go.Figure(go.Heatmap(z=norm.to_numpy(), x=list(cm.columns), y=list(cm.index), colorscale=escala, zmin=0, zmax=1,
                               text=texto, texttemplate="%{text}", colorbar=dict(title="% linha", tickformat=".0%"),
                               customdata=cm.to_numpy(),
                               hovertemplate="Real: %{y}<br>Previsto: %{x}<br>%{customdata} pessoas (%{z:.0%} da linha)<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Classe real")
    fig.update_xaxes(title="Classe prevista (majoritária do grupo)", tickangle=-30)
    return _layout(fig, titulo, "Diagonal = acertos (recall de cada classe). Colunas vazias = classe nunca prevista", altura=470)


def tabela_metricas(av: dict) -> pd.DataFrame:
    t = av["por_classe"].copy()
    return t[["classe", "suporte_real", "previstos", "acertos", "precisao", "recall", "f1"]]


def ari(y, rotulos) -> float:
    return adjusted_rand_score(y, rotulos)
