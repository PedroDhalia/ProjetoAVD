"""Gera o relatório (relatorio.pdf) e a Declaração de Uso de IA (docs/DECLARACAO_IA.pdf).

Uso (na raiz do repositório, depois de rodar os scripts de análise):
    python scripts/gerar_documentos.py

O texto é montado em HTML (docs/relatorio.html e docs/DECLARACAO_IA.html) com as
figuras e tabelas de out/ e convertido em PDF pelo Microsoft Edge ou Google Chrome
em modo headless. Sem navegador disponível, abra os .html e use "Imprimir > Salvar
como PDF".
"""

from __future__ import annotations

import datetime as dt
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.parte_especifica import notas  # noqa: E402

DOCS = RAIZ / "docs"
OUT_C = RAIZ / "out" / "parte_comum"
OUT_E = RAIZ / "out" / "parte_especifica"
INTEGRANTES = ["Pedro de Melo Dhalia", "Matheos de Oliveira Guerra", "Miguel José França de Andrade"]
HOJE = dt.date.today().strftime("%d/%m/%Y")

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
        @bottom-right { content: counter(page); font: 9pt system-ui, sans-serif; color: #898781; } }
body { font-family: "Segoe UI", system-ui, sans-serif; font-size: 10.3pt; line-height: 1.45; color: #0b0b0b; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 15pt; margin: 18pt 0 6pt; border-bottom: 2px solid #2a78d6; padding-bottom: 2pt; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; } h4 { font-size: 10.5pt; margin: 8pt 0 2pt; }
h2, h3, h4 { break-after: avoid; page-break-after: avoid; }
.bloco { break-inside: avoid; page-break-inside: avoid; }
p { margin: 4pt 0 6pt; text-align: justify; }
.parte { page-break-before: always; font-size: 22pt; color: #2a78d6; border: none; margin-top: 0; }
.quebra { page-break-before: always; }
figure { margin: 8pt 0 10pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 118mm; }
figure.alta img { max-height: 170mm; }
figcaption { font-size: 8.8pt; color: #52514e; text-align: left; margin-top: 3pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 6pt 0 10pt; page-break-inside: avoid; }
th { background: #eeede8; text-align: left; } th, td { border: 1px solid #d9d8d2; padding: 3pt 5pt; vertical-align: top; }
.nota { border-left: 3px solid #eb6834; background: #fbf7f4; padding: 5pt 9pt; margin: 4pt 0 12pt; font-size: 9.4pt; page-break-inside: avoid; }
.nota p { margin: 2pt 0; }
.destaque { background: #eef5fd; border-left: 3px solid #2a78d6; padding: 6pt 10pt; margin: 6pt 0 10pt; page-break-inside: avoid; }
.capa { text-align: center; padding-top: 55mm; } .capa h1 { font-size: 26pt; } .capa p { text-align: center; }
.sub { color: #52514e; } ul, ol { margin: 3pt 0 6pt 16pt; padding: 0; } li { margin: 1pt 0; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f4f4f1; padding: 0 2pt; }
.assinatura { margin-top: 26pt; display: flex; gap: 18pt; flex-wrap: wrap; }
.assinatura div { flex: 1 1 45%; border-top: 1px solid #0b0b0b; padding-top: 3pt; margin-top: 26pt; font-size: 9.5pt; }
"""


# --------------------------------------------------------------------------
def img(caminho: Path, legenda: str, alta: bool = False) -> str:
    rel = Path("..") / caminho.relative_to(RAIZ)
    return (f'<figure class="{"alta" if alta else ""}"><img src="{rel.as_posix()}" alt="{html.escape(legenda)}">'
            f"<figcaption>{legenda}</figcaption></figure>")


def nota(chave: str) -> str:
    n = notas.NOTAS[chave]
    return (f'<div class="nota"><p><b>O que mostra.</b> {html.escape(n["o_que"])}</p>'
            f'<p><b>Insight.</b> {html.escape(n["insight"])}</p>'
            f'<p><b>Por que este gráfico.</b> {html.escape(n["por_que"])}</p></div>')


def tabela(df: pd.DataFrame, formatos: dict | None = None, renomear: dict | None = None) -> str:
    d = df.copy()
    for col, f in (formatos or {}).items():
        if col in d:
            d[col] = d[col].map(lambda v, f=f: "—" if pd.isna(v) else f(v))
    if renomear:
        d = d.rename(columns=renomear)
    return d.to_html(index=False, escape=True, border=0)


def pct(casas=1):
    return lambda v: f"{v:.{casas}%}".replace(".", ",")


def dec(casas=3):
    return lambda v: f"{v:.{casas}f}".replace(".", ",")


def inteiro(v):
    return f"{int(v):,}".replace(",", ".")


# --------------------------------------------------------------------------
def relatorio() -> str:
    res_c = pd.read_csv(OUT_C / "metricas" / "resumo_passadas.csv")
    met_l = pd.read_csv(OUT_C / "metricas" / "livre_metricas_por_classe.csv")
    met_f = pd.read_csv(OUT_C / "metricas" / "fixo_metricas_por_classe.csv")
    gran_l = pd.read_csv(OUT_C / "metricas" / "livre_granularidade.csv")
    gran_f = pd.read_csv(OUT_C / "metricas" / "fixo_granularidade.csv")
    sens = pd.read_csv(OUT_C / "kmeans_sensibilidade_variaveis.csv")
    curva = pd.read_csv(OUT_C / "kmeans_cotovelo.csv")
    estab = pd.read_csv(OUT_C / "kmeans_estabilidade_cotovelo.csv")
    trat_c = pd.read_csv(OUT_C / "eda" / "tratamentos.csv")
    out_c = pd.read_csv(OUT_C / "eda" / "outliers.csv")
    estr_c = pd.read_csv(OUT_C / "eda" / "estrutura.csv")

    estr_e = pd.read_csv(OUT_E / "tabelas" / "estrutura.csv")
    falt_e = pd.read_csv(OUT_E / "tabelas" / "faltantes.csv")
    trat_e = pd.read_csv(OUT_E / "tabelas" / "tratamentos.csv")
    out_e = pd.read_csv(OUT_E / "tabelas" / "outliers_iqr.csv")
    sens_e = pd.read_csv(OUT_E / "tabelas" / "sensibilidade_duplicatas.csv")
    perfil = pd.read_csv(OUT_E / "tabelas" / "perfil_risco.csv")
    met_m = json.loads((OUT_E / "tabelas" / "modelo_metricas.json").read_text(encoding="utf-8"))
    resumo_e = json.loads((OUT_E / "tabelas" / "resumo.json").read_text(encoding="utf-8"))

    p1, p2 = res_c.iloc[0], res_c.iloc[1]
    fmt_met = {"precisao": dec(2), "recall": dec(2), "f1": dec(2)}
    ren_met = {"classe": "Classe", "suporte_real": "Reais", "previstos": "Previstos", "acertos": "Acertos",
               "precisao": "Precisão", "recall": "Recall", "f1": "F1"}
    fmt_gran = {"acuracia_mapeada": pct(), "ari": dec(), "nmi": dec(), "acuracia_chute_majoritario": pct()}
    ren_gran = {"agrupamento_de_classes": "Classes reavaliadas", "n_classes": "Nº", "acuracia_mapeada": "Acurácia",
                "ari": "ARI", "nmi": "NMI", "acuracia_chute_majoritario": "Linha de base"}

    h = []
    a = h.append
    # ---------------------------------------------------------------- capa
    a(f"""<section class="capa">
<p class="sub">Análise e Visualização de Dados (AVD) — 2026.2 · Avaliação II (AV2)</p>
<h1>Projeto Final: Agrupamento, EDA e Dashboard Interativo</h1>
<p style="font-size:13pt">Parte Comum: K-Means no dataset <b>Obesity</b> (UCI 544)<br>
Parte Específica: <b>Por que os hóspedes cancelam?</b> — Hotel Booking Demand</p>
<p style="margin-top:40mm"><b>Integrantes</b><br>{"<br>".join(INTEGRANTES)}</p>
<p class="sub" style="margin-top:20mm">{HOJE}</p>
</section>""")

    # ---------------------------------------------------------------- sumário / intro
    a("""<h2 class="quebra">Sumário</h2>
<ol>
<li>Introdução e organização do repositório</li>
<li><b>Parte Comum</b> — EDA do dataset Obesity (4 pilares)</li>
<li>Padronização e escolha das variáveis</li>
<li>K-Means: Passada 1 (k livre) e Passada 2 (k fixo)</li>
<li>Mapeamento, matrizes de confusão e métricas</li>
<li>Discussão crítica</li>
<li><b>Parte Específica</b> — problema, pergunta de pesquisa e público-alvo</li>
<li>EDA do Hotel Booking Demand (4 pilares, faltantes e outliers)</li>
<li>Visualizações exploratórias e notas analíticas</li>
<li>Storytelling e visualizações explanatórias</li>
<li>O risco é previsível? Modelo de classificação</li>
<li>Dashboard em Streamlit</li>
<li>Conclusões, limitações e uso de IA</li>
</ol>
<h2>1. Introdução e organização do repositório</h2>
<p>Este relatório documenta as duas partes do Projeto Final. Na <b>Parte Comum</b>, exploramos o dataset <i>Obesity</i>,
padronizamos as variáveis e aplicamos o K-Means em duas passadas (k escolhido pelo Método do Cotovelo e k igual ao número
real de classes), avaliando os grupos por matrizes de confusão. Na <b>Parte Específica</b>, investigamos o cancelamento de
reservas em dois hotéis portugueses, construímos uma história explanatória para a gestão do hotel e um dashboard interativo
em Streamlit.</p>
<p>Todo o código é reprodutível: os scripts de <code>src/</code> geram as figuras e tabelas de <code>out/</code> usadas neste
relatório, e o dashboard usa as mesmas funções (os números batem). Bibliotecas: pandas, numpy, matplotlib, seaborn, plotly,
scikit-learn e streamlit, em Python 3.12.</p>
<table><tr><th>Caminho</th><th>Conteúdo</th></tr>
<tr><td><code>notebooks/parte_comum/eda.ipynb</code>, <code>kmeans_metricas.ipynb</code></td><td>Parte Comum executada, com saídas e comentários</td></tr>
<tr><td><code>src/parte_comum/eda.py · kmeans.py · metricas.py</code></td><td>Mesmas análises pela linha de comando (<code>--dataset obesity --modo livre|fixo</code>)</td></tr>
<tr><td><code>src/parte_especifica/eda.py</code></td><td>EDA, figuras estáticas e modelo da Parte Específica</td></tr>
<tr><td><code>src/parte_especifica/dashboard.py</code> + <code>app_pages/</code></td><td>Dashboard Streamlit (7 páginas, inclui a Parte Comum)</td></tr>
<tr><td><code>out/parte_comum/</code>, <code>out/parte_especifica/</code></td><td>Figuras (PNG) e tabelas (CSV/JSON) geradas</td></tr>
<tr><td><code>data/</code></td><td>Datasets e <code>baixar_dados.py</code></td></tr></table>
""")

    # ================================================================ PARTE COMUM
    a('<h2 class="parte">PARTE COMUM — Dataset Obesity</h2>')
    a(f"""<h2>2. Análise exploratória (EDA)</h2>
<p>O dataset <i>Estimation of obesity levels based on eating habits and physical condition</i> (UCI 544; Palechor &amp; de la Hoz
Manotas, 2019) tem <b>2.111 pessoas</b> do México, Peru e Colômbia e <b>17 colunas</b>: 16 atributos e a classe
<code>NObeyesdad</code> com 7 níveis. Segundo os autores, <b>77% das linhas foram geradas sinteticamente com SMOTE</b> e só 23%
vêm do questionário web — fato que reaparece em toda a análise.</p>
<h3>2.1 Pilar estrutural e qualidade dos dados</h3>
{tabela(estr_c[["coluna", "descricao", "tipo_estatistico", "valores_unicos", "faltantes"]],
        renomear={"coluna": "Coluna", "descricao": "Descrição", "tipo_estatistico": "Tipo", "valores_unicos": "Únicos", "faltantes": "Faltantes"})}
<p>São 8 variáveis numéricas (3 contínuas e 5 escalas de questionário), 5 binárias, 2 ordinais de frequência, 1 nominal
(MTRANS) e a classe, que é <b>ordinal</b> (ordenada pelo IMC). Não há valores ausentes. Encontramos duas questões de qualidade:</p>""")
    a(tabela(trat_c, renomear={"etapa": "Etapa", "linhas_afetadas": "Linhas", "diagnostico": "Diagnóstico", "tratamento": "Tratamento"}))
    a(img(OUT_C / "eda" / "04_inconsistencia_sintetica.png",
          "Figura 1 — Escalas de questionário com valores fracionários (rastro do SMOTE). Tratamento: arredondamento."))
    a(f"""<h3>2.2 Pilar distribucional</h3>
{img(OUT_C / "eda" / "01_classes.png", "Figura 2 — Distribuição da classe: 7 níveis praticamente balanceados (12,8% a 16,8%), efeito do SMOTE.")}
<p>As classes estão quase balanceadas — algo improvável em dados reais de obesidade. Consequência: "chutar" sempre a classe
mais comum acerta só <b>16,8%</b>, a linha de base para julgar o K-Means.</p>
{img(OUT_C / "eda" / "02_numericas.png", "Figura 3 — Distribuição das variáveis numéricas (ordinais já arredondadas).", alta=True)}
<p><b>Idade</b> é fortemente assimétrica à direita (mediana 22,8 anos; assimetria +1,5): a amostra é jovem. <b>Altura</b> é
aproximadamente simétrica; <b>peso</b> e <b>IMC</b> são espalhados de forma quase uniforme (13 a 51 kg/m²), de novo pelo
balanceamento artificial. Nas escalas, 70% fazem 3 refeições, 53% bebem 1–2 L de água, 70% fazem pouca ou nenhuma atividade
física (FAF 0–1).</p>
{img(OUT_C / "eda" / "03_categoricas.png", "Figura 4 — Distribuição das variáveis qualitativas.")}
<p>Gênero é equilibrado (50% masculino); 82% têm histórico familiar de sobrepeso e 88% consomem alimentos calóricos com
frequência. Fumar (2%), monitorar calorias (5%) e moto/bicicleta (&lt; 1%) são <b>muito raros</b> — o que terá efeito no K-Means.</p>
<h3>2.3 Pilar relacional</h3>
{img(OUT_C / "eda" / "05_correlacao.png", "Figura 5 — Correlação de Spearman entre todas as variáveis codificadas e a classe (0 = Abaixo do peso … 6 = Obesidade III).", alta=True)}
<p>Usamos Spearman porque metade das variáveis é ordinal. <b>A classe é quase uma função do IMC (ρ = 0,99)</b> e do peso
(ρ = 0,92): os níveis de obesidade são <i>definidos</i> por faixas de IMC. Entre os hábitos, as associações com a classe são
moderadas ou fracas: histórico familiar 0,50; idade 0,40; beliscar entre refeições −0,37; alimentos calóricos 0,25;
vegetais 0,21; atividade física −0,20.</p>
{img(OUT_C / "eda" / "07_imc_por_classe.png", "Figura 6 — IMC por classe com os cortes da OMS: as classes são faixas de IMC; Sobrepeso I e II se sobrepõem.")}
{img(OUT_C / "eda" / "08_altura_peso.png", "Figura 7 — Altura × peso por classe com curvas de IMC constante.")}
<p>Sobrepeso I (IMC 22,8–28,8) e Sobrepeso II (25,7–30,4) <b>se sobrepõem</b> — e a própria OMS não divide sobrepeso em dois
níveis. Obesidade II e III também se tocam (34–39,8 vs. 36,8–50,8).</p>
{img(OUT_C / "eda" / "09_categoricas_por_classe.png", "Figura 8 — Variáveis qualitativas por classe (barras 100%).", alta=True)}
<p><b>Artefato do SMOTE:</b> Obesidade II tem 295 homens e 2 mulheres; Obesidade III tem 323 mulheres e 1 homem, e 323 das
324 pessoas dessa classe têm exatamente os mesmos valores de álcool, beliscar e transporte. O SMOTE criou essas classes a partir
de poucos casos reais, copiando seus atributos — o que as torna artificialmente homogêneas.</p>
{img(OUT_C / "eda" / "06_numericas_por_classe.png", "Figura 9 — Variáveis numéricas por classe: o peso separa as classes; os hábitos variam pouco.")}
<h3>2.4 Pilar de outliers</h3>
{img(OUT_C / "eda" / "10_outliers.png", "Figura 10 — Variáveis numéricas padronizadas; pontos laranja fora de 1,5×IQR.")}
{tabela(out_c[["variavel", "q1", "q3", "limite_inf", "limite_sup", "n_outliers_iqr", "n_outliers_z3", "min", "max"]],
        formatos={c: dec(2) for c in ["q1", "q3", "limite_inf", "limite_sup", "min", "max"]},
        renomear={"variavel": "Variável", "n_outliers_iqr": "Fora de 1,5×IQR", "n_outliers_z3": "|z| > 3", "limite_inf": "Lim. inf.", "limite_sup": "Lim. sup."})}
<p><b>Decisão: nenhum outlier removido.</b> As 167 idades acima de ~35 anos, a altura de 1,98 m e o peso de 173 kg são
plausíveis e definem justamente as classes extremas. NCP e CH2O "acusam" centenas de outliers só porque o IQR deles é zero (escala
de 3–4 pontos concentrada em um valor) — a regra de Tukey não se aplica bem a elas. A padronização controla o efeito de escala.</p>
<div class="destaque"><b>Síntese da EDA.</b> Base limpa com 2.087 pessoas. A classe é, na prática, o IMC discretizado; hábitos
têm associação fraca a moderada com ela; o balanceamento e a homogeneidade de algumas classes são artefatos do SMOTE.
Expectativa: o K-Means vai refletir gênero e peso mais do que as 7 classes.</div>""")

    # ---------------------------------------------------------------- padronização
    a("""<h2 class="quebra">3. Padronização e escolha das variáveis</h2>
<p><b>Variáveis usadas (15):</b> as 8 numéricas, as 5 binárias codificadas como 0/1 e as 2 ordinais de frequência (CAEC e CALC)
codificadas de 0 (<i>no</i>) a 3 (<i>Always</i>), preservando a ordem. <b>MTRANS foi excluída</b>: é nominal sem ordem (virar
0–4 inventaria distâncias falsas) e o <i>one-hot</i> transformaria uma única variável em 5 dimensões, dando a ela ~5× o peso
das demais. A classe, obviamente, não entra.</p>
<p><b>Por que padronizar?</b> O K-Means atribui cada ponto ao centróide mais próximo pela <b>distância euclidiana</b>. Sem
padronização, cada variável pesa proporcionalmente à sua escala: o peso tem variância ≈ 686 kg², a altura ≈ 0,009 m² e as escalas
de questionário ≈ 0,3–0,8. A distância seria praticamente só a diferença de peso, e trocar a unidade (kg → g) mudaria os
grupos. Com o <b>z-score</b> (<code>StandardScaler</code>: z = (x − média)/desvio), todas as variáveis passam a ter média 0 e
desvio 1 e contribuem igualmente.</p>
<p><b>Limitação registrada:</b> variáveis binárias muito raras (fumar: 2%; monitorar calorias: 5%) também ganham valores z
altos (≈ +6,8 e +4,6) após a padronização — e, como veremos, o K-Means dedica grupos a elas.</p>""")

    # ---------------------------------------------------------------- k-means
    a(f"""<h2>4. K-Means em duas passadas</h2>
<p>Configuração: <code>KMeans(n_init=20, random_state=42)</code> do scikit-learn.</p>
<h3>4.1 Passada 1 — k livre (Método do Cotovelo)</h3>
<p>Rodamos k = 1 a 10 (k = 1 como linha de base: inércia total). Para não escolher o cotovelo "no olho", usamos um critério
objetivo — o ponto da curva <b>mais distante da reta</b> que liga o primeiro ao último ponto (<i>kneedle</i>).</p>
{img(OUT_C / "kmeans" / "livre_cotovelo.png", "Figura 11 — Método do Cotovelo e ganho marginal de cada grupo adicional.")}
{tabela(curva, formatos={"inercia": dec(0), "silhueta": dec(3), "queda_pct": dec(1), "distancia_corda": dec(3)},
        renomear={"inercia": "Inércia", "silhueta": "Silhueta", "queda_pct": "Queda vs k−1 (%)", "distancia_corda": "Dist. à reta"})}
<p>A curva é <b>suave</b>, sem cotovelo nítido — sinal de que os dados não têm grupos bem separados. O critério indica
<b>k = 4</b>, estável em 5 sementes; depois dele o ganho marginal cai de ~8% para ~6% por grupo. Sem o k = 1, o critério
oscila entre 6 e 7 conforme a semente — mais uma evidência de que não há um k "natural" forte. A silhueta (complementar) fica
abaixo de 0,25 em todo k.</p>
{tabela(estab, renomear={"semente": "Semente", "k_cotovelo_1a10": "k (curva 1–10)", "k_cotovelo_2a10": "k (curva 2–10)"})}
{img(OUT_C / "kmeans" / "livre_pca.png", "Figura 12 — Passada 1 (k = 4) na projeção PCA (esq.) e classes reais (dir.).")}
{img(OUT_C / "kmeans" / "livre_centroides.png", "Figura 13 — Perfil dos centróides da Passada 1 em z-score.")}
<p><b>Quem são os 4 grupos:</b> G0 — jovens (21 anos), leves (57 kg), sem histórico familiar → Abaixo do peso/Normal;
G1 — <b>mulheres</b> (96%) pesadas (99 kg) → 322 das 324 pessoas com Obesidade III; G2 — os 44 <b>fumantes</b>;
G3 — <b>homens</b> (99%) altos e pesados (93 kg) → Obesidade II. O K-Means separou por <b>gênero + peso</b>, não pelos 7 níveis.</p>
<h3>4.2 Passada 2 — k fixo = 7 (número real de classes)</h3>
{img(OUT_C / "kmeans" / "fixo_pca.png", "Figura 14 — Passada 2 (k = 7) na projeção PCA e classes reais.")}
{img(OUT_C / "kmeans" / "fixo_centroides.png", "Figura 15 — Perfil dos centróides da Passada 2 em z-score.")}
<p>Com 7 grupos surgem perfis mais finos, mas <b>dois grupos são "gastos" com atributos raros</b>: G6 reúne os 44 fumantes
(z = +6,8) e G4 quem monitora calorias (z = +4,6). G0 é quase puro: 321 das 367 pessoas são Obesidade III.</p>""")

    # ---------------------------------------------------------------- métricas
    a(f"""<h2 class="quebra">5. Mapeamento, matrizes de confusão e métricas</h2>
<p>Cada grupo recebe o rótulo da <b>classe real majoritária</b> dentro dele. A tabela de contingência (classe real × grupo) é
<b>retangular</b> (7 × k); a matriz de confusão (classe real × classe mapeada) é 7 × 7. As métricas foram calculadas
<b>diretamente da matriz</b> e conferidas com o scikit-learn: acurácia = diagonal/total; precisão de X = acertos de X / total da
coluna X; recall de X = acertos de X / total da linha X. Classe nunca prevista tem precisão 0/0 (indefinida, "—"), contada como 0
na média macro.</p>
<h3>5.1 Passada 1 (k = 4)</h3>
{img(OUT_C / "metricas" / "livre_contingencia.png", "Figura 16 — Passada 1: tabela de contingência 7 × 4 (em negrito, a classe majoritária de cada grupo).")}
{img(OUT_C / "metricas" / "livre_matriz_confusao.png", "Figura 17 — Passada 1: matriz de confusão (contagem e % da linha).")}
{tabela(met_l, fmt_met, ren_met)}
<h3>5.2 Passada 2 (k = 7)</h3>
{img(OUT_C / "metricas" / "fixo_contingencia.png", "Figura 18 — Passada 2: tabela de contingência 7 × 7.")}
{img(OUT_C / "metricas" / "fixo_matriz_confusao.png", "Figura 19 — Passada 2: matriz de confusão.")}
{tabela(met_f, fmt_met, ren_met)}
{img(OUT_C / "metricas" / "comparacao_precisao_recall.png", "Figura 20 — Precisão e recall por classe nas duas passadas.")}
{tabela(res_c[["passada", "k", "acuracia", "precisao_macro", "recall_macro", "ari", "nmi", "classes_nunca_previstas"]],
        {"acuracia": pct(), "precisao_macro": dec(3), "recall_macro": dec(3), "ari": dec(3), "nmi": dec(3)},
        {"passada": "Passada", "acuracia": "Acurácia", "precisao_macro": "Precisão macro", "recall_macro": "Recall macro",
         "ari": "ARI", "nmi": "NMI", "classes_nunca_previstas": "Classes nunca previstas"})}
<p><b>Leitura.</b> Passada 1: acurácia {p1.acuracia:.1%} com 4 classes nunca previstas; só Obesidade III (recall 0,99) e Abaixo do
peso (0,70) são bem recuperadas. Passada 2: acurácia {p2.acuracia:.1%} — 2,7× a linha de base de 16,8% —, 6 classes previstas;
Obesidade III com precisão 0,87 e recall 0,99. <b>Peso normal nunca é previsto</b>: essas pessoas se espalham pelos grupos e não
são maioria em nenhum. Os erros se concentram em <b>classes vizinhas</b> (Sobrepeso I ↔ II ↔ Obesidade I; Abaixo do peso ↔
Normal), seguindo a ordem do IMC. O ARI (que desconta o acaso) sobe de 0,18 para 0,24: concordância real, mas baixa.</p>
<p><i>Importante:</i> o K-Means nunca viu os rótulos; a "acurácia" mede a <b>coincidência</b> entre grupos descobertos e
classes conhecidas, não um acerto de previsão.</p>""".replace(f"{p1.acuracia:.1%}", f"{p1.acuracia:.1%}".replace(".", ","))
      .replace(f"{p2.acuracia:.1%}", f"{p2.acuracia:.1%}".replace(".", ",")))

    # ---------------------------------------------------------------- discussão
    a(f"""<h2 class="quebra">6. Discussão crítica</h2>
<h3>6.1 Passada 1 (k livre) × Passada 2 (k fixo): qual gerou grupos mais coerentes?</h3>
<ul>
<li><b>Coerência com as classes reais:</b> a Passada 2 vence em tudo — acurácia (44,8% vs 37,2%), precisão macro (0,40 vs 0,17),
recall macro (0,44 vs 0,37), ARI (0,24 vs 0,18). Era esperado: 7 grupos dão "vagas" para mais classes.</li>
<li><b>Coerência interna:</b> nenhuma é boa. Silhueta baixa nas duas (0,12 vs 0,15) e cotovelo sem joelho claro. Os 4 grupos
do cotovelo fazem sentido <b>como perfis</b> (jovens leves, mulheres pesadas, homens pesados, fumantes) — só não são as classes
de obesidade.</li>
<li><b>Conclusão:</b> os dados têm estrutura mais forte em <b>gênero + peso</b> do que nos 7 níveis. A Passada 2 só se aproxima
das classes porque é "forçada" a 7 grupos, e ainda gasta 2 deles com fumantes e com quem monitora calorias. O k do cotovelo
(4) não bater com o número de classes (7) não é erro: é um resultado sobre a estrutura dos dados.</li>
</ul>
<h3>6.2 A granularidade das classes faz sentido do ponto de vista dos dados?</h3>
<p>Reavaliamos <b>os mesmos agrupamentos</b> contra versões mais grossas das classes:</p>
<h4>Passada 1 (k = 4)</h4>{tabela(gran_l, fmt_gran, ren_gran)}
<h4>Passada 2 (k = 7)</h4>{tabela(gran_f, fmt_gran, ren_gran)}
<ul>
<li>Juntar <b>Sobrepeso I + II</b> melhora a acurácia (37,2% → 39,5% e 44,8% → 46,8%) sem piorar o ARI: os dados <b>não
distinguem</b> esses dois níveis (faixas de IMC sobrepostas; a OMS não faz essa divisão).</li>
<li>Com as 4 classes da OMS a acurácia sobe para 55–58%, mas a linha de base também sobe (46,6%) e o ARI <b>não</b> melhora —
juntar classes facilita a acurácia "de graça".</li>
<li>Abaixo do peso × Normal e Sobrepeso II × Obesidade I também se misturam: a fronteira é um corte arbitrário numa variável
contínua (IMC), não uma separação natural.</li>
</ul>
<h3>6.3 O que define a classe?</h3>
{tabela(sens, {"acuracia_mapeada": pct(), "ari": dec(), "silhueta": dec()},
        {"conjunto": "Variáveis usadas (k = 7)", "n_variaveis": "Nº", "acuracia_mapeada": "Acurácia", "ari": "ARI", "silhueta": "Silhueta"})}
<p>Só <b>altura e peso</b> dão a mesma acurácia que as 15 variáveis (44,8%) com silhueta 2,6× maior: a classe é o IMC. Só
hábitos e perfil ainda acertam 39%, parte por causa do artefato "Obesidade III = mulheres com hábitos idênticos". Incluir MTRANS
por one-hot <b>piora</b> (40,7%), confirmando a decisão de excluí-la. (Esta tabela não foi usada para escolher variáveis — isso
seria usar a classe para ajustar um método não supervisionado —, apenas para a discussão.)</p>
<div class="destaque"><b>Conclusões da Parte Comum.</b> (1) A granularidade de 7 classes não é sustentada pelos dados como
estrutura natural; Sobrepeso I e II são indistinguíveis e uma divisão em 4–5 níveis (como a da OMS) seria mais defensável.
(2) O K-Means mede semelhança global entre pessoas, enquanto a classe depende de 2 variáveis — por isso o agrupamento revela
outras estruturas reais (gênero, fumo). (3) Parte do "acerto" em Obesidade III é artefato do SMOTE.</div>""")

    # ================================================================ PARTE ESPECÍFICA
    a('<h2 class="parte">PARTE ESPECÍFICA — Cancelamento de reservas de hotel</h2>')
    a(f"""<h2>7. Problema, pergunta de pesquisa e público-alvo</h2>
<p><b>Dataset:</b> <i>Hotel Booking Demand</i> (Kaggle, jessemostipak; artigo original de Antonio, Almeida &amp; Nunes, 2019,
<i>Data in Brief</i> 22). <b>{inteiro(resumo_e["linhas_brutas"])} reservas × {int(resumo_e["colunas_brutas"])} colunas</b>,
quantitativas e qualitativas, de dois hotéis em Portugal — um hotel urbano (Lisboa) e um resort (Algarve) — com chegadas
de julho de 2015 a agosto de 2017. <b>Tipo de problema: classificação</b> (alvo <code>is_canceled</code>).</p>
<div class="destaque"><p><b>Pergunta de pesquisa.</b> {html.escape(notas.PERGUNTA)}</p>
<p><b>Hipótese.</b> {html.escape(notas.HIPOTESE)}</p></div>
<p><b>Público-alvo.</b> {html.escape(notas.PUBLICO)}</p>

<h2>8. EDA — estrutura, qualidade, faltantes e outliers</h2>
<h3>8.1 Pilar estrutural</h3>
{tabela(estr_e[["coluna", "descricao", "tipo_estatistico", "valores_unicos", "faltantes"]],
        renomear={"coluna": "Coluna", "descricao": "Descrição", "tipo_estatistico": "Tipo", "valores_unicos": "Únicos", "faltantes": "NaN"})}
<p>Há 16 variáveis quantitativas, 15 qualitativas e 1 data. <code>reservation_status</code> e
<code>reservation_status_date</code> registram o desfecho (vazamento do alvo) e não entram em análises de causa nem no modelo.
Os nomes originais das colunas foram mantidos; os valores foram traduzidos para português e as variáveis derivadas
(total de noites, de hóspedes, faixa de antecedência, receita estimada etc.) têm nomes em português sem acentos.</p>
<h3>8.2 Dados faltantes: mecanismo e tratamento</h3>
{img(OUT_E / "figuras" / "q1_faltantes.png", "Figura 21 — Valores ausentes por variável e mecanismo de ausência diagnosticado.")}
{nota("q1_faltantes")}
{tabela(falt_e, {"pct": lambda v: f"{v:.3f}%".replace(".", ",")}, {"coluna": "Coluna", "faltantes_nan": "NaN", "faltantes_undefined": "'Undefined'", "total": "Total", "pct": "% linhas"})}
<p>Para diagnosticar o mecanismo, comparamos as reservas <b>com</b> e <b>sem</b> o valor ausente em variáveis observadas
(hotel, segmento, taxa de cancelamento). <b>Nenhum dado faltante foi simplesmente removido</b>:</p>
<ul>
<li><b>company (94,3%) e agent (13,7%) — ausência estrutural ("não se aplica")</b>: reservas sem empresa pagadora ou feitas
sem agência. Viraram as variáveis <code>tem_empresa</code> e <code>tem_agente</code>. A ausência é informativa: 17,5% de
cancelamento com empresa vs. 38,2% sem; 24,7% sem agente vs. 39,0% com.</li>
<li><b>country (0,4%) — MAR</b>: 95% no resort, concentrado em reservas Corporativas/Diretas, com 13,7% de cancelamento (vs. 37%).
A ausência depende de variáveis observadas; excluir enviesaria a base. Mantido como categoria "Desconhecido".</li>
<li><b>children (4 linhas), market_segment (2) e distribution_channel (5) — MCAR</b>: volumes ínfimos sem padrão. Imputados pela
moda (0 crianças; segmento mais comum do hotel urbano; canal deduzido do segmento).</li>
<li><b>meal = 'Undefined' (1,0%)</b>: pela documentação, é o mesmo que 'SC' (sem refeição) — unificado.</li>
</ul>
<h3>8.3 Inconsistências, linhas repetidas e outliers</h3>
{tabela(trat_e[["etapa", "colunas", "linhas_afetadas", "tratamento"]], {"linhas_afetadas": inteiro},
        {"etapa": "Etapa", "colunas": "Colunas", "linhas_afetadas": "Linhas", "tratamento": "Tratamento"})}
<p><b>Linhas repetidas (26,8%).</b> Não há número de reserva no dataset; linhas idênticas são, em sua maioria, quartos de uma
mesma reserva de grupo (98% das reservas "Não reembolsáveis" se repetem). Como cada linha é um quarto — o que importa para a
ocupação —, <b>mantivemos</b> as linhas e fizemos uma <b>análise de sensibilidade</b> (também disponível no dashboard):</p>
{tabela(sens_e, {c: pct() for c in sens_e.columns if c.startswith("taxa")} | {"reservas": inteiro},
        {"base": "Base", "reservas": "Reservas", "taxa_geral": "Taxa geral", "taxa_0_7_dias": "0–7 dias", "taxa_mais_1_ano": "> 1 ano",
         "taxa_sem_pedido": "Sem pedido esp.", "taxa_com_pedido": "Com pedido esp.", "taxa_nao_reembolsavel": "Não reembolsável"})}
<p>Sem as repetidas a taxa geral cai de 37% para 28%, mas <b>todas as conclusões se mantêm</b>: o cancelamento cresce com a
antecedência e quem faz pedido especial cancela menos.</p>
{img(OUT_E / "figuras" / "q2_outliers.png", "Figura 22 — Outliers das principais variáveis numéricas por hotel (após correção dos erros).")}
{nota("q2_outliers")}
{tabela(out_e[["variavel", "q1", "q3", "limite_sup", "max", "n_outliers", "pct_outliers"]],
        {"q1": dec(1), "q3": dec(1), "limite_sup": dec(1), "max": dec(1), "n_outliers": inteiro, "pct_outliers": lambda v: f"{v:.1f}%".replace(".", ",")},
        {"variavel": "Variável", "limite_sup": "Lim. sup. (Tukey)", "max": "Máximo", "n_outliers": "Outliers", "pct_outliers": "%"})}""")

    # ---------------------------------------------------------------- exploratórias
    a('<h2 class="quebra">9. Visualizações exploratórias e notas analíticas</h2>'
      "<p>As seis visualizações a seguir cobrem os pilares distribucional e relacional, sempre em relação ao alvo. No dashboard "
      "elas respondem a filtros (hotel, ano, meses, segmento, tipo de cliente e remoção das linhas repetidas).</p>")
    for chave, arq, alta in [("e1", "e1_antecedencia", False), ("e2", "e2_sazonalidade", False), ("e3", "e3_correlacao", True),
                             ("e4", "e4_adr", False), ("e5", "e5_segmento_deposito", False), ("e6", "e6_compromisso", False)]:
        a(f"<div class='bloco'><h3>{html.escape(notas.NOTAS[chave]['titulo'])}</h3>"
          + img(OUT_E / "figuras" / f"{arq}.png", html.escape(notas.NOTAS[chave]["titulo"]), alta=alta) + "</div>")
        a(nota(chave))

    # ---------------------------------------------------------------- storytelling
    a(f'<h2 class="quebra">10. Storytelling: da exploratória à explanatória</h2>'
      f'<div class="destaque"><b>Grande ideia.</b> {html.escape(notas.GRANDE_IDEIA)}</div>'
      f"<p><b>Público-alvo.</b> {html.escape(notas.PUBLICO)}</p><h3>Raciocínio registrado</h3>")
    for titulo, texto in notas.RACIOCINIO:
        a(f"<h4>{html.escape(titulo)}</h4><p>{html.escape(texto)}</p>")
    for chave, arq in [("x1", "x1_antecedencia"), ("x2", "x2_pedidos"), ("x3", "x3_perfis")]:
        a(f"<div class='bloco'><h3>{html.escape(notas.NOTAS[chave]['titulo'])}</h3>"
          + img(OUT_E / "figuras" / f"{arq}.png", html.escape(notas.NOTAS[chave]["titulo"])) + "</div>")
        a(nota(chave))
    a(tabela(perfil, {"reservas": inteiro, "canceladas": inteiro, "taxa": pct(), "pct_reservas": pct(), "pct_cancelamentos": pct()},
             {"perfil": "Perfil", "reservas": "Reservas", "canceladas": "Canceladas", "taxa": "Taxa", "pct_reservas": "% das reservas",
              "pct_cancelamentos": "% dos cancelamentos"}))
    a("<h3>Recomendações para a gestão</h3><ol>" + "".join(f"<li>{html.escape(r)}</li>" for r in notas.RECOMENDACOES) + "</ol>")

    # ---------------------------------------------------------------- modelo
    a(f"""<h2 class="quebra">11. O risco é previsível? Modelo de classificação</h2>
<p>Para responder à segunda parte da pergunta, treinamos um <code>HistGradientBoostingClassifier</code> (scikit-learn) usando
<b>apenas informações conhecidas no momento da reserva</b>. Excluímos: <code>reservation_status(_date)</code> (desfecho),
<code>assigned_room_type</code> (definido no check-in), <code>booking_changes</code> (posterior à reserva),
<code>required_car_parking_spaces</code> (0% de cancelamento entre quem pediu vaga — indício de registro na chegada),
ano e IDs de agente/empresa. A divisão é <b>temporal</b>: treino com chegadas até dez/2016 ({inteiro(met_m["n_treino"])}
reservas) e teste com 2017 ({inteiro(met_m["n_teste"])}), o que simula o uso real e evita que quartos repetidos de um mesmo
grupo caiam no treino e no teste.</p>
<table><tr><th>AUC</th><th>Acurácia</th><th>Acurácia "nunca cancela"</th><th>Precisão</th><th>Recall</th><th>F1</th></tr>
<tr><td>{dec(2)(met_m["auc"])}</td><td>{pct()(met_m["acuracia"])}</td><td>{pct()(met_m["acuracia_baseline"])}</td>
<td>{pct()(met_m["precisao"])}</td><td>{pct()(met_m["recall"])}</td><td>{dec(2)(met_m["f1"])}</td></tr></table>
{img(OUT_E / "figuras" / "m2_matriz_confusao.png", "Figura — Matriz de confusão do modelo no teste (2017).")}
{nota("m2")}
{img(OUT_E / "figuras" / "m1_importancia.png", "Figura — Importância das variáveis por permutação (queda da AUC).")}
{nota("m1")}
<div class="destaque"><b>Resposta à pergunta de pesquisa.</b> As características da reserva que mais indicam risco são a
<b>antecedência longa</b> e a <b>ausência de pedidos especiais</b> (hipótese confirmada), além do segmento/canal (agências e
grupos), do país de origem (reservas domésticas) e do tipo de depósito (lotes não reembolsáveis de operadoras). O risco é
<b>parcialmente previsível</b> (AUC 0,84; 77% de acurácia contra 61% da linha de base): suficiente para <b>priorizar</b> ações
— depósito, reconfirmação, overbooking por perfil —, não para decidir reserva a reserva.</div>""")

    # ---------------------------------------------------------------- dashboard
    a("""<h2>12. Dashboard em Streamlit</h2>
<p>Execução: <code>streamlit run src/parte_especifica/dashboard.py</code>. O dashboard tem 7 páginas em duas seções:</p>
<table><tr><th>Página</th><th>Conteúdo</th></tr>
<tr><td>Visão geral</td><td>Pergunta, hipótese, público-alvo, grande ideia, KPIs e guia de navegação</td></tr>
<tr><td>História: o que fazer</td><td>As 3 explanatórias com notas, recomendações, raciocínio do storytelling e comparação lado a lado com/sem linhas repetidas</td></tr>
<tr><td>Exploração</td><td>As 6 exploratórias com notas analíticas, organizadas em seções</td></tr>
<tr><td>Explorador de variáveis</td><td>Escolha livre de X e Y: gráfico e medida de associação (Spearman, η², V de Cramér) se adaptam aos tipos — útil para interpretar ao vivo qualquer par de variáveis</td></tr>
<tr><td>Simulador de risco</td><td>Desempenho do modelo (ROC, matriz de confusão, importâncias) e simulação do risco de uma reserva hipotética, com curva de risco × antecedência</td></tr>
<tr><td>Qualidade dos dados</td><td>Estrutura, faltantes e evidências do mecanismo, outliers por variável, linhas repetidas e log de tratamentos</td></tr>
<tr><td>K-Means: duas passadas (Parte Comum)</td><td>Cotovelo, <b>animação</b> dos agrupamentos de k = 2 a 10, acurácia/ARI por k, Passada 1 × Passada 2 lado a lado, contingência, matrizes, métricas e discussão crítica</td></tr></table>
<p><b>Recursos de interatividade (bônus Visual/UX):</b> filtros globais interligados na barra lateral (hotel, ano, meses,
segmento e tipo de cliente com contagens que se atualizam conforme os demais filtros; os filtros persistem entre páginas);
botão de análise de sensibilidade (remover linhas repetidas); seleção interativa de variáveis/eixos; comparação lado a lado;
animação de transição entre valores de k do K-Means; página dedicada à Parte Comum; simulador com o modelo; e
<b>exportação</b> dos dados filtrados (CSV) e de um <b>relatório HTML</b> da seleção atual, com gráficos interativos e notas.</p>
<p><b>Boas práticas visuais aplicadas:</b> paleta com significado fixo em todo o projeto (azul = não cancelada, laranja =
cancelamento/risco, cinza = contexto; violeta/verde para os hotéis), checada para daltonismo; um único eixo Y por gráfico
(volume e taxa em painéis separados); títulos que dizem a conclusão nas explanatórias; rótulos diretos; nenhum gráfico de
pizza ou radar; dados pré-agregados para manter o dashboard rápido.</p>""")

    # ---------------------------------------------------------------- conclusões
    a("""<h2>13. Conclusões, limitações e uso de IA</h2>
<ul>
<li><b>Parte Comum:</b> o cotovelo sugere 4 grupos, que refletem gênero e peso; com k = 7 a concordância com as classes sobe
(44,8%, ARI 0,24), mas os dados não sustentam 7 níveis distintos — em especial Sobrepeso I e II. A classe é o IMC
discretizado e parte da estrutura vem de artefatos do SMOTE.</li>
<li><b>Parte Específica:</b> 37% das reservas são canceladas (39% da receita estimada). Antecedência e pedidos especiais são os
sinais mais acionáveis: o perfil "mais de 90 dias e nenhum pedido especial" concentra 47% dos cancelamentos em 27% das
reservas. O risco é parcialmente previsível (AUC 0,84).</li>
<li><b>Limitações:</b> dados de dois hotéis portugueses entre 2015 e 2017 (generalização limitada); ausência de ID de reserva
(linhas repetidas tratadas por sensibilidade); a receita é estimada (ADR × noites); o modelo é associativo, não causal;
na Parte Comum, os 77% de dados sintéticos limitam conclusões sobre a população real.</li>
<li><b>Uso de IA:</b> ferramentas de IA foram usadas no desenvolvimento, conforme permitido pelo enunciado e detalhado em
<code>docs/DECLARACAO_IA.pdf</code>, assinada por todos os integrantes.</li>
</ul>
<h3>Referências</h3>
<ul>
<li>Antonio, N.; Almeida, A.; Nunes, L. (2019). Hotel booking demand datasets. <i>Data in Brief</i>, 22, 41–49.</li>
<li>Palechor, F. M.; de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical
condition. <i>Data in Brief</i>, 25, 104344. UCI Machine Learning Repository, dataset 544.</li>
<li>Knaflic, C. N. (2015). <i>Storytelling with Data</i>. Wiley.</li>
<li>Satopää, V. et al. (2011). Finding a "kneedle" in a haystack: detecting knee points in system behavior. ICDCS Workshops.</li>
<li>Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i>, 12, 2825–2830.</li>
</ul>""")
    return (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Relatório — Projeto Final AVD 2026.2</title>'
            f"<style>{CSS}</style></head><body>{''.join(h)}</body></html>")


# --------------------------------------------------------------------------
def declaracao() -> str:
    linhas_uso = [
        ("Código (EDA, K-Means, métricas, modelo e dashboard)", "Código gerado e depurado com o assistente de IA a partir das instruções do grupo; revisado e executado pelos integrantes."),
        ("Limpeza, tratamento de faltantes e outliers", "Diagnósticos e tratamentos propostos com apoio da IA e decididos/validados pelo grupo."),
        ("Padronização, agrupamento e métricas", "Implementação e cálculo com apoio da IA; métricas conferidas com o scikit-learn."),
        ("Interpretação, storytelling e notas analíticas", "Textos redigidos com apoio da IA a partir dos resultados obtidos."),
        ("Visualizações e dashboard Streamlit", "Gráficos, paleta de cores e páginas construídos com apoio da IA."),
        ("Relatório, README e documentação", "Redigidos com apoio da IA e revisados pelo grupo."),
    ]
    tabela_uso = "".join(f"<tr><td>{html.escape(e)}</td><td>{html.escape(u)}</td></tr>" for e, u in linhas_uso)
    assinaturas = "".join(f"<div>{html.escape(n)}<br><span class='sub'>Assinatura · Data: ____/____/2026</span></div>"
                          for n in INTEGRANTES)
    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Declaração de Uso de IA</title>
<style>{CSS} body {{ font-size: 10pt; }} h3 {{ margin: 8pt 0 2pt; }} table {{ font-size: 8.4pt; }} .assinatura div {{ margin-top: 20pt; }} @page {{ margin: 14mm 16mm; }}</style></head><body>
<p class="sub">Análise e Visualização de Dados (AVD) — 2026.2 · Projeto Final (AV2)</p>
<h1>Declaração de Uso de Inteligência Artificial</h1>
<p><b>Integrantes:</b> {", ".join(INTEGRANTES)}</p>
<p>Declaramos, para os devidos fins, que utilizamos ferramentas de Inteligência Artificial no desenvolvimento deste projeto,
conforme permitido pelo enunciado ("o uso de ferramentas de IA está totalmente liberado em todas as etapas e requisitos").</p>
<h3>Ferramenta utilizada</h3>
<p><b>Claude Code</b> (assistente de programação da Anthropic, modelo Claude Opus 5.5), executado no editor VS Code,
como ferramenta principal de apoio.</p>
<h3>Etapas em que a IA foi utilizada</h3>
<table><tr><th style="width:38%">Etapa</th><th>Forma de uso</th></tr>{tabela_uso}</table>
<h3>Decisões do grupo</h3>
<p>A escolha dos datasets (Obesity na Parte Comum; Hotel Booking Demand na Parte Específica), o
direcionamento do projeto e a validação final dos resultados foram feitos pelos integrantes.</p>
<h3>Responsabilidade</h3>
<p>Declaramos que revisamos o conteúdo produzido com auxílio de IA, que compreendemos o funcionamento do projeto, as decisões
tomadas, os métodos utilizados e os resultados apresentados, e que estamos aptos a explicar qualquer parte do trabalho —
código, análises e interpretações — durante a defesa.</p>
<p style="margin-top:14pt">Local e data: ______________________, ____ de novembro de 2026.</p>
<section class="assinatura">{assinaturas}</section>
</body></html>"""


# --------------------------------------------------------------------------
def _navegador() -> str | None:
    candidatos = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for c in candidatos:
        if Path(c).exists():
            return c
    for nome in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge", "msedge", "chrome"):
        if shutil.which(nome):
            return shutil.which(nome)
    return None


def para_pdf(origem: Path, destino: Path) -> bool:
    nav = _navegador()
    if not nav:
        print(f"Navegador não encontrado: abra {origem} e use Imprimir > Salvar como PDF.")
        return False
    cmd = [nav, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
           f"--print-to-pdf={destino}", origem.resolve().as_uri()]
    subprocess.run(cmd, check=False, capture_output=True, timeout=180)
    ok = destino.exists() and destino.stat().st_size > 10_000
    print(("[ok] " if ok else "[falhou] ") + str(destino))
    return ok


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    rel_html = DOCS / "relatorio.html"
    rel_html.write_text(relatorio(), encoding="utf-8")
    dec_html = DOCS / "DECLARACAO_IA.html"
    dec_html.write_text(declaracao(), encoding="utf-8")
    para_pdf(rel_html, RAIZ / "relatorio.pdf")
    para_pdf(dec_html, DOCS / "DECLARACAO_IA.pdf")


if __name__ == "__main__":
    main()
