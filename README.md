# Projeto Final — Agrupamento, EDA e Dashboard Interativo

**Análise e Visualização de Dados (AVD) · 2026.2 · Avaliação II**

Integrantes: **Pedro de Melo Dhalia · Matheos de Oliveira Guerra · Miguel José França de Andrade**

| Parte | Dataset | O que foi feito |
|---|---|---|
| **Comum** | [Obesity — UCI 544](https://archive.ics.uci.edu/dataset/544) | EDA (4 pilares), padronização, K-Means em duas passadas (k pelo cotovelo e k = 7), mapeamento grupo → classe, matrizes de confusão, acurácia/precisão/recall e discussão crítica |
| **Específica** | [Hotel Booking Demand — Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) | EDA com mecanismos de ausência e outliers, storytelling para a gestão do hotel, 6 visualizações exploratórias + 3 explanatórias com notas analíticas, modelo de risco de cancelamento e dashboard Streamlit |

**Pergunta de pesquisa (Parte Específica):** quais características de uma reserva, já conhecidas no momento em que ela é
feita, indicam maior risco de cancelamento — e esse risco é previsível o suficiente para orientar a política de reservas?

**Principais resultados**

- *Parte Comum:* o cotovelo sugere **k = 4** (grupos de gênero + peso); com k = 7 a acurácia mapeada é **44,8%** (linha de
  base 16,8%). A classe é praticamente o IMC discretizado, e Sobrepeso I e II não são distinguíveis pelos dados.
- *Parte Específica:* 37% das reservas são canceladas. Reservas com **mais de 90 dias de antecedência e nenhum pedido
  especial** são 27% do total, mas **47% dos cancelamentos**. O modelo de risco atinge **AUC 0,84** no teste de 2017.

---

## Estrutura do repositório

```text
projeto-avd/
├─ README.md
├─ requirements.txt
├─ relatorio.pdf                      # relatório técnico completo (Partes Comum e Específica)
├─ .streamlit/config.toml             # tema do dashboard
├─ docs/
│  ├─ DECLARACAO_IA.pdf               # declaração de uso de IA (assinar)
│  ├─ MATRIZ_CONTRIBUICAO.md          # matriz de contribuição dos integrantes
│  ├─ relatorio.html                  # fonte do relatorio.pdf
│  └─ DECLARACAO_IA.html              # fonte da declaração
├─ data/
│  ├─ baixar_dados.py                 # baixa os datasets se não estiverem aqui
│  ├─ parte_comum/ObesityDataSet_raw_and_data_sinthetic.csv
│  └─ parte_especifica/hotel_bookings.csv
├─ out/
│  ├─ parte_comum/                    # eda/, kmeans/, metricas/ (figuras PNG + tabelas CSV/JSON)
│  └─ parte_especifica/               # figuras/ e tabelas/
├─ notebooks/
│  └─ parte_comum/
│     ├─ eda.ipynb                    # EDA executada, com comentários
│     └─ kmeans_metricas.ipynb        # padronização, 2 passadas, métricas e discussão
├─ scripts/
│  └─ gerar_documentos.py             # gera relatorio.pdf e docs/DECLARACAO_IA.pdf
└─ src/
   ├─ estilo.py                       # paleta de cores e estilo comuns
   ├─ parte_comum/
   │  ├─ dados.py                     # carga, limpeza e padronização
   │  ├─ eda.py
   │  ├─ kmeans.py
   │  ├─ metricas.py
   │  └─ interativo.py                # figuras da Parte Comum para o dashboard
   └─ parte_especifica/
      ├─ dados.py                     # carga, tratamento de faltantes/outliers, variáveis derivadas
      ├─ analise.py                   # agregações (usadas pelo relatório e pelo dashboard)
      ├─ modelo.py                    # classificador de risco de cancelamento
      ├─ notas.py                     # notas analíticas e textos do storytelling
      ├─ graficos.py                  # gráficos plotly do dashboard
      ├─ eda.py                       # EDA da Parte Específica (figuras estáticas + tabelas)
      ├─ app_comum.py                 # filtros, cache e exportação do dashboard
      ├─ dashboard.py                 # ponto de entrada do Streamlit
      └─ app_pages/                   # páginas do dashboard
```

## Instalação

Requer **Python 3.11+** (testado com 3.12). Na raiz do repositório:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Os dois datasets já estão em `data/`. Se precisar baixá-los de novo:

```bash
python data/baixar_dados.py
```

## Como executar

> Rode todos os comandos **a partir da raiz do repositório**.

### Parte Comum — opção 1: notebooks

```bash
jupyter notebook notebooks/parte_comum/eda.ipynb
jupyter notebook notebooks/parte_comum/kmeans_metricas.ipynb
```

Os notebooks já estão salvos com as saídas; os gráficos vêm das funções de `src/parte_comum/`.

### Parte Comum — opção 2: scripts

```bash
python -m src.parte_comum.eda      --dataset obesity               --out ./out/parte_comum/
python -m src.parte_comum.kmeans   --dataset obesity --modo livre  --out ./out/parte_comum/
python -m src.parte_comum.kmeans   --dataset obesity --modo fixo   --out ./out/parte_comum/
python -m src.parte_comum.metricas --dataset obesity               --out ./out/parte_comum/
```

### Parte Específica

```bash
python -m src.parte_especifica.eda --out ./out/parte_especifica/   # figuras, tabelas e modelo (~1 min)
streamlit run src/parte_especifica/dashboard.py                    # dashboard
```

Se o comando `streamlit` não for encontrado, use `python -m streamlit run src/parte_especifica/dashboard.py`.
Na primeira abertura, o dashboard leva alguns segundos para tratar os dados, treinar o modelo (página *Simulador*) e rodar
o K-Means (página da Parte Comum); depois tudo fica em cache.

### Relatório e declaração

```bash
python scripts/gerar_documentos.py
```

Monta `docs/relatorio.html` e `docs/DECLARACAO_IA.html` com as figuras e tabelas de `out/` e converte para PDF com o
Edge/Chrome em modo headless. Sem navegador disponível, abra os `.html` e use *Imprimir → Salvar como PDF*.

## Dashboard

| Página | Conteúdo |
|---|---|
| Visão geral | Pergunta, hipótese, público-alvo, grande ideia e KPIs |
| História: o que fazer | 3 visualizações explanatórias com notas, recomendações e comparação lado a lado com/sem linhas repetidas |
| Exploração | 6 visualizações exploratórias com notas analíticas |
| Explorador de variáveis | Escolha livre de X e Y; gráfico e medida de associação (Spearman, η², V de Cramér) se adaptam aos tipos |
| Simulador de risco | Desempenho do modelo e simulação do risco de uma reserva |
| Qualidade dos dados | Estrutura, faltantes e mecanismos, outliers, linhas repetidas e log de tratamentos |
| K-Means: duas passadas | Parte Comum: cotovelo, animação k = 2…10, passadas lado a lado, matrizes, métricas e discussão |

Interatividade: filtros globais interligados (hotel, ano, meses, segmento, tipo de cliente) que persistem entre páginas,
botão de análise de sensibilidade, seleção de variáveis, comparação lado a lado, animação do K-Means, simulador e
exportação dos dados filtrados (CSV) e de um relatório HTML da seleção.

## Documentos

- [relatorio.pdf](relatorio.pdf): análise técnica completa das duas partes.
- [docs/DECLARACAO_IA.pdf](docs/DECLARACAO_IA.pdf): declaração de uso de IA, a ser assinada por todos os integrantes.
- [docs/MATRIZ_CONTRIBUICAO.md](docs/MATRIZ_CONTRIBUICAO.md): contribuição de cada integrante.

## Fontes dos dados

- Antonio, N.; Almeida, A.; Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.
- Palechor, F. M.; de la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and
  physical condition*. Data in Brief, 25, 104344 (UCI Machine Learning Repository, dataset 544).
