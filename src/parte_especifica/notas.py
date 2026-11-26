"""Notas analíticas de cada visualização + textos do storytelling.

Cada nota responde às três perguntas pedidas no enunciado:
  * o_que   -> o que está sendo mostrado;
  * insight -> qual insight/informação pode ser extraído;
  * por_que -> por que esse tipo de visualização foi escolhido.

Os números citados se referem à BASE COMPLETA tratada (119.210 reservas, com
as linhas repetidas). No dashboard os gráficos mudam com os filtros; as notas
descrevem o padrão geral.
"""

PERGUNTA = (
    "Quais características de uma reserva, já conhecidas no momento em que ela é feita, indicam maior risco "
    "de cancelamento — e esse risco é previsível o suficiente para orientar a política de reservas do hotel?"
)
HIPOTESE = (
    "Reservas feitas com muita antecedência e sem sinais de compromisso do hóspede (nenhum pedido especial, "
    "cliente novo, sem relação prévia com o hotel) cancelam bem mais do que a média."
)
PUBLICO = (
    "Gestão dos dois hotéis — gerente geral e gerente de receitas/reservas (revenue manager). São pessoas de negócio, "
    "não técnicas: querem saber ONDE está o risco e O QUE FAZER (política de depósito, overbooking, contato de "
    "confirmação). Por isso as visualizações explanatórias usam poucos elementos, uma cor de destaque, números "
    "escritos no próprio gráfico e títulos que já dizem a conclusão."
)
GRANDE_IDEIA = (
    "Mais de 1 em cada 3 reservas é cancelada, e o risco já está escrito na reserva: quanto maior a antecedência e "
    "menor o envolvimento do hóspede (nenhum pedido especial), maior a chance de perder o quarto. O perfil "
    "'mais de 90 dias e nenhum pedido especial' é 27% das reservas, mas 47% dos cancelamentos."
)

RACIOCINIO = [
    ("1. O que a exploração encontrou",
     "A distribuição da antecedência (E1) mostrou que reservas canceladas são feitas muito antes (mediana de 113 dias "
     "contra 45). A correlação (E3) confirmou a antecedência como a variável numérica mais associada ao cancelamento "
     "(ρ = +0,32) e os pedidos especiais como o principal fator de proteção (ρ = −0,26). Os sinais de relacionamento "
     "(E6) mostraram que quem faz pedido especial cancela menos da metade (22% contra 48%)."),
    ("2. Que história os dados contam",
     "Os dois fatores se somam: em TODA faixa de antecedência, quem fez ao menos um pedido especial cancela cerca de "
     "metade (ou menos) do que quem não fez. Uma reserva longa e 'fria' (sem pedidos) é o perfil típico de "
     "cancelamento: reserva especulativa, feita para garantir preço/data, que é abandonada depois."),
    ("3. Por que essa mensagem é a mais relevante para o público",
     "Gestores não controlam a sazonalidade nem o país do hóspede, mas controlam a POLÍTICA de reservas: exigir "
     "depósito ou confirmação para reservas longas, calibrar overbooking por perfil e contatar hóspedes sem pedidos "
     "especiais. Antecedência e pedidos especiais são conhecidos no ato da reserva — dá para agir antes do prejuízo. "
     "Achados como 'Não reembolsável cancela 99%' foram deixados na exploração: são contraintuitivos, vêm de lotes "
     "de grupos (linhas repetidas) e confundiriam a mensagem principal."),
    ("4. Técnicas usadas para destacar a mensagem (atributos pré-atentivos)",
     "COR: um único laranja (a cor de 'cancelamento' em todo o projeto) marca o que importa — as faixas acima da média "
     "(X1), a linha 'nenhum pedido especial' (X2) e o perfil de risco (X3); todo o resto fica em cinza. TAMANHO/PESO: "
     "a linha de risco é mais grossa e os percentuais-chave estão em negrito e maiores. POSIÇÃO: barras ordenadas pela "
     "antecedência (ordem natural, da esquerda para a direita) e o perfil de risco sempre na primeira posição, à "
     "esquerda. TEXTO: o título de cada gráfico é a própria conclusão e uma linha de referência marca a média geral."),
]

RECOMENDACOES = [
    "Exigir depósito parcial ou reconfirmação para reservas com mais de 90 dias de antecedência e nenhum pedido "
    "especial (o perfil que concentra 47% dos cancelamentos).",
    "Incentivar pedidos especiais no fluxo de reserva (berço, andar, horário de chegada): além de melhorar a "
    "experiência, é o sinal de compromisso mais forte que os dados mostram.",
    "Calibrar o overbooking por perfil de risco em vez de usar uma taxa única: o hotel urbano (42% de cancelamento) "
    "e reservas de longo prazo comportam mais overbooking do que reservas de última hora (10%).",
    "Rever a política 'Não reembolsável' com grupos e operadoras: na prática ela não protege (99% cancelam), o que "
    "indica bloqueios de quartos liberados em lote — vale negociar prazos de liberação (release) em contrato.",
]

NOTAS = {
    # ------------------------------------------------------------- qualidade
    "q1_faltantes": dict(
        titulo="Valores ausentes por variável e mecanismo",
        o_que="Percentual de linhas com valor ausente (NaN ou o rótulo 'Undefined') em cada variável, colorido pelo "
              "mecanismo de ausência que diagnosticamos.",
        insight="Os dois maiores 'buracos' (company 94%, agent 14%) não são falhas: são ausências ESTRUTURAIS — reservas "
                "sem empresa ou sem agente. Viraram as variáveis tem_empresa e tem_agente, que inclusive ajudam a explicar "
                "o cancelamento (17,5% com empresa vs. 38,2% sem). country (0,4%) é MAR: concentrado no resort e em "
                "reservas corporativas/diretas que quase não cancelam (13,7%) — remover essas linhas enviesaria a base.",
        por_que="Barras horizontais ordenadas permitem comparar magnitudes muito diferentes (94% a 0,002%) com os nomes "
                "das variáveis legíveis; a cor codifica a categoria do mecanismo, não uma quantidade.",
    ),
    "q2_outliers": dict(
        titulo="Outliers das variáveis numéricas",
        o_que="Boxplots por hotel das principais variáveis numéricas; pontos além dos bigodes estão fora de 1,5×IQR (regra "
              "de Tukey).",
        insight="Há valores extremos plausíveis (antecedência de até 737 dias, diárias de até 510 €, estadias de 69 noites) "
                "que foram mantidos, e erros evidentes (diária de 5.400 € e de −6,38 €; 10 bebês com 2 adultos) que foram "
                "corrigidos. Em variáveis discretas com IQR = 0 (adultos, alterações) a regra de Tukey marca até 25% das "
                "linhas — ela não serve para essas variáveis.",
        por_que="O boxplot é o gráfico padrão para outliers: mostra mediana, quartis e pontos extremos de uma vez, e "
                "colocar os dois hotéis lado a lado evita confundir diferenças entre hotéis com valores atípicos.",
    ),
    # ----------------------------------------------------------- exploratórias
    "e1": dict(
        titulo="E1. Distribuição da antecedência — canceladas x não canceladas",
        o_que="Distribuição (em % de cada grupo) dos dias entre a reserva e a chegada, separando reservas canceladas e não "
              "canceladas. Linhas tracejadas = medianas.",
        insight="As reservas não canceladas se concentram nos primeiros dias (26% com até 10 dias de antecedência). As "
                "canceladas se espalham por prazos longos: mediana de 113 dias contra 45. A partir de ~40 dias a curva "
                "laranja fica sempre acima da azul — antecedência longa é um sinal de risco.",
        por_que="Histograma em degraus (step) normalizado por grupo: compara a FORMA de duas distribuições de tamanhos "
                "diferentes (75 mil x 44 mil) sem que uma barra esconda a outra, como aconteceria com barras sobrepostas.",
    ),
    "e2": dict(
        titulo="E2. Sazonalidade — volume e cancelamento por mês",
        o_que="Painel de cima: reservas por mês de chegada (média dos anos disponíveis). Painel de baixo: taxa de "
              "cancelamento no mesmo mês. Uma linha por hotel.",
        insight="O hotel urbano tem mais volume e cancela mais em todos os meses (42% vs. 28% no ano). O resort tem a "
                "taxa mais sazonal: 15% em janeiro e ~33% no verão (jun–set), quando a demanda é maior e o hóspede "
                "'segura' a vaga com antecedência. No urbano o pico de cancelamento é na primavera (abr–jun, ~45%).",
        por_que="Linhas são a forma natural para tempo. Volume e taxa têm escalas diferentes, então usamos DOIS painéis "
                "com o mesmo eixo X em vez de um gráfico com dois eixos Y (que induz comparações falsas).",
    ),
    "e3": dict(
        titulo="E3. Correlação entre variáveis numéricas",
        o_que="Matriz de correlação de Spearman entre as variáveis numéricas e o alvo (Cancelada = 1). Vermelho = "
              "correlação positiva, azul = negativa, cinza = nenhuma.",
        insight="Com o cancelamento: antecedência (+0,32) e cancelamentos anteriores (+0,27) aumentam o risco; pedidos "
                "especiais (−0,26), vaga de estacionamento (−0,20) e alterações na reserva (−0,18) diminuem. Entre as "
                "explicativas: reservas anteriores mantidas e hóspede recorrente andam juntas (+0,76); via agente e via "
                "empresa são quase excludentes (−0,59). Nenhuma correlação isolada é forte: o risco é multivariado.",
        por_que="Heatmap triangular com escala divergente centrada em zero mostra todas as relações de uma vez sem "
                "repetição; Spearman porque as variáveis são contagens assimétricas e as relações não são lineares.",
    ),
    "e4": dict(
        titulo="E4. Diária média (ADR) por mês e hotel",
        o_que="Boxplots da diária média (€) por mês de chegada, um para cada hotel (cortesias com ADR = 0 excluídas).",
        insight="O resort tem preço fortemente sazonal: mediana de ~48 € em janeiro e ~188 € em agosto. O hotel urbano é "
                "estável (80–120 €). A dispersão também cresce no verão do resort — época em que a receita perdida com "
                "cada cancelamento é muito maior.",
        por_que="Boxplot resume a distribuição (mediana, quartis, amplitude) de milhares de diárias por mês e permite "
                "comparar 24 grupos lado a lado; cores fixas por hotel mantêm a leitura consistente com E2.",
    ),
    "e5": dict(
        titulo="E5. Cancelamento por segmento e tipo de depósito",
        o_que="Taxa de cancelamento em cada combinação de segmento de mercado e tipo de depósito; o número de reservas "
              "(n) está em cada célula.",
        insight="Paradoxo: reservas 'Não reembolsáveis' cancelam 99%. Não é o hóspede pagando para cancelar — são lotes "
                "de quartos de Grupos e Operadoras offline (98% são linhas repetidas, quase todos de Portugal, sem "
                "nenhum pedido especial) liberados em bloco. Sem depósito, o maior risco é a Agência online (37%) e o "
                "menor, Direto/Corporativo (14–15%). Células com n pequeno (ex.: Reembolsável) não devem ser lidas como "
                "tendência.",
        por_que="Heatmap é a forma adequada para uma métrica em duas dimensões categóricas; a rampa de uma só cor (laranja "
                "= cancelamento) vai de 'pouco' a 'muito' sem sugerir sinal, e o n escrito evita superinterpretar "
                "células pequenas.",
    ),
    "e6": dict(
        titulo="E6. Sinais de relacionamento do hóspede",
        o_que="Taxa de cancelamento por número de pedidos especiais, por hóspede recorrente e por reserva via agente; a "
              "linha tracejada é a média geral.",
        insight="Qualquer pedido especial derruba o cancelamento de 48% para 22% (e 17% com 3 ou mais). Hóspedes "
                "recorrentes cancelam 15% contra 38%. Reservas via agente cancelam mais (39% vs. 25%). Todos apontam "
                "na mesma direção: quanto mais vínculo/planejamento, menor o risco.",
        por_que="Barras verticais comparam taxas entre poucas categorias; a mesma escala Y nos três painéis e a linha de "
                "média permitem ver rapidamente o que fica acima ou abaixo do normal.",
    ),
    # ----------------------------------------------------------- explanatórias
    "x1": dict(
        titulo="X1. Quanto mais cedo o hóspede reserva, maior a chance de cancelar",
        o_que="Taxa de cancelamento por faixa de antecedência. Laranja = faixas acima da média geral (37%); cinza = "
              "abaixo. RESPONDE DIRETAMENTE À PERGUNTA DE PESQUISA.",
        insight="O risco cresce de forma contínua com a antecedência: 10% (até 7 dias) → 28% → 38% → 45% → 55% → 68% "
                "(mais de 1 ano). Reservas com mais de 30 dias já ficam acima da média; acima de 6 meses, mais da metade "
                "é cancelada. O padrão se mantém em cada hotel e também removendo as linhas repetidas.",
        por_que="Barras ordenadas pela ordem natural da antecedência, com UMA cor de destaque e o resto em cinza: o "
                "gestor vê a escada sem precisar ler o eixo. Os percentuais escritos nas barras dispensam consulta à "
                "escala e a linha da média dá a referência.",
    ),
    "x2": dict(
        titulo="X2. Um pedido especial corta o cancelamento pela metade",
        o_que="Taxa de cancelamento por faixa de antecedência, separando reservas sem nenhum pedido especial (laranja, "
              "linha grossa) e com pelo menos um (cinza).",
        insight="Em todas as faixas, quem fez pedido especial cancela metade ou menos: 36% → 20% (8–30 dias), 58% → 26% "
                "(91–180 dias), 67% → 32% (181–365 dias). O pior cenário é antecedência longa SEM pedido (85% acima de "
                "1 ano). Os dois fatores da hipótese se somam.",
        por_que="Duas linhas no mesmo eixo mostram ao mesmo tempo o efeito da antecedência (inclinação) e o do pedido "
                "especial (distância entre as linhas). Rótulos diretos no fim das linhas substituem a legenda e o "
                "contraste cor/espessura dirige o olhar para a linha de risco.",
    ),
    "x3": dict(
        titulo="X3. 27% das reservas geram 47% dos cancelamentos",
        o_que="Participação de cada perfil (antecedência > ou ≤ 90 dias × com/sem pedido especial) no total de reservas "
              "(barra de cima) e no total de cancelamentos (barra de baixo).",
        insight="O perfil 'mais de 90 dias e nenhum pedido especial' é só 27% das reservas, mas responde por 47% dos "
                "cancelamentos (64% delas são canceladas). É ali que uma política de depósito/confirmação tem mais "
                "retorno: atinge pouco mais de 1/4 dos hóspedes e quase metade do problema.",
        por_que="Barras 100% empilhadas lado a lado comparam composição (parte do todo) entre duas bases; o segmento de "
                "risco vem primeiro, em laranja, alinhado à esquerda para que a diferença de comprimento salte aos "
                "olhos. Usamos barras e não pizza porque comprimentos alinhados são comparados com muito mais precisão "
                "do que ângulos.",
    ),
    # ----------------------------------------------------------------- modelo
    "m1": dict(
        titulo="Importância das variáveis no modelo de risco",
        o_que="Quanto a qualidade do modelo (AUC) cai quando os valores de cada variável são embaralhados no conjunto de "
              "teste (importância por permutação).",
        insight="Segmento de mercado, país de origem e tipo de depósito lideram (capturam os lotes de grupos e as "
                "reservas domésticas, que cancelam 57%); logo depois vêm pedidos especiais e antecedência — as duas "
                "variáveis da hipótese, confirmadas mesmo controlando por todas as outras.",
        por_que="Barras horizontais ordenadas são a forma mais legível para ranking com nomes longos; a barra de erro "
                "mostra a variação entre as repetições da permutação.",
    ),
    "m2": dict(
        titulo="Desempenho do modelo (teste em 2017)",
        o_que="Curva ROC e matriz de confusão do classificador, treinado com chegadas até 2016 e testado em 2017.",
        insight="AUC de 0,84: o modelo ordena bem o risco. Com limiar 0,5 acerta 77% (contra 61% de 'nunca cancela'), com "
                "precisão de 79% — quando aponta risco, geralmente acerta — mas recall de 57%: parte dos cancelamentos "
                "não é previsível só com os dados da reserva. Resposta à 2ª parte da pergunta: o risco é parcialmente "
                "previsível, suficiente para PRIORIZAR ações, não para decidir reserva a reserva.",
        por_que="A ROC mostra o equilíbrio entre acertos e alarmes falsos em todos os limiares; a matriz de confusão "
                "mostra, em contagens absolutas, os dois tipos de erro no limiar escolhido.",
    ),
    "explorador": dict(
        titulo="Explorador livre de variáveis",
        o_que="Relação entre quaisquer duas variáveis escolhidas pelo usuário; o tipo de gráfico e a medida de "
              "associação se ajustam aos tipos das variáveis.",
        insight="Permite verificar ao vivo qualquer relação do dataset (inclusive as não destacadas no relatório) com a "
                "medida adequada: Spearman (numérica x numérica), razão de correlação η² (categórica x numérica) ou "
                "V de Cramér (categórica x categórica).",
        por_que="Numérica x numérica: mapa de densidade (119 mil pontos virariam uma mancha num dispersão). Categórica x "
                "numérica: boxplots. Categórica x categórica: heatmap de % por linha. Alvo binário: taxa de "
                "cancelamento por categoria/faixa.",
    ),
}
