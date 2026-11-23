# Painel de Vigilância Sanitária e Epidemiológica no Abate Animal (Brasil, 2021–2026)

Aplicação web interativa desenvolvida em **Python** e **Streamlit** para monitoramento, análise epidemiológica e democratização dos dados oficiais de abate e condenação sanitária de **bovinos, bubalinos e suínos** no Brasil.

🔗 **Acesse o painel online:** [inspecao-abate.streamlit.app](https://inspecao-abate.streamlit.app)

---

## Objetivo e Relevância em Saúde Pública

Os registros oficiais de inspeção *ante mortem* e *post mortem* em abatedouros-frigoríficos constituem uma das principais fontes de vigilância epidemiológica passiva do país. Contudo, esses dados costumam estar dispersos em planilhas brutas de difícil interpretação para o público geral.

Este projeto transforma esses registros em informação acessível, transparente e tecnicamente plausível sob a ótica da **Inspeção de Produtos de Origem Animal (RIISPOA/MAPA)** e da **Saúde Pública Veterinária**, permitindo:
* **Transparência ao consumidor:** Demonstrar o rigor da fiscalização sanitária, evidenciando que carcaças, cortes e vísceras com alterações patológicas, parasitárias ou falhas tecnológicas de processo são desviados e inutilizados antes de chegarem ao consumo humano.
* **Apoio à vigilância epidemiológica:** Identificar padrões geográficos, sazonais e tendências temporais das principais causas de condenação por espécie e Unidade da Federação (UF).

---

## Estrutura e Funcionalidades do Dashboard

O sistema é dividido em dois módulos independentes — **Bovinos & Bubalinos** e **Suínos** — com filtros dinâmicos por **Região Geográfica, Unidade da Federação (UF), Ano e Espécie**, organizados em três abas analíticas:

### 1. Panorama Geral
* **Indicadores Consolidados (KPIs):** Volume total de animais abatidos, condenações de carcaças/cortes, condenações de vísceras/miúdos e total de ocorrências notificadas.
* **Mapa Coroplético Interativo por UF:** Visualização espacial alternável entre **Taxa Sanitária (%)** (ocorrências por 100 animais abatidos) e **Volume Absoluto**.
* **Top 10 Partes Afetadas:** Ranking dos órgãos e estruturas mais condenados na linha de inspeção (ex.: remoção preventiva de cérebro/MER em ruminantes; pulmões, rins e fígado em suínos).
* **Evolução Temporal e Tabela Consolidada:** Série histórica mensal e consulta tabular dos dados tratados.

### 2. Doenças e Condições
* **Proporção por Grupo Diagnóstico (Treemap):** Distribuição hierárquica das causas de condenação, com controle interativo para ocultar a categoria genérica *"Outro"* e ampliar a legibilidade das causas específicas.
* **Ranking de Achados Sanitários:** As 10 principais causas de condenação, diferenciando enfermidades infecciosas/parasitárias de falhas operacionais de abate e lesões de manejo pré-abate.
* **Investigação por Diagnóstico Específico:** Seletor individual para análise da distribuição estadual (UF) e comportamento temporal de qualquer patologia ou achado registrado.

### 3. Estatística e Modelagem
* **Regressão GLM Poisson (Razão de Taxa de Incidência — IRR):** Estimativa de fatores de risco e tendência temporal mensal ajustada pelo logaritmo do volume de abate (*offset*), com intervalos de confiança (IC 95%) e significância estatística ($p$-valor).
* **Variabilidade Regional (Boxplot):** Dispersão das taxas sanitárias entre os estados brasileiros.
* **Linha de Tendência Ponderada (OLS):** Regressão linear sobre a série histórica agregada.

---

## Rigor Metodológico e Qualidade dos Dados

Para garantir fidelidade científica e evitar distorções estatísticas comuns em bases administrativas alimentadas na ponta, o pipeline aplica critérios explícitos de higiene e transparência:

1. **Otimização ETL (`.csv` $\rightarrow$ `.parquet`):** Os arquivos brutos originais (superiores a 100 MB) são processados via notebooks Jupyter e convertidos em bases colunares compactas (`.parquet`) na pasta `dados/prod/`, assegurando carregamento rápido em nuvem.
2. **Filtro de Qualidade para Anomalias de Digitação:** Registros primários com inconsistências evidentes de preenchimento (taxas superiores a 200%) são automaticamente isolados e documentados no painel para preservar a escala dos mapas e modelos.
3. **Tratamento de Meses Incompletos:** Meses de extremidade com abate declarado, mas ausência total de notificações de condenação (como janeiro de 2021 e meses parciais de 2026), são filtrados das curvas de tendência e dos ajustes GLM/OLS para evitar viés de inclinação.
4. **Fidelidade aos Descritores Oficiais:** As nomenclaturas originais do sistema oficial são preservadas integralmente. Ocorrências eventuais de descritores atípicos entre espécies (como *aborto enzoótico das ovelhas* em bovinos por aproximação diagnóstica de *Chlamydia* spp., ou registros residuais de *MER* em suínos) são mantidas para rastreabilidade e documentadas na seção *Métodos, Fontes e Limitações*.

---

## Estrutura básica do Repositório

```text
Project-data-disease-animal/
├── dados/
│   └── prod/                         # Bases consolidadas leves (.parquet) utilizadas pelo app
├── Tratamento dos dados/             # Notebooks Jupyter de limpeza, padronização e ETL
│   ├── Analise_Condenacao.ipynb
│   ├── Analise_Condenacao_suino.ipynb
│   ├── analise_DP_bovino.ipynb
│   ├── analise_DP_suino.ipynb
│   ├── analise_abate.ipynb
│   └── analise_abate_suino.ipynb
├── app.py                            # Aplicação principal Streamlit (navegação e painéis)
├── requirements.txt                  # Dependências Python do projeto
├── .gitignore                        # Exclusão de CSVs brutos pesados e arquivos temporários
└── README.md                         # Documentação do projeto

Como Executar Localmente
Clone o repositório:

Bash
git clone [https://github.com/Naiasha/Project-data-disease-animal.git](https://github.com/Naiasha/Project-data-disease-animal.git)
cd Project-data-disease-animal
Crie e ative um ambiente virtual:

Bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
Instale as dependências:

Bash
pip install -r requirements.txt
Execute o dashboard:

Bash
streamlit run app.py