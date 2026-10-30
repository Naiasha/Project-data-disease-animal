from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import json
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go

st.set_page_config(page_title="Vigilância Sanitária - Abate e Condenação",
                   page_icon=":bar_chart:",
                   layout="wide")

def aplicar_estilo_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Tipografia Global */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* 1. Fundo Global da Aplicação (Azul Bem Claro) */
        .stApp {
            background-color: #EBF3FA !important;
        }

        /* 2. Estilização dos Cards de Métricas (st.metric) */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #D1E7DD !important;
            border-radius: 14px !important;
            padding: 16px 20px !important;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.12) !important;
            transition: all 0.25s ease-in-out !important;
            min-height: 110px !important; /* Altura mínima para acomodar 2 linhas de texto */
        }

        /* Força a remoção de ellipsis e permite quebra em TODOS os elementos filhos do rótulo e valor */
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            white-space: normal !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
            overflow: visible !important;
            text-overflow: unset !important;
        }

        /* Ajuste de tamanho do Rótulo (Título) */
        div[data-testid="stMetricLabel"] label {
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            color: #475569 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Ajuste do tamanho da fonte e espaçamento do Valor */
        div[data-testid="stMetricValue"] {
            font-size: 1.15rem !important; /* Reduzido levemente para caber diagnósticos longos */
            font-weight: 700 !important;
            color: #0F172A !important;
            line-height: 1.25 !important;
            margin-top: 4px !important;
        }

        /* 3. Containers de Gráficos e Blocos (st.container com borda) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid #D1E7DD !important;
            border-radius: 16px !important;
            padding: 16px !important;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.10) !important;
        }

        /* 4. Moldura Automática para Gráficos Plotly */
        div[data-testid="stPlotlyChart"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            padding: 12px !important;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.10) !important;
        }

        /* 5. Customização Elegante das Abas (st.tabs) */
        button[data-baseweb="tab"] {
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: #64748B !important;
            padding: 10px 20px !important;
            border-radius: 10px !important;
            background-color: transparent !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #0F172A !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 2px 10px rgba(16, 185, 129, 0.15) !important;
        }

        /* Permite quebra de linha e ajusta tamanho do RÓTULO (Título) */
        div[data-testid="stMetricLabel"] > label {
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            color: #475569 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: normal !important; /* Permite quebra de linha */
            word-wrap: break-word !important;
        }

        /* Permite quebra de linha e reduz a fonte do VALOR da Métrica */
        div[data-testid="stMetricValue"] > div {
            font-size: 1.25rem !important; /* Reduzido de 1.8rem para caber diagnósticos longos */
            font-weight: 700 !important;
            color: #0F172A !important;
            white-space: normal !important; /* Permite quebra de linha */
            word-break: break-word !important; /* Quebra palavras muito longas */
            line-height: 1.2 !important;
        }

        /* Permite quebra de linha e ajusta tamanho do RÓTULO (Título) */
        div[data-testid="stMetricLabel"] > label {
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            color: #475569 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: normal !important; /* Permite quebra de linha */
            word-wrap: break-word !important;
        }

        /* Permite quebra de linha e reduz a fonte do VALOR da Métrica */
        div[data-testid="stMetricValue"] > div {
            font-size: 1.25rem !important; /* Reduzido de 1.8rem para caber diagnósticos longos */
            font-weight: 700 !important;
            color: #0F172A !important;
            white-space: normal !important; /* Permite quebra de linha */
            word-break: break-word !important; /* Quebra palavras muito longas */
            line-height: 1.2 !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def formatar_grafico_plotly(fig):
    """Garante fundo branco e eixos harmonizados dentro do container do Plotly."""
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        font=dict(family="Inter, sans-serif", color="#334155"),
        margin=dict(l=15, r=15, t=35, b=15),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None
        )
    )
    return fig

aplicar_estilo_css()

if "cadeia_selecionada" not in st.session_state:
    st.session_state["cadeia_selecionada"] = None  # None = Tela Inicia

# Caminho relativo para o diretório de dados, ajustado para funcionar tanto em execução local quanto em ambiente de nuvem
BASE_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
DATA_DIR = BASE_DIR.parent / "Dados"

REGIOES = {
    "Brasil": None,
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Sul": ["PR", "RS", "SC"]
}

siglas_por_nome = {
    "Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM",
    "Bahia": "BA", "Ceará": "CE", "Distrito Federal": "DF",
    "Espírito Santo": "ES", "Goiás": "GO", "Maranhão": "MA",
    "Mato Grosso": "MT", "Mato Grosso do Sul": "MS", "Minas Gerais": "MG",
    "Pará": "PA", "Paraíba": "PB", "Paraná": "PR", "Pernambuco": "PE",
    "Piauí": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS", "Rondônia": "RO", "Roraima": "RR",
    "Santa Catarina": "SC", "São Paulo": "SP", "Sergipe": "SE",
    "Tocantins": "TO"
}

UF_PARA_REGIAO = {uf: reg for reg, ufs in REGIOES.items() if ufs for uf in ufs}

# funções utilitárias para padronização de nomes e formatação de números
# Formatação de números e percentuais


def formata_numero(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formata_percentual(valor: float) -> str:
    return f"{valor:.2f}%".replace(".", ",")

# Funções de categorização e padronização de partes e órgãos


def categorizar_parte(parte: str) -> str:
    if pd.isna(parte):
        return "Outros"

    p = str(parte).strip().lower()

    # 1. Isolamento de Coletas Fiscais / PNCRC
    if "coleta" in p or "urina" in p:
        return "Coletas / Amostragens"

    # 2. Carcaça e Partes Estruturais
    termos_carcaca = ["carcaça", "carcaca", "meia carcaça", "quarto", "paleta"]
    if any(t in p for t in termos_carcaca):
        return "Carcaça e Cortes"

    # 3. Demais Órgãos e Vísceras
    return "Vísceras e Miúdos"


def padronizar_nome_parte(parte: str) -> str:
    if pd.isna(parte):
        return "Não Especificado"

    p = str(parte).strip().title()
    correcoes = {
        "Pancreas": "Pâncreas",
        "Pés(Mocotó)": "Pés (Mocotó)",
        "Pés(Mocotó)": "Pés (Mocotó)",
        "Cauda(Rabo)": "Rabo / Cauda",
        "Rabo": "Rabo / Cauda",
        "Cérebro(Miolo)": "Cérebro (Miolo)"
    }
    return correcoes.get(p, p)


# camada de dados e cache para otimizar o carregamento
@st.cache_data(ttl=3600, show_spinner="Carregando dados das doenças...")
def carregar_dados_doencas() -> pd.DataFrame:
    caminho_doenca = DATA_DIR / "tratados" / \
        "df_bovino_bubalino_doenças_10.parquet"

    doenca = pd.read_parquet(caminho_doenca)

    # Padronização das chaves primárias
    doenca["UF_PROCEDENCIA"] = doenca["UF_PROCEDENCIA"].astype(
        "string").str.strip().str.upper()
    doenca["ANO"] = pd.to_numeric(doenca["ANO"], errors="coerce")
    doenca["MES"] = pd.to_numeric(doenca["MES"], errors="coerce")

    # Padronização da coluna de diagnósticos/doenças
    col_diag = next((c for c in ["DIAGNOSTICO", "DESCRICAO_DIAGNOSTICO",
                    "CAUSA"] if c in doenca.columns), doenca.columns[0])
    doenca["DIAGNOSTICO_PADRONIZADO"] = doenca[col_diag].astype(
        str).str.strip().str.title()

    # Mapeamento dinâmico da quantidade de registros/peças afetadas
    col_qtd = next((c for c in ["QUANTIDADE", "TOTAL_PARTES_AFETADAS",
                   "TOTAL", "CASOS", "NUMERO_ANIMAIS_ACOMETIDOS"] if c in doenca.columns), None)
    if col_qtd:
        doenca["QTD_OCORRENCIAS"] = pd.to_numeric(
            doenca[col_qtd], errors="coerce").fillna(1)
    else:
        # Caso a tabela possua 1 linha por ocorrência
        doenca["QTD_OCORRENCIAS"] = 1

    # Filtro de recorte temporal mantendo alinhamento com a base geral
    anos_analise = [2021, 2022, 2023, 2024, 2025, 2026]
    doenca = doenca[doenca["ANO"].isin(anos_analise)].copy()

    return doenca

@st.cache_data(ttl=3600, show_spinner="Carregando dados das doenças...")
def carregar_dados_doencas_suinos() -> pd.DataFrame:
    caminho_doenca_s = DATA_DIR / "tratados" / \
        "suino_doenças_final.parquet"

    doenca_suino = pd.read_parquet(caminho_doenca_s)

    # Padronização das chaves primárias
    doenca_suino["UF_PROCEDENCIA"] = doenca_suino["UF_PROCEDENCIA"].astype(
        "string").str.strip().str.upper()
    doenca_suino["ANO"] = pd.to_numeric(doenca_suino["ANO"], errors="coerce")
    doenca_suino["MES"] = pd.to_numeric(doenca_suino["MES"], errors="coerce")

    # Padronização da coluna de diagnósticos/doenças
    col_diag = next((c for c in ["DIAGNOSTICO", "DESCRICAO_DIAGNOSTICO",
                    "CAUSA"] if c in doenca_suino.columns), doenca_suino.columns[0])
    doenca_suino["DIAGNOSTICO_PADRONIZADO"] = doenca_suino[col_diag].astype(
        str).str.strip().str.title()

    # Mapeamento dinâmico da quantidade de registros/peças afetadas
    col_qtd = next((c for c in ["QUANTIDADE", "TOTAL_PARTES_AFETADAS",
                   "TOTAL", "CASOS", "NUMERO_ANIMAIS_ACOMETIDOS"] if c in doenca_suino.columns), None)
    if col_qtd:
        doenca_suino["QTD_OCORRENCIAS"] = pd.to_numeric(
            doenca_suino[col_qtd], errors="coerce").fillna(1)
    else:
        # Caso a tabela possua 1 linha por ocorrência
        doenca_suino["QTD_OCORRENCIAS"] = 1

    # Filtro de recorte temporal mantendo alinhamento com a base geral
    anos_analise = [2021, 2022, 2023, 2024, 2025, 2026]
    doenca_suino = doenca_suino[doenca_suino["ANO"].isin(anos_analise)].copy()

    return doenca_suino

@st.cache_data(ttl=3600, show_spinner="Carregando dados de abate e condenação...")
def carregar_dados() -> pd.DataFrame:
    caminho_abate = DATA_DIR / "tratados" / "bovino_bubalinos_abate_país.parquet"
    caminho_condenacao = DATA_DIR / "tratados" / \
        "bovino_bubalinos_condenacao_ano.parquet"

    if not caminho_abate.exists() or not caminho_condenacao.exists():
        st.error(
            f"❌ Não foi possível encontrar os arquivos em `{DATA_DIR / 'tratados'}`")
        st.stop()

    abate = pd.read_parquet(caminho_abate)
    condenacao = pd.read_parquet(caminho_condenacao)

    chaves = ["ANO", "MES", "UF_PROCEDENCIA", "ESPECIE"]

    # Padroniza os tipos de dados e remove espaços em branco
    for df in [abate, condenacao]:
        df["UF_PROCEDENCIA"] = df["UF_PROCEDENCIA"].astype(
            "string").str.strip().str.upper()
        df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce")
        df["MES"] = pd.to_numeric(df["MES"], errors="coerce")

# Identifica as colunas relevantes para abate e condenação, garantindo compatibilidade com diferentes versões do dataset
    col_abate = next((c for c in [
                     "QUANTIDADE", "ABATE", "TOTAL"] if c in abate.columns), abate.columns[-1])

    col_condenacao_qtd = next((c for c in [
                              "TOTAL_PARTES_AFETADAS", "QUANTIDADE", "CONDENADO"] if c in condenacao.columns), condenacao.columns[-1])

    col_parte = next((c for c in ["PARTE_AFETADA", "DESCRICAO_PARTE",
                     "PARTE"] if c in condenacao.columns), condenacao.columns[4])

    condenacao["PARTE_PADRONIZADA"] = condenacao[col_parte].apply(
        padronizar_nome_parte)
    condenacao["CATEGORIA_SANITY"] = condenacao[col_parte].apply(
        categorizar_parte)

    abate_consolidado = abate.groupby(
        chaves, as_index=False).agg(ABATE=("QUANTIDADE", "sum"))

# Agrupa condenações filtrando coletas fora do cálculo de lesões
    condenacao_lesoes = condenacao[condenacao["CATEGORIA_SANITY"]
                                   != "Coletas / Amostragens"].copy()

    condenacao_categoria = (
        condenacao_lesoes
        .groupby(chaves + ["CATEGORIA_SANITY"], as_index=False)
        .agg(CONDENADO=("TOTAL_PARTES_AFETADAS", "sum"))
    )

    # Pivotamento das categorias de condenação em colunas
    condenacao_pivot = condenacao_categoria.pivot_table(
        index=chaves,
        columns="CATEGORIA_SANITY",
        values="CONDENADO",
        fill_value=0
    ).reset_index()

    # Garantia de colunas no DataFrame
    for col in ["Carcaça e Cortes", "Vísceras e Miúdos"]:
        if col not in condenacao_pivot.columns:
            condenacao_pivot[col] = 0

    # Merge da base nacional com os abates
    dados = abate_consolidado.merge(condenacao_pivot, on=chaves, how="left")
    dados["Carcaça e Cortes"] = dados["Carcaça e Cortes"].fillna(0)
    dados["Vísceras e Miúdos"] = dados["Vísceras e Miúdos"].fillna(0)
    dados["CONDENADO_TOTAL"] = dados["Carcaça e Cortes"] + \
        dados["Vísceras e Miúdos"]

    anos_analise = [2021, 2022, 2023, 2024, 2025, 2026]
    dados = dados[dados["ANO"].isin(anos_analise)].copy()

    dados["DATA"] = pd.to_datetime(
        dict(year=dados["ANO"], month=dados["MES"], day=1),
        errors="coerce"
    )

    # Uma carcaça de bovino tem no máximo cerca de 10 a 15 partes notificáveis.
    # Registros com taxa > 200% (mais de 2 partes/carcaças por animal) são anomalias.

    # Taxas por 100 abates
    dados["taxa_carcaca"] = (dados["Carcaça e Cortes"] / dados["ABATE"]) * 100
    dados["taxa_visceras"] = (
        dados["Vísceras e Miúdos"] / dados["ABATE"]) * 100
    dados["percentual_condenados"] = (
        dados["CONDENADO_TOTAL"] / dados["ABATE"]) * 100
    # Criamos uma flag para identificar anomalias sem perder o dado bruto
    dados["ANOMALIA_REGISTRO"] = dados["percentual_condenados"] > 200.0
    return dados, condenacao

@st.cache_data(ttl=3600, show_spinner="Carregando dados de suínos...")
def carregar_dados_suinos() -> tuple[pd.DataFrame, pd.DataFrame]:
    caminho_abate_suino = DATA_DIR / "tratados" / "suino_abate.parquet"
    caminho_condenacao_suino = DATA_DIR / "tratados" / "suino_ano.parquet"

    if not caminho_abate_suino.exists() or not caminho_condenacao_suino.exists():
        st.error(f"❌ Não foi possível encontrar os arquivos de suínos em `{DATA_DIR / 'tratados'}`")
        st.stop()

    abate_suino = pd.read_parquet(caminho_abate_suino)
    condenacao_suino = pd.read_parquet(caminho_condenacao_suino)

    chaves = ["ANO", "MES", "UF_PROCEDENCIA", "ESPECIE"]

    # Padroniza os tipos de dados e remove espaços em branco
    for df in [abate_suino, condenacao_suino]:
        df["UF_PROCEDENCIA"] = df["UF_PROCEDENCIA"].astype(
            "string").str.strip().str.upper()
        df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce")
        df["MES"] = pd.to_numeric(df["MES"], errors="coerce")

    col_abate_suino = next((c for c in ["QUANTIDADE", "ABATE", "TOTAL"] if c in abate_suino.columns), abate_suino.columns[-1])
    col_condenacao_suino_qtd = next((c for c in ["TOTAL_PARTES_AFETADAS", "QUANTIDADE", "CONDENADO"] if c in condenacao_suino.columns), condenacao_suino.columns[-1])
    col_parte_suino = next((c for c in ["PARTE_AFETADA", "DESCRICAO_PARTE", "PARTE"] if c in condenacao_suino.columns), condenacao_suino.columns[4])

    condenacao_suino["PARTE_PADRONIZADA"] = condenacao_suino[col_parte_suino].apply(padronizar_nome_parte)
    condenacao_suino["CATEGORIA_SANITY"] = condenacao_suino[col_parte_suino].apply(categorizar_parte)

    abate_consolidado_suino = abate_suino.groupby(chaves, as_index=False).agg(ABATE=(col_abate_suino, "sum"))

    condenacao_lesoes_suino = condenacao_suino[condenacao_suino["CATEGORIA_SANITY"] != "Coletas / Amostragens"].copy()

    condenacao_categoria_suino = (
        condenacao_lesoes_suino
        .groupby(chaves + ["CATEGORIA_SANITY"], as_index=False)
        .agg(CONDENADO=(col_condenacao_suino_qtd, "sum"))
    )

    condenacao_suino_pivot = condenacao_categoria_suino.pivot_table(
        index=chaves,
        columns="CATEGORIA_SANITY",
        values="CONDENADO",
        fill_value=0
    ).reset_index()

    for col in ["Carcaça e Cortes", "Vísceras e Miúdos"]:
        if col not in condenacao_suino_pivot.columns:
            condenacao_suino_pivot[col] = 0

    # Merge correto da base de abates com a pivot de condenações de suínos
    dados_suino = abate_consolidado_suino.merge(condenacao_suino_pivot, on=chaves, how="left")
    dados_suino["Carcaça e Cortes"] = dados_suino["Carcaça e Cortes"].fillna(0)
    dados_suino["Vísceras e Miúdos"] = dados_suino["Vísceras e Miúdos"].fillna(0)
    dados_suino["CONDENADO_TOTAL"] = dados_suino["Carcaça e Cortes"] + dados_suino["Vísceras e Miúdos"]

    anos_analise = [2021, 2022, 2023, 2024, 2025, 2026]
    dados_suino = dados_suino[dados_suino["ANO"].isin(anos_analise)].copy()

    dados_suino["DATA"] = pd.to_datetime(
        dict(year=dados_suino["ANO"], month=dados_suino["MES"], day=1),
        errors="coerce"
    )

    dados_suino["taxa_carcaca"] = (dados_suino["Carcaça e Cortes"] / dados_suino["ABATE"]) * 100
    dados_suino["taxa_visceras"] = (dados_suino["Vísceras e Miúdos"] / dados_suino["ABATE"]) * 100
    dados_suino["percentual_condenados"] = (dados_suino["CONDENADO_TOTAL"] / dados_suino["ABATE"]) * 100
    dados_suino["ANOMALIA_REGISTRO"] = dados_suino["percentual_condenados"] > 200.0

    return dados_suino, condenacao_suino

@st.cache_data(ttl=3600, show_spinner="Carregando malha geográfica...")
def carregar_malha_estados() -> dict:
    caminho_malha = DATA_DIR / "malha_estados.json"

    with open(caminho_malha, "r", encoding="utf-8") as f:
        malha_estados = json.load(f)

    for feature in malha_estados["features"]:
        props = feature.setdefault("properties", {})
        if "UF_SIGLA" in props and props["UF_SIGLA"]:
            props["UF_SIGLA"] = str(props["UF_SIGLA"]).strip().upper()

    return malha_estados

# tela inicial


def renderizar_tela_inicial():
    st.markdown("<br>", unsafe_allow_html=True)

    # Cabeçalho Principal
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 40px;'>
            <h1 style='font-size: 2.5rem; font-weight: 800; color: #0F172A; margin-bottom: 8px;'>
                Portal de Vigilância Sanitária — SIGSIF
            </h1>
            <p style='font-size: 1.1rem; color: #64748B; max-width: 700px; margin: 0 auto;'>
                Plataforma integrada de inteligência sanitária para análise epidemiológica de abates, 
                condenações e patologias em frigoríficos sob inspeção federal.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("### 🐮 Bovinos & Bubalinos")
            st.caption("Cadeia de Ruminantes")
            st.markdown(
                """
                • **Painel de Abates e Condenações:** Acompanhamento de perdas estruturais (carcaça) e parciais (vísceras).\n
                • **Análise Geográfica:** Mapeamento de risco sanitário por UF e região.\n
                • **Módulo doenças:** Vigilância e diagnóstico de enfermidades específicas.
                """
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Acessar Painel de Ruminantes", use_container_width=True, type="primary"):
                st.session_state["cadeia_selecionada"] = "bovinos"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### 🐷 Suínos")
            st.caption("Cadeia de Suinocultura")
            st.markdown(
                """
                • **Monitoramento de Lesões de Abate:** Abscessos, pneumonia, cisticercose suína e contaminações.\n
                • **Indicadores Industriais:** Taxas de aproveitamento e descarte em frigoríficos.\n
                • **Rastreabilidade por Procedência:** Mapeamento regional de riscos sanitários.
                """
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Acessar Painel de Suínos", use_container_width=True):
                st.session_state["cadeia_selecionada"] = "suinos"
                st.rerun()

# botão de voltar para a tela inicial


def exibir_botao_voltar():
    st.sidebar.markdown("Navegação")
    if st.sidebar.button("Voltar para a Tela Inicial", use_container_width=True):
        st.session_state["cadeia_selecionada"] = None
        st.rerun()
    st.sidebar.markdown("---")

# Dashboard de bovinos e bubalinos

def renderizar_dashboard_bovinos():
    exibir_botao_voltar()
    st.title("Dashboard Abate e Condenação — Bovinos & Bubalinos")

    # filtros dinâmicos e interativos com Streamlit
    dados_pais, df_condenacao_bruta = carregar_dados()
    geo_estados = carregar_malha_estados()
    df_doenca_bruto = carregar_dados_doencas()

    st.sidebar.header("Filtros de Analise")
    recorte_regiao = st.sidebar.radio(
        "Região Geográfica", list(REGIOES.keys()))

    if REGIOES[recorte_regiao] is not None:
        dados_base = dados_pais[dados_pais["UF_PROCEDENCIA"].isin(
            REGIOES[recorte_regiao])].copy()
    else:
        dados_base = dados_pais.copy()

    anos_disponiveis = sorted(dados_base["ANO"].dropna().unique())
    ufs_disponiveis = sorted(dados_base["UF_PROCEDENCIA"].dropna().unique())
    especies_disponiveis = sorted(dados_base["ESPECIE"].dropna().unique())

    anos = st.sidebar.multiselect(
        "Ano", options=anos_disponiveis, default=anos_disponiveis)
    ufs = st.sidebar.multiselect(
        "Unidade Federativa", options=ufs_disponiveis, default=ufs_disponiveis)
    especie = st.sidebar.multiselect(
        "Espécie", options=especies_disponiveis, default=especies_disponiveis)

    st.sidebar.markdown("---")
    st.sidebar.header("Qualidade e Sanitização")
    remover_outliers = st.sidebar.checkbox(
        "Remover Picos Atípicos (> 200%)",
        value=True,
        help="Filtra erros de digitação e registros inconsistentes do SIGSIF (ex: SC 05/2024) para evitar a distorção da escala do mapa e dos gráficos."
    )

    # Filtragem da base de dados com base nos filtros selecionados
    dados_filtrados = dados_base[
        dados_base["ANO"].isin(anos) &
        dados_base["UF_PROCEDENCIA"].isin(ufs) &
        dados_base["ESPECIE"].isin(especie)
    ].copy()

    # Filtro de doenças e diagnósticos
    if not df_doenca_bruto.empty:
        df_doencas_filtrado = df_doenca_bruto[
            (df_doenca_bruto["ANO"].isin(anos)) &
            (df_doenca_bruto["UF_PROCEDENCIA"].isin(ufs)) &
            (df_doenca_bruto["ESPECIE"].isin(especie))
        ].copy()

    # Se o filtro regional estiver ativo (ex: Sudeste, Sul, etc.)
        if REGIOES[recorte_regiao] is not None:
            df_doencas_filtrado = df_doencas_filtrado[
                df_doencas_filtrado["UF_PROCEDENCIA"].isin(
                    REGIOES[recorte_regiao])
            ].copy()

        dados_outliers = dados_filtrados[dados_filtrados["percentual_condenados"] > 200.0].copy(
        )

    else:
        # DataFrame vazio se não houver dados de doenças
        df_doencas_filtrado = pd.DataFrame()

    dados_outliers = dados_filtrados[dados_filtrados["percentual_condenados"] > 200.0].copy(
    )

    if remover_outliers:
        dados_filtrados = dados_filtrados[dados_filtrados["percentual_condenados"] <= 200.0].copy(
        )

    # divisão das abas
    tab_geral, tab_doencas, tab_estatistica = st.tabs([
        "Panorama Geral",
        "Doenças e Condições",
        "Estatística e Modelagem"
    ])

    # Aba "Panorama Geral" - KPIs, Mapa e Gráficos
    # KPIS e métricas principais
    with tab_geral:
        st.subheader("Indicadores Gerais de Abate e Condenação")
        st.caption(
            "Painel interativo de vigilância sanitária baseado nos dados de abate e condenação")

        if df_doencas_filtrado.empty:
            st.warning(
                "Nenhum registro de doença/diagnóstico encontrado para os filtros selecionados.")
        else:
            if not dados_outliers.empty and remover_outliers:
                st.warning(
                    f"🧹 **Filtro de Qualidade Ativo:** {len(dados_outliers)} registro(s) com inconsistência de digitação/taxa > 200% foi(foram) ocultado(s) "
                    f"para preservar a escala do mapa e dos gráficos. *(Ex: SC em 05/2024)*."
                )

            total_abatidos = dados_filtrados["ABATE"].sum()
            total_carcaca = dados_filtrados["Carcaça e Cortes"].sum()
            total_visceras = dados_filtrados["Vísceras e Miúdos"].sum()

            taxa_carcaca = (total_carcaca / total_abatidos *
                            100) if total_abatidos > 0 else 0
            taxa_visceras = (total_visceras / total_abatidos *
                             100) if total_abatidos > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Volume de Abate", formata_numero(total_abatidos))
            c2.metric("Condenações de Carcaça", formata_numero(total_carcaca),
                      delta=f"{taxa_carcaca:.2f}% dos abates", delta_color="inverse")
            c3.metric("Condenações de Vísceras/Miúdos", formata_numero(total_visceras),
                      delta=f"{taxa_visceras:.2f}% dos abates", delta_color="off")
            c4.metric("Registros de Ocorrências", formata_numero(
                total_carcaca + total_visceras))

            st.info(
                "💡 **Nota Técnica Sanitária:** O índice exibido representa o número de **ocorrências notificadas por 100 animais abatidos**. "
                "A categoria *Carcaça e Cortes* refere-se a condenações totais/estruturais, enquanto *Vísceras/Miúdos* refere-se a condenações parciais de órgãos."
            )

            st.markdown("---")

            # mapa interativo com Plotly Express
            st.subheader(
                f"Indicadores Geográficos por UF — Anos: {', '.join(map(str, anos))}")

            col_map_fig, col_map_ctrl = st.columns([3.5, 1])

            with col_map_ctrl:
                st.markdown("##### Métrica do Mapa")
                metrica_mapa = st.radio(
                    "Visualizar por:",
                    options=["Taxa Sanitária (%)",
                             "Volume Absoluto (Ocorrências)"],
                    index=0,
                    help=(
                        "• Taxa Sanitária (%): Ocorrências a cada 100 abates (relevância epidemiológica).\n"
                        "• Volume Absoluto: Total de condenações notificadas (escala operacional)."
                    )
                )

            resumo_mapa = (
                dados_filtrados
                .groupby("UF_PROCEDENCIA", as_index=False)
                .agg(
                    ABATE=("ABATE", "sum"),
                    CONDENADO_CARCACA=("Carcaça e Cortes", "sum"),
                    CONDENADO_VISCERAS=("Vísceras e Miúdos", "sum"),
                    CONDENADO_TOTAL=("CONDENADO_TOTAL", "sum")
                )
            )
            resumo_mapa["INDICE_CONDENACAO"] = (
                resumo_mapa["CONDENADO_TOTAL"] / resumo_mapa["ABATE"]) * 100
            resumo_mapa = resumo_mapa[resumo_mapa["ABATE"] > 0].copy()

            # Configuração dinâmica das variáveis do Plotly
            if metrica_mapa == "Taxa Sanitária (%)":
                coluna_cor = "INDICE_CONDENACAO"
                rotulo_cor = "Taxa (%)"
                hover_fmt = "<b>%{location}</b><br>Taxa Sanitária: %{z:.2f}%<br>Abates: %{customdata[0]:,.0f}<extra></extra>"
            else:
                coluna_cor = "CONDENADO_TOTAL"
                rotulo_cor = "Ocorrências"
                hover_fmt = "<b>%{location}</b><br>Ocorrências: %{z:,.0f}<br>Abates: %{customdata[0]:,.0f}<extra></extra>"

            fig_mapa = px.choropleth(
                resumo_mapa,
                geojson=geo_estados,
                locations="UF_PROCEDENCIA",
                featureidkey="properties.UF_SIGLA",
                color=coluna_cor,
                custom_data=["ABATE", "CONDENADO_TOTAL", "INDICE_CONDENACAO"],
                color_continuous_scale="YlOrRd",
                labels={coluna_cor: rotulo_cor},
            )

            fig_mapa.update_geos(fitbounds="locations", visible=False,
                                 projection_type="mercator")
            fig_mapa.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                   coloraxis_colorbar_title=rotulo_cor)
            fig_mapa.update_traces(hovertemplate=hover_fmt)

            with col_map_fig:
                evento = st.plotly_chart(
                    fig_mapa,
                    use_container_width=True,
                    key="mapa_brasil",
                    on_select="rerun",
                    selection_mode="points",
                    config={"displayModeBar": False}
                )

            pontos_selecionados = evento.get("selection", {}).get("points", [])
            if pontos_selecionados:
                uf_selecionada = pontos_selecionados[-1]["location"]
                df_uf = resumo_mapa[resumo_mapa["UF_PROCEDENCIA"]
                                    == uf_selecionada]

                if not df_uf.empty:
                    detalhe_uf = df_uf.iloc[0]
                    st.info(
                        f"**Detalhes selecionados — {uf_selecionada}:** "
                        f"Abates: {formata_numero(detalhe_uf['ABATE'])} | "
                        f"Carcaças Condenadas: {formata_numero(detalhe_uf['CONDENADO_CARCACA'])} | "
                        f"Vísceras Condenadas: {formata_numero(detalhe_uf['CONDENADO_VISCERAS'])} | "
                        f"Taxa Sanitária Global: {formata_percentual(detalhe_uf['INDICE_CONDENACAO'])}"
                    )
            else:
                st.caption(
                    "💡 Clique em um estado no mapa para ver o detalhamento isolado.")

            # gráfico de ranking de condenações por UF

            # Filtra a base bruta com base nos anos e UFs selecionados
            col_graf1, col_graf2 = st.columns([1, 1])

            with col_graf1:
                st.subheader("Top 10 Partes Afetadas por Condenação")

                df_partes_top = (
                    df_condenacao_bruta[
                        (df_condenacao_bruta["ANO"].isin(anos)) &
                        (df_condenacao_bruta["UF_PROCEDENCIA"].isin(ufs)) &
                        (df_condenacao_bruta["CATEGORIA_SANITY"]
                         != "Coletas / Amostragens")
                    ]
                    .groupby("PARTE_PADRONIZADA", as_index=False)
                    .agg(TOTAL=("TOTAL_PARTES_AFETADAS", "sum"))
                    .sort_values("TOTAL", ascending=True)
                    .tail(10)
                )

                fig_partes = px.bar(
                    df_partes_top,
                    x="TOTAL",
                    y="PARTE_PADRONIZADA",
                    orientation="h",
                    text_auto=",.0f",
                    labels={"TOTAL": "Registros",
                            "PARTE_PADRONIZADA": "Parte Afetada"},
                    color="TOTAL",
                    color_continuous_scale="Reds",
                    template="plotly_white"
                )
                fig_partes.update_layout(coloraxis_showscale=False,
                                         margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_partes, use_container_width=True)

                with col_graf2:
                    st.subheader("📈 Evolução Temporal da Taxa Sanitária")
                    # 1. Base nacional respeitando o filtro de qualidade/outliers
                    if remover_outliers:
                        dados_pais_base = dados_pais[dados_pais["percentual_condenados"] <= 200.0].copy()
                    else:
                        dados_pais_base = dados_pais.copy()

                    # 2. Agrupamento da Média Nacional Sanitizada
                    df_br_tempo = (
                        dados_pais_base
                        .groupby("DATA", as_index=False)
                        .agg(ABATE=("ABATE", "sum"), CONDENADO=("CONDENADO_TOTAL", "sum"))
                    )
                    df_br_tempo["taxa_brasil"] = (df_br_tempo["CONDENADO"] / df_br_tempo["ABATE"]) * 100

                    # 3. Agrupamento do Recorte Selecionado
                    df_sel_tempo = (
                        dados_filtrados
                        .groupby("DATA", as_index=False)
                        .agg(ABATE=("ABATE", "sum"), CONDENADO=("CONDENADO_TOTAL", "sum"))
                    )
                    df_sel_tempo["taxa_selecao"] = (df_sel_tempo["CONDENADO"] / df_sel_tempo["ABATE"]) * 100

                    # 4. Verifica se o usuário filtrou algo específico (Estado, Região ou Espécie)
                    is_filtro_especifico = (
                        (recorte_regiao != "Brasil") or 
                        (len(ufs) < len(ufs_disponiveis)) or 
                        (len(especie) < len(especies_disponiveis))
                    )

                    fig_linha = go.Figure()

                    if is_filtro_especifico:
                        # Mostra o Benchmark Nacional (pontilhado) + Recorte Selecionado (sólido)
                        fig_linha.add_trace(go.Scatter(
                            x=df_br_tempo["DATA"],
                            y=df_br_tempo["taxa_brasil"],
                            mode="lines",
                            name="Média Brasil (Benchmark)",
                            line=dict(color="#94A3B8", width=2, dash="dash"),
                            hovertemplate="Brasil: %{y:.2f}%<extra></extra>"
                        ))

                        if len(ufs) == 1:
                            nome_trace = f"UF: {ufs[0]}"
                        elif recorte_regiao != "Brasil":
                            nome_trace = f"Região: {recorte_regiao}"
                        else:
                            nome_trace = "Filtro Selecionado"

                        fig_linha.add_trace(go.Scatter(
                            x=df_sel_tempo["DATA"],
                            y=df_sel_tempo["taxa_selecao"],
                            mode="lines+markers",
                            name=nome_trace,
                            line=dict(color="#0284C7", width=3),
                            marker=dict(size=6),
                            hovertemplate=f"{nome_trace}: %{{y:.2f}}%<extra></extra>"
                        ))
                    else:
                        # Se a visão for o Brasil como um todo, exibe apenas UMA linha limpa
                        fig_linha.add_trace(go.Scatter(
                            x=df_br_tempo["DATA"],
                            y=df_br_tempo["taxa_brasil"],
                            mode="lines+markers",
                            name="Taxa Sanitária Nacional (Brasil)",
                            line=dict(color="#0284C7", width=3),
                            marker=dict(size=6),
                            hovertemplate="Brasil: %{y:.2f}%<extra></extra>"
                        ))

                    fig_linha = formatar_grafico_plotly(fig_linha)
                    fig_linha.update_layout(
                        yaxis_title="Taxa Sanitária (%)",
                        xaxis_title="Mês/Ano",
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_linha, use_container_width=True)

            UF_PARA_REGIAO = {uf: reg for reg,
                              ufs in REGIOES.items() if ufs for uf in ufs}
            
            if not dados_outliers.empty:
                with st.expander("Ver registros atípicos isolados pelo filtro de qualidade"):
                    st.write(
                        "Abaixo estão os registros originais da base que apresentaram inconsistências de proporção:")
                    st.dataframe(
                        dados_outliers[["ANO", "MES", "UF_PROCEDENCIA", "ESPECIE",
                                        "ABATE", "CONDENADO_TOTAL", "percentual_condenados"]],
                        use_container_width=True,
                        hide_index=True
                    )
            # tabela de dados consolidados
            st.subheader("Tabela de Dados Consolidados")

            colunas_tabela = [
                "ANO", "MES", "UF_PROCEDENCIA", "ESPECIE",
                "ABATE", "Carcaça e Cortes", "Vísceras e Miúdos",
                "taxa_carcaca", "taxa_visceras", "percentual_condenados"
            ]

            st.dataframe(
                dados_filtrados[colunas_tabela].sort_values(
                    ["ANO", "MES", "UF_PROCEDENCIA"]),
                use_container_width=True,
                hide_index=True
            )

            with st.expander("Métodos, Fontes e Limitações"):
                st.markdown(
                    """
                    **Fonte dos dados:** Sistema de Informações Gerenciais do Serviço de Inspeção Federal (SIGSIF).

                    **Metodologia de Agrupamento:**
                    * **Carcaça e Cortes:** Inclui registros em carcaças inteiras, meias carcaças, quartos e paletas.
                    * **Vísceras e Miúdos:** Inclui lesões registradas em órgãos e vísceras (fígado, pulmão, rins, etc.).
                    * **Amostragens/Coletas:** Registros de coleta fiscal (PNCRC) foram excluídos da contagem de lesões sanitárias.

                    **Fórmula:** $\\text{Taxa (\\%)} = \\left( \\frac{\\sum \\text{Registros de Condenções}}{\\sum \\text{Animais Abatidos}} \\right) \\times 100$
                    """
                )

    # aba 2 doenças e condições
    with tab_doencas:
        st.subheader("Análise Epidemiológica de Doenças e Diagnósticos")
        st.caption(
            "Painel interativo de vigilância sanitária baseado nos dados do arquivo Doença por procedência, permitindo uma análise detalhada das ocorrências de doenças e diagnósticos.")

        if df_doencas_filtrado.empty:
            st.info(
                "⚠️ Nenhum registro de doença/diagnóstico encontrado para os filtros selecionados.")
        else:
            # 1. KPIs da Aba 2
            col_d1, col_d2, col_d3 = st.columns(3)

            total_casos = df_doencas_filtrado["QTD_OCORRENCIAS"].sum()

            agrupado_diag = (
                df_doencas_filtrado
                .groupby("DIAGNOSTICO_PADRONIZADO")["QTD_OCORRENCIAS"]
                .sum()
            )

            top_diag_nome = agrupado_diag.idxmax()
            total_top_diag = agrupado_diag.max()
            pct_top_diag = (total_top_diag / total_casos *
                            100) if total_casos > 0 else 0

            col_d1.metric("Registros de Doenças", formata_numero(total_casos))
            col_d2.metric("Principal Diagnóstico", str(top_diag_nome))
            col_d3.metric("Prevalência do Top 1",
                          f"{pct_top_diag:.1f}% dos casos")

            st.markdown("---")

            # 2. Ranking e Distribuição
            col_c1, col_c2 = st.columns([1.8, 1.2])

            with col_c1:
                st.markdown("Treemap de Proporção dos Diagnósticos")
                df_tree = (
                    df_doencas_filtrado
                    .groupby("CLASSIFICACAO", as_index=False)
                    .agg(TOTAL=("QTD_OCORRENCIAS", "sum"))
                    .sort_values("TOTAL", ascending=False)
                    .head(15)  # Top 15 para não poluir o gráfico
                )

                fig_tree = px.treemap(
                    df_tree,
                    path=["CLASSIFICACAO"],
                    values="TOTAL",
                    color="TOTAL",
                    color_continuous_scale="Reds",
                    template="plotly_white"
                )
                fig_tree.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_tree, use_container_width=True)

            with col_c2:
                st.markdown("Top 10 Enfermidades com Maior Incidência")
                df_top_diag = (
                    df_doencas_filtrado
                    .groupby("DIAGNOSTICO_PADRONIZADO", as_index=False)
                    .agg(TOTAL=("QTD_OCORRENCIAS", "sum"))
                    .sort_values("TOTAL", ascending=True)
                    .tail(10)
                )

                fig_bar_diag = px.bar(
                    df_top_diag,
                    x="TOTAL",
                    y="DIAGNOSTICO_PADRONIZADO",
                    orientation="h",
                    text_auto=",.0f",
                    labels={"TOTAL": "Ocorrências",
                            "DIAGNOSTICO_PADRONIZADO": "Diagnóstico"},
                    color="TOTAL",
                    color_continuous_scale="OrRd",
                    template="plotly_white"
                )
                fig_bar_diag.update_layout(
                    coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_bar_diag, use_container_width=True)

            st.markdown("---")

            # 3. Análise Isolada de uma Enfermidade
            st.markdown("Investigação por Doença Específica")

            lista_diagnosticos = sorted(
                df_doencas_filtrado["DIAGNOSTICO_PADRONIZADO"].unique())
            diag_selecionado = st.selectbox(
                "Selecione um diagnóstico para detalhar:", lista_diagnosticos)

            df_diag_isolado = df_doencas_filtrado[df_doencas_filtrado["DIAGNOSTICO_PADRONIZADO"]
                                                  == diag_selecionado]

            col_det1, col_det2 = st.columns(2)

            with col_det1:
                st.markdown(
                    f"**Distribuição por Estado (UF) — {diag_selecionado}:**")
                df_uf_diag = (
                    df_diag_isolado
                    .groupby("UF_PROCEDENCIA", as_index=False)
                    .agg(TOTAL=("QTD_OCORRENCIAS", "sum"))
                    .sort_values("TOTAL", ascending=False)
                    .head(8)
                )
                fig_uf_diag = px.bar(
                    df_uf_diag,
                    x="UF_PROCEDENCIA",
                    y="TOTAL",
                    text_auto=",.0f",
                    labels={"TOTAL": "Casos", "UF_PROCEDENCIA": "UF"},
                    color_discrete_sequence=["#B22222"]
                )
                fig_uf_diag.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_uf_diag, use_container_width=True)

            with col_det2:
                st.markdown(
                    f"**Evolução Temporal dos Casos — {diag_selecionado}:**")

                df_diag_isolado["DATA"] = pd.to_datetime(
                    dict(year=df_diag_isolado["ANO"],
                         month=df_diag_isolado["MES"], day=1),
                    errors="coerce"
                )

                df_tempo_diag = (
                    df_diag_isolado
                    .groupby("DATA", as_index=False)
                    .agg(TOTAL=("QTD_OCORRENCIAS", "sum"))
                    .sort_values("DATA")
                )

                fig_tempo_diag = px.line(
                    df_tempo_diag,
                    x="DATA",
                    y="TOTAL",
                    markers=True,
                    labels={"DATA": "Mês/Ano", "TOTAL": "Casos Notificados"},
                    template="plotly_white"
                )
                fig_tempo_diag.update_traces(line_color="#CD5C5C")
                fig_tempo_diag.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_tempo_diag, use_container_width=True)

    # tab 3 - estatística e modelagem

    with tab_estatistica:
        st.subheader("Modelagem Estatística e Riscos Epidemiológicos")
        st.caption(
            "Aplicações estatísticas avançadas para estimativa de fatores de risco (GLM Poisson) e análise de variabilidade.")

        if dados_filtrados.empty or len(dados_filtrados) < 5:
            st.warning(
                "⚠️ Dados insuficientes para ajuste dos modelos estatísticos no recorte selecionado.")
        else:
            # ---------------------------------------------------------------------
            # 1. MODELO GLM POISSON (RAZÃO DE TAXA DE INCIDÊNCIA - IRR)
            # ---------------------------------------------------------------------
            st.markdown(
                "1. Estimativa de Fatores de Risco (GLM Poisson)")

            # Preparação da base para regressão
            df_model = dados_filtrados[dados_filtrados["ABATE"] > 0].copy()

            # Variável de tempo contínuo (em meses desde o início da série)
            min_ano = df_model["ANO"].min()
            df_model["TEMPO_MES"] = (
                df_model["ANO"] - min_ano) * 12 + df_model["MES"]
            df_model["LOG_ABATE"] = np.log(df_model["ABATE"])

            # Ajuste defensivo do modelo GLM via statsmodels
            try:
                # Ajuste de GLM Poisson com offset
                fórmula = "CONDENADO_TOTAL ~ TEMPO_MES"
                if df_model["ESPECIE"].nunique() > 1:
                    fórmula += " + C(ESPECIE)"

                modelo_glm = smf.glm(
                    formula=fórmula,
                    data=df_model,
                    family=sm.families.Poisson(),
                    offset=df_model["LOG_ABATE"]
                ).fit()

                # Extração dos coeficientes e conversão para IRR (e^beta)
                coef_df = pd.DataFrame({
                    "Coeficiente (β)": modelo_glm.params,
                    "IRR (e^β)": np.exp(modelo_glm.params),
                    "Erro Padrão": modelo_glm.bse,
                    "p-valor": modelo_glm.pvalues,
                    "IC 95% Inferior": np.exp(modelo_glm.conf_int()[0]),
                    "IC 95% Superior": np.exp(modelo_glm.conf_int()[1])
                })

                col_m1, col_m2 = st.columns([1.5, 1])

                with col_m1:
                    st.markdown(
                        "**Tabela de Estimadores e Risco Relativo (IRR):**")

                    # Formatando a tabela para exibição amigável
                    coef_formatado = coef_df.copy()
                    coef_formatado["IRR (e^β)"] = coef_formatado["IRR (e^β)"].map(
                        "{:.4f}".format)
                    coef_formatado["p-valor"] = coef_formatado["p-valor"].map(
                        lambda x: "< 0.001" if x < 0.001 else f"{x:.4f}")
                    coef_formatado["IC 95%"] = coef_formatado.apply(
                        lambda r: f"[{r['IC 95% Inferior']:.3f} ; {r['IC 95% Superior']:.3f}]", axis=1)

                    rotulos_variaveis = {
                        "Intercept": "Constante (Taxa Base)",
                        "C(ESPECIE)[T.Bubalino]": "Fator Espécie: Bubalino vs. Bovino",
                        "TEMPO_MES": "Tendência Temporal (Variação Mensal)"
                    }

                    # Aplica a renomeação dos índices
                    coef_formatado.index = coef_formatado.index.map(
                        lambda x: rotulos_variaveis.get(x, x))

                    st.dataframe(
                        coef_formatado[["IRR (e^β)", "p-valor", "IC 95%"]],
                        use_container_width=True
                    )

                with col_m2:
                    st.markdown("**💡 Interpretação Epidemiológica:**")
                    irr_tempo = np.exp(modelo_glm.params.get("TEMPO_MES", 0))
                    var_mensal = (irr_tempo - 1) * 100

                    if var_mensal > 0:
                        st.error(
                            f"• **Tendência Temporal:** A taxa de condenação apresenta uma tendência de **aumento médio de {var_mensal:.2f}% a cada mês** no período analisado.")
                    else:
                        st.success(
                            f"• **Tendência Temporal:** A taxa de condenação apresenta uma tendência de **redução média de {abs(var_mensal):.2f}% a cada mês** no período analisado.")

                    p_val_tempo = modelo_glm.pvalues.get("TEMPO_MES", 1.0)
                    if p_val_tempo < 0.05:
                        st.caption(
                            f"✓ Tendência estatisticamente significante (p-valor = {p_val_tempo:.4f} < 0.05).")
                    else:
                        st.caption(
                            f"⚠️ Tendência sem significância estatística conclusiva (p-valor = {p_val_tempo:.4f} ≥ 0.05).")

            except Exception as e:
                st.warning(
                    f"Não foi possível ajustar o modelo GLM Poisson: `{e}`")

            st.markdown("---")

            # ---------------------------------------------------------------------
            # 2. ANÁLISE DE DISPERSÃO E VARIABILIDADE (BOXPLOT POR UF)
            # ---------------------------------------------------------------------
            col_st1, col_st2 = st.columns(2)

            with col_st1:
                st.markdown("2. Variabilidade da Taxa por Estado (UF)")

                fig_box = px.box(
                    dados_filtrados,
                    x="UF_PROCEDENCIA",
                    y="percentual_condenados",
                    color="UF_PROCEDENCIA",
                    labels={"UF_PROCEDENCIA": "UF",
                            "percentual_condenados": "Taxa Sanitária (%)"},
                    template="plotly_white"
                )
                fig_box.update_layout(
                    showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                fig_box = formatar_grafico_plotly(fig_box)
                st.plotly_chart(fig_box, use_container_width=True)

            with col_st2:
                st.markdown("3. Linha de Tendência Ponderada (OLS)")

                df_tend = (
                    dados_filtrados
                    .groupby("DATA", as_index=False)
                    .agg(ABATE=("ABATE", "sum"), CONDENADO=("CONDENADO_TOTAL", "sum"))
                )
                df_tend["taxa"] = (df_tend["CONDENADO"] /
                                   df_tend["ABATE"]) * 100

                fig_trend = px.scatter(
                    df_tend,
                    x="DATA",
                    y="taxa",
                    trendline="ols",
                    trendline_color_override="#B22222",
                    labels={"DATA": "Mês/Ano", "taxa": "Taxa Sanitária (%)"},
                    template="plotly_white"
                )
                fig_trend.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                fig_trend = formatar_grafico_plotly(fig_trend)
                st.plotly_chart(fig_trend, use_container_width=True)

# Dashboard de suínos

def renderizar_dashboard_suinos():
    exibir_botao_voltar()
    st.title("🐷 Dashboard Sanitário de Suínos")
    st.caption("Painel de monitoramento de abates e condenações na suinocultura.")

    # Carga dos dados exclusivos da cadeia de suínos
    dados_pais, df_condenacao_bruta = carregar_dados_suinos()
    geo_estados = carregar_malha_estados()
    
    # Tenta carregar base de doenças de suínos (se existir) ou gera DataFrame vazio defensivo
    try:
        df_doenca_bruto = carregar_dados_doencas_suinos()
    except Exception:
        df_doenca_bruto = pd.DataFrame()

    # -------------------------------------------------------------------------
    # SIDEBAR & FILTROS DE SUÍNOS
    # -------------------------------------------------------------------------
    st.sidebar.header("Filtros de Análise — Suínos")
    recorte_regiao = st.sidebar.radio(
        "Região Geográfica", 
        list(REGIOES.keys()), 
        key="radio_regiao_suinos"
    )

    if REGIOES[recorte_regiao] is not None:
        dados_base = dados_pais[dados_pais["UF_PROCEDENCIA"].isin(REGIOES[recorte_regiao])].copy()
    else:
        dados_base = dados_pais.copy()

    anos_disponiveis = sorted(dados_base["ANO"].dropna().unique())
    ufs_disponiveis = sorted(dados_base["UF_PROCEDENCIA"].dropna().unique())
    especies_disponiveis = sorted(dados_base["ESPECIE"].dropna().unique())

    anos = st.sidebar.multiselect(
        "Ano", options=anos_disponiveis, default=anos_disponiveis, key="ms_anos_suinos"
    )
    ufs = st.sidebar.multiselect(
        "Unidade Federativa", options=ufs_disponiveis, default=ufs_disponiveis, key="ms_ufs_suinos"
    )
    especie = st.sidebar.multiselect(
        "Espécie / Categoria", options=especies_disponiveis, default=especies_disponiveis, key="ms_esp_suinos"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Qualidade e Sanitização")
    remover_outliers = st.sidebar.checkbox(
        "Remover Picos Atípicos (> 200%)",
        value=True,
        help="Filtra erros de digitação e registros inconsistentes do SIGSIF na cadeia de suínos.",
        key="chk_outliers_suinos"
    )

    # Filtragem principal da base de suínos
    dados_filtrados = dados_base[
        dados_base["ANO"].isin(anos) &
        dados_base["UF_PROCEDENCIA"].isin(ufs) &
        dados_base["ESPECIE"].isin(especie)
    ].copy()

    # Filtragem da base de doenças de suínos
    if not df_doenca_bruto.empty:
        df_doencas_filtrado = df_doenca_bruto[
            (df_doenca_bruto["ANO"].isin(anos)) &
            (df_doenca_bruto["UF_PROCEDENCIA"].isin(ufs)) &
            (df_doenca_bruto["ESPECIE"].isin(especie))
        ].copy()

        if REGIOES[recorte_regiao] is not None:
            df_doencas_filtrado = df_doencas_filtrado[
                df_doencas_filtrado["UF_PROCEDENCIA"].isin(REGIOES[recorte_regiao])
            ].copy()
    else:
        df_doencas_filtrado = pd.DataFrame()

    # Identificação e remoção de registros atípicos
    dados_outliers = dados_filtrados[dados_filtrados["percentual_condenados"] > 200.0].copy()

    if remover_outliers:
        dados_filtrados = dados_filtrados[dados_filtrados["percentual_condenados"] <= 200.0].copy()

    # -------------------------------------------------------------------------
    # ABAS DA APLICAÇÃO
    # -------------------------------------------------------------------------
    tab_geral, tab_doencas, tab_estatistica = st.tabs([
        "Panorama Geral",
        "Doenças e Condições",
        "Estatística e Modelagem"
    ])

    # =========================================================================
    # ABA 1: PANORAMA GERAL
    # =========================================================================
    with tab_geral:
        st.subheader("Indicadores Gerais de Abate e Condenação — Suínos")
        st.caption("Painel interativo de vigilância sanitária em abatedouros-frigoríficos de suínos.")

        if dados_filtrados.empty:
            st.warning("⚠️ Nenhum registro de abate/condenação de suínos encontrado para os filtros selecionados.")
        else:
            if not dados_outliers.empty and remover_outliers:
                st.warning(
                    f"🧹 **Filtro de Qualidade Ativo:** {len(dados_outliers)} registro(s) com taxa > 200% foi(foram) ocultado(s) "
                    f"para preservar a escala do mapa e dos gráficos."
                )

            total_abatidos = dados_filtrados["ABATE"].sum()
            total_carcaca = dados_filtrados["Carcaça e Cortes"].sum()
            total_visceras = dados_filtrados["Vísceras e Miúdos"].sum()

            taxa_carcaca = (total_carcaca / total_abatidos * 100) if total_abatidos > 0 else 0
            taxa_visceras = (total_visceras / total_abatidos * 100) if total_abatidos > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Volume de Abate Suíno", formata_numero(total_abatidos))
            c2.metric("Condenações de Carcaça", formata_numero(total_carcaca), delta=f"{taxa_carcaca:.2f}% dos abates", delta_color="inverse")
            c3.metric("Condenações de Vísceras/Miúdos", formata_numero(total_visceras), delta=f"{taxa_visceras:.2f}% dos abates", delta_color="off")
            c4.metric("Registros de Ocorrências", formata_numero(total_carcaca + total_visceras))

            st.info(
                "💡 **Nota Técnica Sanitária:** O índice exibido representa o número de **ocorrências notificadas por 100 suínos abatidos**. "
                "Inclui condenações totais de carcaça e descarte parcial de órgãos (fígado, pulmões, rins)."
            )

            st.markdown("---")

            # MAPA INTERATIVO DE SUÍNOS
            st.subheader(f"Indicadores Geográficos por UF — Anos: {', '.join(map(str, anos))}")
            col_map_fig, col_map_ctrl = st.columns([3.5, 1])

            with col_map_ctrl:
                st.markdown("##### Métrica do Mapa")
                metrica_mapa = st.radio(
                    "Visualizar por:",
                    options=["Taxa Sanitária (%)", "Volume Absoluto (Ocorrências)"],
                    index=0,
                    key="radio_mapa_suinos"
                )

            resumo_mapa = (
                dados_filtrados
                .groupby("UF_PROCEDENCIA", as_index=False)
                .agg(
                    ABATE=("ABATE", "sum"),
                    CONDENADO_CARCACA=("Carcaça e Cortes", "sum"),
                    CONDENADO_VISCERAS=("Vísceras e Miúdos", "sum"),
                    CONDENADO_TOTAL=("CONDENADO_TOTAL", "sum")
                )
            )
            resumo_mapa["INDICE_CONDENACAO"] = (resumo_mapa["CONDENADO_TOTAL"] / resumo_mapa["ABATE"]) * 100
            resumo_mapa = resumo_mapa[resumo_mapa["ABATE"] > 0].copy()

            if metrica_mapa == "Taxa Sanitária (%)":
                coluna_cor = "INDICE_CONDENACAO"
                rotulo_cor = "Taxa (%)"
                hover_fmt = "<b>%{location}</b><br>Taxa Sanitária: %{z:.2f}%<br>Abates: %{customdata[0]:,.0f}<extra></extra>"
            else:
                coluna_cor = "CONDENADO_TOTAL"
                rotulo_cor = "Ocorrências"
                hover_fmt = "<b>%{location}</b><br>Ocorrências: %{z:,.0f}<br>Abates: %{customdata[0]:,.0f}<extra></extra>"

            fig_mapa = px.choropleth(
                resumo_mapa,
                geojson=geo_estados,
                locations="UF_PROCEDENCIA",
                featureidkey="properties.UF_SIGLA",
                color=coluna_cor,
                custom_data=["ABATE", "CONDENADO_TOTAL", "INDICE_CONDENACAO"],
                color_continuous_scale="Oranges",
                labels={coluna_cor: rotulo_cor},
            )
            fig_mapa.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
            fig_mapa = formatar_grafico_plotly(fig_mapa)
            fig_mapa.update_traces(hovertemplate=hover_fmt)

            with col_map_fig:
                evento = st.plotly_chart(
                    fig_mapa,
                    use_container_width=True,
                    key="mapa_brasil_suinos",
                    on_select="rerun",
                    selection_mode="points",
                    config={"displayModeBar": False}
                )

            pontos_selecionados = evento.get("selection", {}).get("points", [])
            if pontos_selecionados:
                uf_selecionada = pontos_selecionados[-1]["location"]
                df_uf = resumo_mapa[resumo_mapa["UF_PROCEDENCIA"] == uf_selecionada]

                if not df_uf.empty:
                    detalhe_uf = df_uf.iloc[0]
                    st.info(
                        f"**Detalhes selecionados — {uf_selecionada}:** "
                        f"Abates: {formata_numero(detalhe_uf['ABATE'])} | "
                        f"Carcaças Condenadas: {formata_numero(detalhe_uf['CONDENADO_CARCACA'])} | "
                        f"Vísceras Condenadas: {formata_numero(detalhe_uf['CONDENADO_VISCERAS'])} | "
                        f"Taxa Sanitária Global: {formata_percentual(detalhe_uf['INDICE_CONDENACAO'])}"
                    )
            else:
                st.caption("💡 Clique em um estado no mapa para ver o detalhamento isolado.")

            st.markdown("---")

            # TOP 10 PARTES E SÉRIE TEMPORAL
            col_graf1, col_graf2 = st.columns([1, 1])

            with col_graf1:
                st.subheader("Top 10 Partes Afetadas por Condenção")
                df_partes_top = (
                    df_condenacao_bruta[
                        (df_condenacao_bruta["ANO"].isin(anos)) &
                        (df_condenacao_bruta["UF_PROCEDENCIA"].isin(ufs)) &
                        (df_condenacao_bruta["CATEGORIA_SANITY"] != "Coletas / Amostragens")
                    ]
                    .groupby("PARTE_PADRONIZADA", as_index=False)
                    .agg(TOTAL=("TOTAL_PARTES_AFETADAS", "sum"))
                    .sort_values("TOTAL", ascending=True)
                    .tail(10)
                )

                fig_partes = px.bar(
                    df_partes_top,
                    x="TOTAL",
                    y="PARTE_PADRONIZADA",
                    orientation="h",
                    text_auto=",.0f",
                    labels={"TOTAL": "Registros", "PARTE_PADRONIZADA": "Parte Afetada"},
                    color="TOTAL",
                    color_continuous_scale="Oranges",
                    template="plotly_white"
                )
                fig_partes = formatar_grafico_plotly(fig_partes)
                fig_partes.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_partes, use_container_width=True)

            with col_graf2:
                st.subheader("📈 Evolução Temporal da Taxa Sanitária")
                # 1. Base nacional respeitando o filtro de qualidade/outliers
                if remover_outliers:
                    dados_pais_base = dados_pais[dados_pais["percentual_condenados"] <= 200.0].copy()
                else:
                    dados_pais_base = dados_pais.copy()

                # 2. Agrupamento da Média Nacional Sanitizada
                df_br_tempo = (
                    dados_pais_base
                    .groupby("DATA", as_index=False)
                    .agg(ABATE=("ABATE", "sum"), CONDENADO=("CONDENADO_TOTAL", "sum"))
                )
                df_br_tempo["taxa_brasil"] = (df_br_tempo["CONDENADO"] / df_br_tempo["ABATE"]) * 100

                # 3. Agrupamento do Recorte Selecionado
                df_sel_tempo = (
                    dados_filtrados
                    .groupby("DATA", as_index=False)
                    .agg(ABATE=("ABATE", "sum"), CONDENADO=("CONDENADO_TOTAL", "sum"))
                )
                df_sel_tempo["taxa_selecao"] = (df_sel_tempo["CONDENADO"] / df_sel_tempo["ABATE"]) * 100

                # 4. Verifica se o usuário filtrou algo específico (Estado, Região ou Espécie)
                is_filtro_especifico = (
                    (recorte_regiao != "Brasil") or 
                    (len(ufs) < len(ufs_disponiveis)) or 
                    (len(especie) < len(especies_disponiveis))
                )

                fig_linha = go.Figure()

                if is_filtro_especifico:
                    # Mostra o Benchmark Nacional (pontilhado) + Recorte Selecionado (sólido)
                    fig_linha.add_trace(go.Scatter(
                        x=df_br_tempo["DATA"],
                        y=df_br_tempo["taxa_brasil"],
                        mode="lines",
                        name="Média Brasil (Benchmark)",
                        line=dict(color="#94A3B8", width=2, dash="dash"),
                        hovertemplate="Brasil: %{y:.2f}%<extra></extra>"
                    ))

                    if len(ufs) == 1:
                        nome_trace = f"UF: {ufs[0]}"
                    elif recorte_regiao != "Brasil":
                        nome_trace = f"Região: {recorte_regiao}"
                    else:
                        nome_trace = "Filtro Selecionado"

                    fig_linha.add_trace(go.Scatter(
                        x=df_sel_tempo["DATA"],
                        y=df_sel_tempo["taxa_selecao"],
                        mode="lines+markers",
                        name=nome_trace,
                        line=dict(color="#0284C7", width=3),
                        marker=dict(size=6),
                        hovertemplate=f"{nome_trace}: %{{y:.2f}}%<extra></extra>"
                    ))
                else:
                    # Se a visão for o Brasil como um todo, exibe apenas UMA linha limpa
                    fig_linha.add_trace(go.Scatter(
                        x=df_br_tempo["DATA"],
                        y=df_br_tempo["taxa_brasil"],
                        mode="lines+markers",
                        name="Taxa Sanitária Nacional (Brasil)",
                        line=dict(color="#0284C7", width=3),
                        marker=dict(size=6),
                        hovertemplate="Brasil: %{y:.2f}%<extra></extra>"
                    ))

                fig_linha = formatar_grafico_plotly(fig_linha)
                fig_linha.update_layout(
                    yaxis_title="Taxa Sanitária (%)",
                    xaxis_title="Mês/Ano",
                    hovermode="x unified"
                )

                st.plotly_chart(fig_linha, use_container_width=True)
            if not dados_outliers.empty:
                with st.expander("Ver registros atípicos isolados pelo filtro de qualidade"):
                    st.write(
                        "Abaixo estão os registros originais da base que apresentaram inconsistências de proporção:")
                    st.dataframe(
                        dados_outliers[["ANO", "MES", "UF_PROCEDENCIA", "ESPECIE",
                                        "ABATE", "CONDENADO_TOTAL", "percentual_condenados"]],
                        use_container_width=True,
                        hide_index=True
                                    )
                        # tabela de dados consolidados
            st.subheader("Tabela de Dados Consolidados")

            colunas_tabela = [
                "ANO", "MES", "UF_PROCEDENCIA", "ESPECIE",
                "ABATE", "Carcaça e Cortes", "Vísceras e Miúdos",
                "taxa_carcaca", "taxa_visceras", "percentual_condenados"
            ]

            st.dataframe(
                dados_filtrados[colunas_tabela].sort_values(
                    ["ANO", "MES", "UF_PROCEDENCIA"]),
                use_container_width=True,
                hide_index=True
            )

            with st.expander("Métodos, Fontes e Limitações"):
                st.markdown(
                    """
                    **Fonte dos dados:** Sistema de Informações Gerenciais do Serviço de Inspeção Federal (SIGSIF).

                    **Metodologia de Agrupamento:**
                    * **Carcaça e Cortes:** Inclui registros em carcaças inteiras, meias carcaças, quartos e paletas.
                    * **Vísceras e Miúdos:** Inclui lesões registradas em órgãos e vísceras (fígado, pulmão, rins, etc.).
                    * **Amostragens/Coletas:** Registros de coleta fiscal (PNCRC) foram excluídos da contagem de lesões sanitárias.

                    **Fórmula:** $\\text{Taxa (\\%)} = \\left( \\frac{\\sum \\text{Registros de Condenções}}{\\sum \\text{Animais Abatidos}} \\right) \\times 100$
                    """
                )


    # =========================================================================
    # ABA 2: DOENÇAS E CONDIÇÕES
    # =========================================================================
    with tab_doencas:
        st.subheader("Análise Epidemiológica de Lesões e Patologias em Suínos")
        st.caption("Painel interativo de diagnóstico e causas de condenação em abates suínos.")

        if df_doencas_filtrado.empty:
            st.info("⚠️ Nenhum registro de lesão/diagnóstico de suínos encontrado para os filtros selecionados.")
        else:
            col_d1, col_d2, col_d3 = st.columns(3)
            total_casos = df_doencas_filtrado["QTD_OCORRENCIAS"].sum()

            agrupado_diag = df_doencas_filtrado.groupby("DIAGNOSTICO_PADRONIZADO")["QTD_OCORRENCIAS"].sum()
            top_diag_nome = agrupado_diag.idxmax()
            total_top_diag = agrupado_diag.max()
            pct_top_diag = (total_top_diag / total_casos * 100) if total_casos > 0 else 0

            with col_d1:
                st.metric("Registros de Lesões", formata_numero(total_casos))

                col_d2.metric("Principal Diagnóstico", str(top_diag_nome))
                col_d3.metric("Prevalência do Top 1",
                            f"{pct_top_diag:.1f}% dos casos")

            st.markdown("---")

            col_c1, col_c2 = st.columns([1.8, 1.2])

            with col_c1:
                st.markdown("Treemap de Proporção das Lesões")
                df_tree = (
                    df_doencas_filtrado
                    .groupby("CLASSIFICACAO", as_index=False)
                    .agg(TOTAL=("QTD_OCORRENCIAS", "sum"))
                    .sort_values("TOTAL", ascending=False)
                    .head(15)
                )

                fig_tree = px.treemap(
                    df_tree,
                    path=["CLASSIFICACAO"],
                    values="TOTAL",
                    color="TOTAL",
                    color_continuous_scale="Oranges",
                    template="plotly_white"
                )
                fig_tree.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_tree, use_container_width=True)

            with col_c2:
                st.markdown("Top 10 Enfermidades / Achados com Maior Incidência")
                df_top_diag = (
                    df_doencas_filtrado
                    .groupby("DIAGNOSTICO_PADRONIZADO", as_index=False)
                    .agg(TOTAL=("QTD_OCORRENCIAS", "sum"))
                    .sort_values("TOTAL", ascending=True)
                    .tail(10)
                )

                fig_bar_diag = px.bar(
                    df_top_diag,
                    x="TOTAL",
                    y="DIAGNOSTICO_PADRONIZADO",
                    orientation="h",
                    text_auto=",.0f",
                    labels={"TOTAL": "Ocorrências", "DIAGNOSTICO_PADRONIZADO": "Diagnóstico"},
                    color="TOTAL",
                    color_continuous_scale="Oranges",
                    template="plotly_white"
                )
                fig_bar_diag = formatar_grafico_plotly(fig_bar_diag)
                fig_bar_diag.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_bar_diag, use_container_width=True)

            st.markdown("---")

            st.markdown("##### Investigação por Diagnóstico Específico")
            lista_diagnosticos = sorted(df_doencas_filtrado["DIAGNOSTICO_PADRONIZADO"].unique())
            diag_selecionado = st.selectbox("Selecione uma causa/lesão para detalhar:", lista_diagnosticos, key="sb_diag_suinos")

            df_diag_isolado = df_doencas_filtrado[df_doencas_filtrado["DIAGNOSTICO_PADRONIZADO"] == diag_selecionado].copy()

            col_det1, col_det2 = st.columns(2)

            with col_det1:
                st.markdown(f"**Distribuição por Estado (UF) — {diag_selecionado}:**")
                df_uf_diag = (
                    df_diag_isolado
                    .groupby("UF_PROCEDENCIA", as_index=False)
                    .agg(TOTAL=("QTD_OCORRENCIAS", "sum"))
                    .sort_values("TOTAL", ascending=False)
                    .head(8)
                )
                fig_uf_diag = px.bar(
                    df_uf_diag,
                    x="UF_PROCEDENCIA",
                    y="TOTAL",
                    text_auto=",.0f",
                    labels={"TOTAL": "Casos", "UF_PROCEDENCIA": "UF"},
                    color_discrete_sequence=["#D97706"]
                )
                fig_uf_diag = formatar_grafico_plotly(fig_uf_diag)
                st.plotly_chart(fig_uf_diag, use_container_width=True)

            with col_det2:
                st.markdown(f"**Evolução Temporal dos Casos — {diag_selecionado}:**")
                df_diag_isolado["DATA"] = pd.to_datetime(
                    dict(year=df_diag_isolado["ANO"], month=df_diag_isolado["MES"], day=1),
                    errors="coerce"
                )

                df_tempo_diag = (
                    df_diag_isolado
                    .groupby("DATA", as_index=False)
                    .agg(TOTAL=("QTD_OCORRENCIAS", "sum"))
                    .sort_values("DATA")
                )

                fig_tempo_diag = px.line(
                df_tempo_diag,
                x="DATA",
                y="TOTAL",
                markers=True,
                labels={"DATA": "Mês/Ano", "TOTAL": "Casos Notificados"},
                template="plotly_white"
                )

                fig_tempo_diag.update_traces(line_color="#CD5C5C")
                fig_tempo_diag = formatar_grafico_plotly(fig_tempo_diag)
                fig_tempo_diag.update_xaxes(
                    tickformat="%b/%Y",  # Exibe no formato Mai/2021 (ou use "%m/%Y" para 05/2021)
                    dtick="M1"           # Fixa a menor unidade do eixo em 1 mês
                )

                st.plotly_chart(fig_tempo_diag, use_container_width=True)

    # =========================================================================
    # ABA 3: ESTATÍSTICA E MODELAGEM
    # =========================================================================
    with tab_estatistica:
        st.subheader("📈 Modelagem Estatística e Riscos Epidemiológicos — Suínos")
        st.caption("Aplicações estatísticas avançadas para estimativa de fatores de risco (GLM Poisson) e análise de variabilidade.")

        if dados_filtrados.empty or len(dados_filtrados) < 5:
            st.warning("⚠️ Dados insuficientes para ajuste dos modelos estatísticos no recorte selecionado.")
        else:
            st.markdown("##### 🧪 1. Estimativa de Fatores de Risco (GLM Poisson)")
            df_model = dados_filtrados[dados_filtrados["ABATE"] > 0].copy()

            min_ano = df_model["ANO"].min()
            df_model["TEMPO_MES"] = (df_model["ANO"] - min_ano) * 12 + df_model["MES"]
            df_model["LOG_ABATE"] = np.log(df_model["ABATE"])

            try:
                # Ajusta a fórmula dinamicamente caso haja subcategorias na coluna ESPECIE
                fórmula = "CONDENADO_TOTAL ~ TEMPO_MES"
                if df_model["ESPECIE"].nunique() > 1:
                    fórmula += " + C(ESPECIE)"

                modelo_glm = smf.glm(
                    formula=fórmula,
                    data=df_model,
                    family=sm.families.Poisson(),
                    offset=df_model["LOG_ABATE"]
                ).fit()

                coef_df = pd.DataFrame({
                    "Coeficiente (β)": modelo_glm.params,
                    "IRR (e^β)": np.exp(modelo_glm.params),
                    "Erro Padrão": modelo_glm.bse,
                    "p-valor": modelo_glm.pvalues,
                    "IC 95% Inferior": np.exp(modelo_glm.conf_int()[0]),
                    "IC 95% Superior": np.exp(modelo_glm.conf_int()[1])
                })

                col_m1, col_m2 = st.columns([1.5, 1])

                with col_m1:
                    st.markdown("**Tabela de Estimadores e Risco Relativo (IRR):**")
                    coef_formatado = coef_df.copy()
                    coef_formatado["IRR (e^β)"] = coef_formatado["IRR (e^β)"].map("{:.4f}".format)
                    coef_formatado["p-valor"] = coef_formatado["p-valor"].map(lambda x: "< 0.001" if x < 0.001 else f"{x:.4f}")
                    coef_formatado["IC 95%"] = coef_formatado.apply(
                        lambda r: f"[{r['IC 95% Inferior']:.3f} ; {r['IC 95% Superior']:.3f}]", axis=1
                    )

                    rotulos_variaveis = {
                        "Intercept": "Constante (Taxa Base Suína)",
                        "TEMPO_MES": "Tendência Temporal (Variação Mensal)"
                    }
                    coef_formatado.index = coef_formatado.index.map(lambda x: rotulos_variaveis.get(x, x))

                    st.dataframe(
                        coef_formatado[["IRR (e^β)", "p-valor", "IC 95%"]],
                        use_container_width=True
                    )

                with col_m2:
                    st.markdown("**💡 Interpretação Epidemiológica:**")
                    irr_tempo = np.exp(modelo_glm.params.get("TEMPO_MES", 0))
                    var_mensal = (irr_tempo - 1) * 100

                    if var_mensal > 0:
                        st.error(f"• **Tendência Temporal:** A taxa de condenação de suínos apresenta uma tendência de **aumento médio de {var_mensal:.2f}% a cada mês**.")
                    else:
                        st.success(f"• **Tendência Temporal:** A taxa de condenação de suínos apresenta uma tendência de **redução média de {abs(var_mensal):.2f}% a cada mês**.")

                    p_val_tempo = modelo_glm.pvalues.get("TEMPO_MES", 1.0)
                    if p_val_tempo < 0.05:
                        st.caption(f"✓ Tendência estatisticamente significante (p-valor = {p_val_tempo:.4f} < 0.05).")
                    else:
                        st.caption(f"⚠️ Tendência sem significância estatística conclusiva (p-valor = {p_val_tempo:.4f} ≥ 0.05).")

            except Exception as e:
                st.warning(f"Não foi possível ajustar o modelo GLM Poisson para suínos: `{e}`")

            st.markdown("---")

            col_st1, col_st2 = st.columns(2)

            with col_st1:
                st.markdown("##### 📊 Variabilidade da Taxa por Estado (UF)")
                fig_box = px.box(
                    dados_filtrados,
                    x="UF_PROCEDENCIA",
                    y="percentual_condenados",
                    color="UF_PROCEDENCIA",
                    labels={"UF_PROCEDENCIA": "UF", "percentual_condenados": "Taxa Sanitária (%)"},
                    template="plotly_white"
                )
                fig_box.update_layout(showlegend=False)
                fig_box = formatar_grafico_plotly(fig_box)
                st.plotly_chart(fig_box, use_container_width=True)

            with col_st2:
                st.markdown("##### 📉 Linha de Tendência Ponderada (OLS)")
                df_tend = (
                    dados_filtrados
                    .groupby("DATA", as_index=False)
                    .agg(ABATE=("ABATE", "sum"), CONDENADO=("CONDENADO_TOTAL", "sum"))
                )
                df_tend["taxa"] = (df_tend["CONDENADO"] / df_tend["ABATE"]) * 100

                fig_trend = px.scatter(
                    df_tend,
                    x="DATA",
                    y="taxa",
                    trendline="ols",
                    trendline_color_override="#D97706",
                    labels={"DATA": "Mês/Ano", "taxa": "Taxa Sanitária (%)"},
                    template="plotly_white"
                )
                fig_trend = formatar_grafico_plotly(fig_trend)
                st.plotly_chart(fig_trend, use_container_width=True)


# Roteador principal para renderizar os dashboards de acordo com a seleção do usuário
if st.session_state["cadeia_selecionada"] is None:
    renderizar_tela_inicial()
elif st.session_state["cadeia_selecionada"] == "bovinos":
    renderizar_dashboard_bovinos()
elif st.session_state["cadeia_selecionada"] == "suinos":
    renderizar_dashboard_suinos()
