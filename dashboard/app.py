from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import json
import numpy as np

st.set_page_config(page_title="Dashboard Abate e condenação em frigoríficos",
                   page_icon=":bar_chart:",
                   layout="wide")

st.title("Dashboard Abate e condenação em frigoríficos")
st.caption("Este dashboard apresenta informações sobre o abate e condenação em frigoríficos, permitindo uma análise visual dos dados.")

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
                   "TOTAL", "CASOS"] if c in doenca.columns), None)
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


dados_pais, df_condenacao_bruta = carregar_dados()
geo_estados = carregar_malha_estados()
df_doenca_bruto = carregar_dados_doencas()

# filtros dinâmicos e interativos com Streamlit

st.sidebar.header("Filtros de Analise")
recorte_regiao = st.sidebar.radio("Região Geográfica", list(REGIOES.keys()))

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
    "🧹 Remover Picos Atípicos (> 200%)",
    value=True,
    help="Filtra erros de digitação e registros inconsistentes do SIGSIF (ex: SC 05/2024) para evitar a distorção da escala do mapa e dos gráficos."
)

dados_filtrados = dados_base[
    dados_base["ANO"].isin(anos) &
    dados_base["UF_PROCEDENCIA"].isin(ufs) &
    dados_base["ESPECIE"].isin(especie)
].copy()

df_doencas_filtrado = df_doenca_bruto[
    (df_doenca_bruto["ANO"].isin(anos)) &
    (df_doenca_bruto["UF_PROCEDENCIA"].isin(ufs)) &
    (df_doenca_bruto["ESPECIE"].isin(especie))
].copy()

# Se o filtro regional estiver ativo (ex: Sudeste, Sul, etc.)
if REGIOES[recorte_regiao] is not None:
    df_doencas_filtrado = df_doencas_filtrado[
        df_doencas_filtrado["UF_PROCEDENCIA"].isin(REGIOES[recorte_regiao])
    ].copy()

dados_outliers = dados_filtrados[dados_filtrados["percentual_condenados"] > 200.0].copy(
)

if remover_outliers:
    dados_filtrados = dados_filtrados[dados_filtrados["percentual_condenados"] <= 200.0].copy(
    )

if dados_filtrados.empty:
    st.warning("Nenhum registro encontrado para os filtros selecionados.")
    st.stop()

if not dados_outliers.empty and remover_outliers:
    qtd_outliers = len(dados_outliers)
    st.warning(
        f"🧹 **Filtro de Qualidade Ativo:** {qtd_outliers} registro(s) com inconsistência de digitação/taxa > 200% foi(foram) ocultado(s) "
        f"para preservar a escala do mapa e dos gráficos. *(Ex: SC em 05/2024)*."
    )

# divisão das abas
tab_geral, tab_doencas, tab_estatistica = st.tabs([
    "Panorama Geral",
    "Doenças e Condições",
    "Estatística e Modelagem"
])

# Aba "Panorama Geral" - KPIs, Mapa e Gráficos
# KPIS e métricas principais

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
        options=["Taxa Sanitária (%)", "Volume Absoluto (Ocorrências)"],
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
        labels={"TOTAL": "Registros", "PARTE_PADRONIZADA": "Parte Afetada"},
        color="TOTAL",
        color_continuous_scale="Reds",
        template="plotly_white"
    )
    fig_partes.update_layout(coloraxis_showscale=False,
                             margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_partes, use_container_width=True)

    with col_graf2:
        st.subheader("explicação - Top 10 UFs por Taxa de Condenação")

UF_PARA_REGIAO = {uf: reg for reg,
                  ufs in REGIOES.items() if ufs for uf in ufs}
st.subheader("📈 Evolução Mensal do Índice de Condenação")

if "REGIAO" not in dados_filtrados.columns:
    dados_filtrados["REGIAO"] = dados_filtrados["UF_PROCEDENCIA"].map(
        UF_PARA_REGIAO)

coluna_grupo = "REGIAO" if recorte_regiao == "Brasil" else "UF_PROCEDENCIA"
titulo_legenda = "Região" if recorte_regiao == "Brasil" else "Estado (UF)"

df_linha = (
    dados_filtrados
    .groupby(["DATA", coluna_grupo], as_index=False)
    .agg(ABATE=("ABATE", "sum"), CONDENADO=("CONDENADO_TOTAL", "sum"))
)
df_linha["percentual_condenados"] = (
    df_linha["CONDENADO"] / df_linha["ABATE"]) * 100

fig_linha = px.line(
    df_linha.sort_values("DATA"),
    x="DATA",
    y="percentual_condenados",
    color=coluna_grupo,
    markers=True,
    labels={"DATA": "Mês/Ano",
            "percentual_condenados": "Taxa Sanitária (%)"},
    template="plotly_white"
)

fig_linha.update_yaxes(rangemode="tozero")
fig_linha.update_layout(
    legend_title_text=titulo_legenda,
    margin=dict(l=0, r=0, t=10, b=0),
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

# aba 2 doenças e condições
with tab_doencas:
    st.subheader("🦠 Análise Epidemiológica de Doenças e Diagnósticos")
    st.caption(
        "Painel interativo de vigilância sanitária baseado nos dados do arquivo SE_10.")

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
        col_d3.metric("Prevalência do Top 1", f"{pct_top_diag:.1f}% dos casos")

        st.markdown("---")

        # 2. Ranking e Distribuição
        col_c1, col_c2 = st.columns([1.8, 1.2])

        with col_c1:
            st.markdown("##### 🧩 Treemap de Proporção dos Diagnósticos")
            df_tree = (
                df_doencas_filtrado
                .groupby("DIAGNOSTICO_PADRONIZADO", as_index=False)
                .agg(TOTAL=("QTD_OCORRENCIAS", "sum"))
                .sort_values("TOTAL", ascending=False)
                .head(15)  # Top 15 para não poluir o gráfico
            )

            fig_tree = px.treemap(
                df_tree,
                path=["DIAGNOSTICO_PADRONIZADO"],
                values="TOTAL",
                color="TOTAL",
                color_continuous_scale="Reds",
                template="plotly_white"
            )
            fig_tree.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)

        with col_c2:
            st.markdown("##### 🏆 Top 10 Enfermidades com Maior Incidência")
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
        st.markdown("##### 🔍 Investigação por Doença Específica")

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
    st.subheader("Estatística e Modelagem de Dados")
