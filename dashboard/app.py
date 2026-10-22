from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import json

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

# camada de dados e cache para otimizar o carregamento


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

    # Consolida o abate para o país todo
    abate_consolidado = (
        abate
        .groupby(chaves, as_index=False)
        .agg(ABATE=("QUANTIDADE", "sum"))
    )

    # Consolida as condenações para o país todo
    condenacao_consolidado = (
        condenacao
        .groupby(chaves, as_index=False)
        .agg(CONDENADO=("TOTAL_PARTES_AFETADAS", "sum"))
    )

    # Base nacional: abate é o denominador e, por isso, é a base do merge
    dados = abate_consolidado.merge(
        condenacao_consolidado,
        on=chaves,
        how="left",
        validate="one_to_one"
    )

    anos_analise = [2021, 2022, 2023, 2024, 2025, 2026]

    dados = dados[dados["ANO"].isin(anos_analise)].copy()

    dados["CONDENADO"] = dados["CONDENADO"].fillna(0)

    dados["DATA"] = pd.to_datetime(
        dict(
            year=dados["ANO"],
            month=dados["MES"],
            day=1
        ),
        errors="coerce"
    )

    dados["percentual_condenados"] = (
        dados["CONDENADO"] / dados["ABATE"] * 100
    )

    return dados


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


dados_pais = carregar_dados()
geo_estados = carregar_malha_estados()

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

dados_filtrados = dados_base[
    dados_base["ANO"].isin(anos) &
    dados_base["UF_PROCEDENCIA"].isin(ufs) &
    dados_base["ESPECIE"].isin(especie)
].copy()

if dados_filtrados.empty:
    st.warning("Nenhum registro encontrado para os filtros selecionados.")
    st.stop()

# KPIS e métricas principais

total_abatidos = dados_filtrados["ABATE"].sum()
total_condenados = dados_filtrados["CONDENADO"].sum()
indice_global = (total_condenados / total_abatidos *
                 100) if total_abatidos > 0 else 0
mediana_mensal = dados_filtrados["percentual_condenados"].median()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Animais Abatidos", formata_numero(total_abatidos))
c2.metric("Condenações Registradas", formata_numero(total_condenados))
c3.metric("Índice Global de Condenação", formata_percentual(indice_global))
c4.metric("Mediana Mensal do Índice", formata_percentual(mediana_mensal))

st.info(
    "💡 **Nota de Interpretação Sanitária:** O índice exibido reflete o volume de "
    "**partes/carcaças afetadas por 100 animais abatidos**. Uma mesma carcaça com duas "
    "meias-carcaças condenadas gerará 2 registros no indicador."
)

st.markdown("---")

# mapa interativo com Plotly Express
st.subheader(f"Indicadores Geográficos — Anos: {', '.join(map(str, anos))}")

resumo_mapa = (
    dados_filtrados
    .groupby("UF_PROCEDENCIA", as_index=False)
    .agg(ABATE=("ABATE", "sum"), CONDENADO=("CONDENADO", "sum"))
)
resumo_mapa["INDICE_CONDENACAO"] = (
    resumo_mapa["CONDENADO"] / resumo_mapa["ABATE"]) * 100
resumo_mapa = resumo_mapa[resumo_mapa["ABATE"] > 0].copy()

if resumo_mapa.empty:
    st.info("Nenhum dado disponível para renderizar o mapa com os filtros atuais.")
else:
    fig_mapa = px.choropleth(
        resumo_mapa,
        geojson=geo_estados,
        locations="UF_PROCEDENCIA",
        featureidkey="properties.UF_SIGLA",
        color="INDICE_CONDENACAO",
        color_continuous_scale="YlOrRd",
        labels={"INDICE_CONDENACAO": "Índice (%)"},
    )

    # Ajusta o enquadramento geográfico no Brasil
    fig_mapa.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator"
    )

    fig_mapa.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar_title="Índice (%)"
    )

    fig_mapa.update_traces(
        hovertemplate="<b>%{location}</b><br>Índice de condenação: %{z:.2f}%<extra></extra>"
    )

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
                f"Condenações: {formata_numero(detalhe_uf['CONDENADO'])} | "
                f"Taxa: {formata_percentual(detalhe_uf['INDICE_CONDENACAO'])}"
            )
    else:
        st.caption("Clique em um estado no mapa para ver o detalhamento isolado.")

# Análise temporal com gráfico de linha e distribuição de densidade

st.subheader("Evolução Mensal do Índice de Condenção")
df_linha = (
    # agrupamento para evitar ruídos
    dados_filtrados.groupby(["DATA", "UF_PROCEDENCIA"], as_index=False)
    .agg(ABATE=("ABATE", "sum"), CONDENADO=("CONDENADO", "sum"))
)
df_linha["percentual_condenados"] = (
    df_linha["CONDENADO"] / df_linha["ABATE"]) * 100

fig_linha = px.line(
    df_linha.sort_values("DATA"),
    x="DATA",
    y="percentual_condenados",
    color="UF_PROCEDENCIA",
    markers=True,
    labels={"DATA": "Mês/Ano",
            "percentual_condenados": "Índice (%)", "UF_PROCEDENCIA": "UF"},
    template="plotly_white"
)
fig_linha.update_yaxes(rangemode="tozero")
st.plotly_chart(fig_linha, use_container_width=True)

st.subheader("Distribuição e Dispersão Mensal por UF")

fig_box = px.box(
    dados_filtrados,
    x="UF_PROCEDENCIA",
    y="percentual_condenados",
    color="UF_PROCEDENCIA",
    points="all",
    labels={"UF_PROCEDENCIA": "UF", "percentual_condenados": "Índice (%)"},
    template="plotly_white"
)
fig_box.update_layout(showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

# Tabela e documentação dos métodos e limitações
st.subheader("Tabela de Dados Consolidados")

colunas_tabela = ["ANO", "MES", "UF_PROCEDENCIA", "ESPECIE",
                  "ABATE", "CONDENADO", "percentual_condenados"]
st.dataframe(
    dados_filtrados[colunas_tabela].sort_values(
        ["ANO", "MES", "UF_PROCEDENCIA"]),
    use_container_width=True,
    hide_index=True
)

with st.expander("Métodos e limitações"):
    st.markdown(
        """
        **Fonte dos dados:** Sistema de Informações Gerenciais do Serviço de
        Inspeção Federal (SIGSIF), extraído em 14 de abril de 2026.

        **Indicador:** condenações de carcaças e meias carcaças por 100 animais
        abatidos.

        **Cálculo:** `(número de condenações / número de animais abatidos) × 100`.

        **Limitação:** os registros representam condenações de carcaças e meias
        carcaças; não permitem afirmar que cada registro corresponda a um animal
        individual. Os valores devem ser interpretados como índices de registros
        de condenação por animais abatidos.
        """
    )
