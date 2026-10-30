from pathlib import Path
import shutil
import pandas as pd

BASE_DIR = Path(__file__).parent
DIR_TRATADOS = BASE_DIR.parent / "dados" / "tratados"
DIR_MALHA = BASE_DIR.parent / "dados"
DIR_PROD = BASE_DIR / "dados" / "prod"
DIR_PROD.mkdir(parents=True, exist_ok=True)

#Caminhos
CAMINHO_BOVINO_ABATE = DIR_TRATADOS / "bovino_bubalinos_abate_país.parquet"
CAMINHO_BOVINO_COND  = DIR_TRATADOS / "bovino_bubalinos_condenacao_ano.parquet"

CAMINHO_SUINO_ABATE  = DIR_TRATADOS / "suino_abate.parquet"
CAMINHO_SUINO_COND   = DIR_TRATADOS / "suino_ano.parquet"

CAMINHO_SUINO_DOENCAS  = DIR_TRATADOS / "suino_doenças_final.parquet"
CAMINHO_BOVINO_DOENCAS = DIR_TRATADOS / "df_bovino_bubalino_doenças_10.parquet"
CAMINHO_MALHA          = DIR_TRATADOS / "malha_municipal.parquet"

def normalizar_nome_especie(val) -> str:
    v = str(val).upper().strip()
    if "BUBAL" in v:
        return "BUBALINO"
    if "BOVIN" in v:
        return "BOVINO"
    if "SUIN" in v:
        return "SUINO"
    return v

def padronizar_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    col_uf = next((c for c in ["UF_PROCEDENCIA", "UF", "ESTADO", "SG_UF"] if c in df.columns), "UF_PROCEDENCIA")
    if col_uf != "UF_PROCEDENCIA":
        df.rename(columns={col_uf: "UF_PROCEDENCIA"}, inplace=True)

    col_esp = next((c for c in ["ESPECIE", "ESPÉCIE", "ESPECIE_ANIMAL", "DESCRICAO_ESPECIE", "DSC_ESPECIE"] if c in df.columns), None)
    if col_esp:
        if col_esp != "ESPECIE":
            df.rename(columns={col_esp: "ESPECIE"}, inplace=True)
        df["ESPECIE"] = df["ESPECIE"].apply(normalizar_nome_especie)

    df["UF_PROCEDENCIA"] = df["UF_PROCEDENCIA"].astype(str).str.strip().str.upper()
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").fillna(0).astype("uint16")
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(0).astype("uint8")
    return df


def categorizar_parte(desc: str) -> str:
    if not isinstance(desc, str): return "Vísceras e Miúdos"
    d = desc.upper()
    if any(p in d for p in ["CARCAÇA", "CARCACA", "QUARTO", "MEIA CARCAÇA", "PALETA", "CORTE"]):
        return "Carcaça e Cortes"
    if any(p in d for p in ["COLETA", "AMOSTRA", "MONITORAMENTO"]):
        return "Coletas / Amostragens"
    return "Vísceras e Miúdos"


def processar_especie(path_abate: Path, path_cond: Path, nome_saida: str):
    if not path_abate.exists() or not path_cond.exists():
        print(f"⚠️ Arquivos de {nome_saida} não encontrados.")
        return

    print(f"⏳ Processando {nome_saida}...")
    df_abate = padronizar_df(pd.read_parquet(path_abate))
    df_cond  = padronizar_df(pd.read_parquet(path_cond))

    chaves = ["ANO", "MES", "UF_PROCEDENCIA"]
    if "ESPECIE" in df_abate.columns and "ESPECIE" in df_cond.columns:
        chaves.append("ESPECIE")

    col_qtd_abate = "QUANTIDADE" if "QUANTIDADE" in df_abate.columns else df_abate.columns[-1]
    col_qtd_cond  = next((c for c in ["TOTAL_PARTES_AFETADAS", "QUANTIDADE", "QTDE", "TOTAL"] if c in df_cond.columns), df_cond.columns[-1])
    col_parte     = next((c for c in ["PARTE_PADRONIZADA", "PARTE_AFETADA", "DESCRICAO_PARTE", "PARTE"] if c in df_cond.columns), df_cond.columns[0])

    abate_cons = df_abate.groupby(chaves, as_index=False).agg(ABATE=(col_qtd_abate, "sum"))

    df_cond["CATEGORIA_SANITY"] = df_cond[col_parte].apply(categorizar_parte)
    df_lesoes = df_cond[df_cond["CATEGORIA_SANITY"] != "Coletas / Amostragens"].copy()

    cat_summary = df_lesoes.groupby(chaves + ["CATEGORIA_SANITY"], as_index=False).agg(CONDENADO=(col_qtd_cond, "sum"))
    pivot_summary = cat_summary.pivot_table(index=chaves, columns="CATEGORIA_SANITY", values="CONDENADO", fill_value=0).reset_index()

    for c in ["Carcaça e Cortes", "Vísceras e Miúdos"]:
        if c not in pivot_summary.columns: pivot_summary[c] = 0

    df_final = abate_cons.merge(pivot_summary, on=chaves, how="left")
    df_final["Carcaça e Cortes"] = df_final["Carcaça e Cortes"].fillna(0).astype("uint32")
    df_final["Vísceras e Miúdos"] = df_final["Vísceras e Miúdos"].fillna(0).astype("uint32")
    df_final["ABATE"] = df_final["ABATE"].astype("uint32")
    df_final["CONDENADO_TOTAL"] = df_final["Carcaça e Cortes"] + df_final["Vísceras e Miúdos"]

    if "ESPECIE" not in df_final.columns:
        df_final["ESPECIE"] = "SUINO" if "suino" in nome_saida.lower() else "BOVINO"

    df_valid = df_final[(df_final["ANO"] > 2000) & (df_final["MES"].between(1, 12))].copy()
    df_valid["DATA"] = pd.to_datetime(dict(year=df_valid["ANO"], month=df_valid["MES"], day=1))
    
    colunas_merge = ["ANO", "MES", "UF_PROCEDENCIA", "DATA"]
    if "ESPECIE" in df_final.columns:
        colunas_merge.append("ESPECIE")

    df_final = df_final.merge(df_valid[colunas_merge], on=chaves, how="left")
    df_final["percentual_condenados"] = (df_final["CONDENADO_TOTAL"] / df_final["ABATE"]) * 100

    out_path = DIR_PROD / f"{nome_saida}_consolidado.parquet"
    df_final.to_parquet(out_path, index=False, compression="snappy")
    print(f"✅ Salvo base principal: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"   👉 Espécies encontradas em {nome_saida}: {df_final['ESPECIE'].unique().tolist()}")

    # Partes Afetadas
    df_lesoes["PARTE_PADRONIZADA"] = df_lesoes[col_parte].astype(str).str.strip().str.title()
    partes_cons = df_lesoes.groupby(chaves + ["PARTE_PADRONIZADA"], as_index=False).agg(
        TOTAL_PARTES_AFETADAS=(col_qtd_cond, "sum")
    )
    partes_cons["TOTAL_PARTES_AFETADAS"] = pd.to_numeric(partes_cons["TOTAL_PARTES_AFETADAS"], errors="coerce").fillna(0).astype("uint32")

    if "ESPECIE" not in partes_cons.columns:
        partes_cons["ESPECIE"] = "SUINO" if "suino" in nome_saida.lower() else "BOVINO"

    out_partes = DIR_PROD / f"{nome_saida}_partes.parquet"
    partes_cons.to_parquet(out_partes, index=False, compression="snappy")
    print(f"✅ Salvo partes: {out_partes.name} ({out_partes.stat().st_size / 1024:.1f} KB)")


def processar_todas_doencas(caminhos_doencas: dict):
    dfs = []
    for chave_especie, path in caminhos_doencas.items():
        if not path.exists():
            continue

        print(f"⏳ Processando doenças ({chave_especie})...")
        df = padronizar_df(pd.read_parquet(path))

        col_doenca = "DIAGNOSTICO" if "DIAGNOSTICO" in df.columns else df.columns[0]
        col_qtd    = "CASOS" if "CASOS" in df.columns else df.columns[-1]
        col_class  = next((c for c in ["CLASSIFICACAO", "CLASSIFICACAO_DOENCA", "GRUPO", "CATEGORIA"] if c in df.columns), None)

        chaves_doenca = ["ANO", "MES", "UF_PROCEDENCIA", col_doenca]
        if "ESPECIE" in df.columns:
            chaves_doenca.append("ESPECIE")
        if col_class:
            chaves_doenca.append(col_class)

        df_cons = df.groupby(chaves_doenca, as_index=False).agg(TOTAL=(col_qtd, "sum"))

        if "ESPECIE" not in df_cons.columns:
            df_cons["ESPECIE"] = chave_especie

        renames = {col_doenca: "DIAGNOSTICO_PADRONIZADO", "TOTAL": "QTD_OCORRENCIAS"}
        if col_class:
            renames[col_class] = "CLASSIFICACAO"

        df_cons.rename(columns=renames, inplace=True)
        if "CLASSIFICACAO" not in df_cons.columns:
            df_cons["CLASSIFICACAO"] = df_cons["DIAGNOSTICO_PADRONIZADO"]

        dfs.append(df_cons)

    if dfs:
        df_doencas_final = pd.concat(dfs, ignore_index=True)
        out_path = DIR_PROD / "doencas_consolidado.parquet"
        df_doencas_final.to_parquet(out_path, index=False, compression="snappy")
        print(f"✅ Salvo doenças consolidado: {out_path.name}")


def copiar_malha(path_malha: Path):
    if path_malha.exists():
        shutil.copy(path_malha, DIR_MALHA / path_malha.name)
        print(f"✅ Malha copiada: {path_malha.name}")


if __name__ == "__main__":
    processar_especie(CAMINHO_BOVINO_ABATE, CAMINHO_BOVINO_COND, "bovino")
    processar_especie(CAMINHO_SUINO_ABATE, CAMINHO_SUINO_COND, "suino")
    processar_todas_doencas({"BOVINO": CAMINHO_BOVINO_DOENCAS, "SUÍNO": CAMINHO_SUINO_DOENCAS})
    copiar_malha(CAMINHO_MALHA)