import geopandas as gpd

gdf = gpd.read_file(
    "C:\\Users\\thami\\OneDrive\\Desktop\\Projeto PPDI\\dashboard\\BR_UF_2025\\BR_UF_2025.shp")

print(gdf.columns)

gdf = gdf.rename(columns={"NM_UF": "UF", "SIGLA_UF": "UF_SIGLA"})

gdf = gdf.to_crs(epsg=4326)

gdf = gdf[["UF", "UF_SIGLA", "geometry"]]

gdf.to_file("dados/malha_estados.json", driver="GeoJSON", encoding="utf-8")
