# =================================================================================================
# SCRIPT 6: analise_climatica.py
#
# OBJETIVO:
# Ler os dados climáticos anuais já processados (GeoTIFFs de temperatura e pluviosidade)
# e gerar um conjunto completo de análises e visualizações. As análises incluem estatísticas
# nacionais, tendências temporais, anomalias, clusterização espacial para identificar
# zonas climáticas e estatísticas por estado.
#
# PASSOS EXECUTADOS:
# 1.  Calcula estatísticas anuais (média) para temperatura e chuva para o Brasil.
# 2.  Gera gráficos de tendência temporal para as variáveis climáticas.
# 3.  Gera gráficos de anomalias, comparando cada ano com a média do período.
# 4.  Cria mapas de anomalia espacial para um ano específico, mostrando onde o clima
#     foi diferente da média histórica.
# 5.  Aplica Machine Learning (K-Means) para agrupar o Brasil em zonas climáticas
#     com base na média de temperatura e chuva.
# 6.  Extrai estatísticas climáticas anuais para cada estado brasileiro.
# =================================================================================================

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.warp import reproject, Resampling
import numpy as np
import glob
import os
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 2. CONFIGURAÇÃO DOS CAMINHOS E DIRETÓRIOS ---
# Define os caminhos de entrada e saída para manter o projeto organizado.

INPUT_GEOTIFF_DIR = "resultados_climaticos/geotiff_maps"  # Pasta com os GeoTIFFs climáticos anuais.
OUTPUT_DIR_BASE = "analises_climaticas"                  # Pasta principal para salvar todos os resultados deste script.
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_DIR_BASE, 'plots') # Subpasta para salvar gráficos e mapas em PNG.
OUTPUT_CSV_DIR = os.path.join(OUTPUT_DIR_BASE, 'csv_data')   # Subpasta para salvar as tabelas de dados (CSVs).
OUTPUT_GEOTIFF_DIR = os.path.join(OUTPUT_DIR_BASE, 'geotiff_maps') # Subpasta para salvar novos GeoTIFFs (ex: mapa de clusters).

# Cria todos os diretórios de saída, se ainda não existirem.
for d in [OUTPUT_PLOTS_DIR, OUTPUT_CSV_DIR, OUTPUT_GEOTIFF_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 3. FUNÇÕES DE ANÁLISE ---
# Cada análise é encapsulada em sua própria função para clareza e modularidade.

def gerar_estatisticas_climaticas(geotiff_dir, brasil_aoi_reprojected, output_csv_path):
    """
    Calcula estatísticas climáticas médias anuais para todo o Brasil.

    Args:
        geotiff_dir (str): Caminho para a pasta com os GeoTIFFs anuais.
        brasil_aoi_reprojected (dict): Dicionário contendo o shapefile do Brasil já reprojetado
                                       para o CRS de cada tipo de dado ('temp' e 'rain').
        output_csv_path (str): Caminho completo para salvar o arquivo CSV com as estatísticas.

    Returns:
        pd.DataFrame: DataFrame com as estatísticas anuais.
    """
    print("[Análise 1/6] Gerando estatísticas climáticas anuais...")
    
    # Lista os arquivos de temperatura e pluviosidade.
    temp_files = sorted(glob.glob(os.path.join(geotiff_dir, 'temperatura_media_*.tif')))
    rain_files = sorted(glob.glob(os.path.join(geotiff_dir, 'pluviosidade_total_*.tif')))
    
    # Extrai os anos a partir dos nomes dos arquivos de temperatura.
    anos = [os.path.basename(f).split('_')[-1].replace('.tif', '') for f in temp_files]
    
    climate_data = []
    
    # Itera sobre cada ano para calcular as estatísticas nacionais.
    for ano in tqdm(anos, desc="Calculando estatísticas nacionais"):
        temp_file = os.path.join(geotiff_dir, f'temperatura_media_{ano}.tif')
        rain_file = os.path.join(geotiff_dir, f'pluviosidade_total_{ano}.tif')
        
        # Processa a temperatura: recorta o raster usando a máscara do Brasil e calcula a média.
        with rasterio.open(temp_file) as src:
            brasil_mask = brasil_aoi_reprojected['temp']
            # A função mask recorta a imagem, retornando apenas os pixels dentro da geometria do Brasil.
            temp_data, _ = mask(src, brasil_mask.geometry, crop=True, nodata=np.nan)
            temp_media = np.nanmean(temp_data[0]) # Calcula a média dos pixels, ignorando valores NaN.
        
        # Processa a pluviosidade de forma similar.
        with rasterio.open(rain_file) as src:
            brasil_mask = brasil_aoi_reprojected['rain']
            rain_data, _ = mask(src, brasil_mask.geometry, crop=True, nodata=np.nan)
            chuva_media_anual = np.nanmean(rain_data[0])
            
        climate_data.append({
            'Ano': int(ano),
            'Temp_Media_C': temp_media,
            'Chuva_Media_Anual_mm': chuva_media_anual
        })

    # Cria um DataFrame e salva como CSV.
    df_climate = pd.DataFrame(climate_data)
    df_climate.to_csv(output_csv_path, index=False, decimal=',', sep=';')
    print(f" -> Estatísticas climáticas salvas em: {output_csv_path}")
    return df_climate

def analise_tendencia_climatica(df_climate, output_dir):
    """
    Gera e salva gráficos da série temporal e linha de tendência para temperatura e pluviosidade.
    """
    print("[Análise 2/6] Gerando gráficos de tendência climática...")
    
    if df_climate is None or df_climate.empty: return
    
    # Gráfico de tendência para Temperatura
    fig, ax = plt.subplots(figsize=(10, 6))
    slope, intercept, r_value, _, _ = stats.linregress(df_climate['Ano'], df_climate['Temp_Media_C'])
    ax.plot(df_climate['Ano'], df_climate['Temp_Media_C'], marker='o', linestyle='-', color='red')
    ax.plot(df_climate['Ano'], intercept + slope*df_climate['Ano'], 'r--', label=f'Tendência (R²={r_value**2:.2f})')
    ax.set_title('Tendência da Temperatura Média Anual - Brasil', fontsize=16)
    ax.set_xlabel('Ano'); ax.set_ylabel('Temperatura Média (°C)'); ax.legend(); ax.grid(True)
    plt.savefig(os.path.join(output_dir, 'tendencia_temperatura_brasil.png'), dpi=300, bbox_inches='tight'); plt.close(fig)

    # Gráfico de tendência para Pluviosidade
    fig, ax = plt.subplots(figsize=(10, 6))
    slope, intercept, r_value, _, _ = stats.linregress(df_climate['Ano'], df_climate['Chuva_Media_Anual_mm'])
    ax.plot(df_climate['Ano'], df_climate['Chuva_Media_Anual_mm'], marker='o', linestyle='-', color='blue')
    ax.plot(df_climate['Ano'], intercept + slope*df_climate['Ano'], 'b--', label=f'Tendência (R²={r_value**2:.2f})')
    ax.set_title('Tendência da Pluviosidade Média Anual - Brasil', fontsize=16)
    ax.set_xlabel('Ano'); ax.set_ylabel('Pluviosidade Média Anual (mm)'); ax.legend(); ax.grid(True)
    plt.savefig(os.path.join(output_dir, 'tendencia_pluviosidade_brasil.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
    print(" -> Gráficos de tendência salvos.")

def analise_anomalias_climaticas(df_climate, output_dir):
    """
    Calcula e plota as anomalias anuais (desvio de cada ano em relação à média do período).
    """
    print("[Análise 3/6] Gerando gráficos de anomalias climáticas...")

    if df_climate is None or df_climate.empty: return

    # Gráfico de anomalia de Temperatura
    mean_temp = df_climate['Temp_Media_C'].mean()
    df_climate['Anomalia_Temp'] = df_climate['Temp_Media_C'] - mean_temp
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in df_climate['Anomalia_Temp']] # Vermelho para > média, Azul para < média
    fig, ax = plt.subplots(figsize=(10, 6)); ax.bar(df_climate['Ano'], df_climate['Anomalia_Temp'], color=colors)
    ax.set_title('Anomalia da Temperatura Anual em Relação à Média do Período', fontsize=16); ax.set_xlabel('Ano'); ax.set_ylabel('Anomalia de Temperatura (°C)'); ax.axhline(0, color='grey', lw=1)
    plt.savefig(os.path.join(output_dir, 'anomalias_temperatura.png'), dpi=300, bbox_inches='tight'); plt.close(fig)

    # Gráfico de anomalia de Pluviosidade
    mean_rain = df_climate['Chuva_Media_Anual_mm'].mean()
    df_climate['Anomalia_Chuva'] = df_climate['Chuva_Media_Anual_mm'] - mean_rain
    colors = ['#1f77b4' if x > 0 else '#8c564b' for x in df_climate['Anomalia_Chuva']] # Azul para > média, Marrom para < média
    fig, ax = plt.subplots(figsize=(10, 6)); ax.bar(df_climate['Ano'], df_climate['Anomalia_Chuva'], color=colors)
    ax.set_title('Anomalia da Pluviosidade Anual em Relação à Média do Período', fontsize=16); ax.set_xlabel('Ano'); ax.set_ylabel('Anomalia de Pluviosidade (mm)'); ax.axhline(0, color='grey', lw=1)
    plt.savefig(os.path.join(output_dir, 'anomalias_pluviosidade.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
    print(" -> Gráficos de anomalias salvos.")

def mapa_anomalia_espacial(geotiff_dir, brasil_aoi, ano_analise, var_prefix, titulo, cmap, label_barra, output_dir):
    """
    Gera um mapa de anomalia espacial, mostrando onde um ano foi diferente da média do período.
    """
    print(f"[Análise Bônus] Gerando mapa de anomalia espacial para '{var_prefix}' em {ano_analise}...")
    
    files = sorted(glob.glob(os.path.join(geotiff_dir, f'{var_prefix}_*.tif')))
    if not files: return

    # 1. Calcula o raster da média de todo o período.
    with rasterio.open(files[0]) as src:
        soma = np.zeros(src.shape, dtype=np.float32); profile = src.profile
        
    for f in files:
        with rasterio.open(f) as src: soma += np.nan_to_num(src.read(1))
    media_periodo = soma / len(files)

    # 2. Calcula o raster de anomalia para o ano específico.
    ano_file = os.path.join(geotiff_dir, f'{var_prefix}_{ano_analise}.tif')
    with rasterio.open(ano_file) as src:
        dados_ano = src.read(1)
        anomalia = dados_ano - media_periodo
        anomalia[dados_ano < -200] = np.nan # Garante que 'no data' seja NaN.
        brasil_reproj = brasil_aoi.to_crs(src.crs)
        
        # 3. Plota o mapa.
        fig, ax = plt.subplots(figsize=(10, 10))
        lim = np.nanpercentile(np.abs(anomalia), 98) # Limite de cor para melhor visualização.
        img = ax.imshow(anomalia, cmap=cmap, vmin=-lim, vmax=lim) # `imshow` é mais simples para rasters.
        brasil_reproj.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)
        ax.set_title(titulo, fontsize=16); ax.set_axis_off()
        fig.colorbar(img, ax=ax, orientation='horizontal', label=label_barra, pad=0.01, shrink=0.7)
        plt.savefig(os.path.join(output_dir, f'mapa_anomalia_{var_prefix}_{ano_analise}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f" -> Mapa de anomalia para {ano_analise} salvo.")

def clusterizacao_climatica(geotiff_dir, brasil_aoi, output_geotiff_dir, output_plots_dir):
    """
    Aplica K-Means para agrupar o Brasil em zonas com características climáticas similares.
    """
    print("[Análise 4/7] Gerando clusterização de zonas climáticas...")
    
    temp_files = sorted(glob.glob(os.path.join(geotiff_dir, 'temperatura_media_*.tif')))
    rain_files = sorted(glob.glob(os.path.join(geotiff_dir, 'pluviosidade_total_*.tif')))
    if not temp_files or not rain_files: return

    # 1. Calcula o raster da média de temperatura para todo o período.
    with rasterio.open(temp_files[0]) as src:
        dest_profile = src.profile
        soma_temp = np.zeros(src.shape, dtype=np.float32)
    for f in temp_files:
        with rasterio.open(f) as src: soma_temp += np.nan_to_num(src.read(1))
    media_temp = soma_temp / len(temp_files)

    # 2. Calcula o raster da média de chuva e o reprojeta para a mesma grade da temperatura.
    #    Isso é ESSENCIAL para poder comparar os pixels de ambas as variáveis.
    media_rain_reprojected = np.empty_like(media_temp)
    with rasterio.open(rain_files[0]) as src_rain:
        soma_rain = np.zeros(src_rain.shape, dtype=np.float32)
        for f in rain_files:
            with rasterio.open(f) as src: soma_rain += np.nan_to_num(src.read(1))
        media_rain = soma_rain / len(rain_files)
        # Reprojeta a média de chuva para a grade da temperatura.
        reproject(source=media_rain, destination=media_rain_reprojected, src_transform=src_rain.transform, src_crs=src_rain.crs,
                  dst_transform=dest_profile['transform'], dst_crs=dest_profile['crs'], resampling=Resampling.bilinear)
            
    # 3. Prepara os dados para o Machine Learning.
    valid_mask = (media_temp > -50) & (media_rain_reprojected > 0) # Ignora pixels sem dados.
    # Empilha as duas variáveis (temperatura e chuva) como features.
    features = np.vstack((media_temp[valid_mask], media_rain_reprojected[valid_mask])).T
    # Normaliza os dados. Isso é CRÍTICO para K-Means, para que uma variável com escala maior (chuva) não domine o cálculo da distância.
    features_scaled = StandardScaler().fit_transform(features)
    
    # 4. Roda o K-Means para encontrar 5 clusters (zonas).
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto').fit(features_scaled)
    
    # 5. Cria o mapa de clusters.
    cluster_map = np.full(media_temp.shape, -1, dtype=float)
    cluster_map[valid_mask] = kmeans.labels_
    cluster_map[~valid_mask] = np.nan

    # Salva o mapa de zonas climáticas como um GeoTIFF.
    dest_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(os.path.join(output_geotiff_dir, 'mapa_zonas_climaticas.tif'), 'w', **dest_profile) as dst:
        dst.write(cluster_map, 1)
        
    # 6. Analisa e imprime as características de cada zona.
    cluster_stats = []
    for i in range(kmeans.n_clusters):
        cluster_mask = (cluster_map == i)
        temp_in_cluster = media_temp[cluster_mask]; rain_in_cluster = media_rain_reprojected[cluster_mask]
        cluster_stats.append({'Cluster': f'Zona {i}', 'Temp. Média (°C)': np.mean(temp_in_cluster), 'Chuva Média (mm)': np.mean(rain_in_cluster)})
    df_clusters = pd.DataFrame(cluster_stats).set_index('Cluster')
    print("\nCaracterísticas médias das zonas climáticas encontradas:\n", df_clusters)

    # 7. Gera um gráfico combinado com o mapa e as estatísticas das zonas.
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2.5, 1]})
    show(cluster_map, ax=ax_map, cmap='viridis'); brasil_aoi.to_crs(dest_profile['crs']).plot(ax=ax_map, facecolor='none', edgecolor='black', linewidth=0.5)
    ax_map.set_title("Zonas Climáticas do Brasil (K-Means)", fontsize=16); ax_map.set_axis_off()
    df_clusters.plot(kind='bar', ax=ax_bar, subplots=True, legend=False, rot=0, layout=(2,1), sharex=True)
    ax_bar.get_figure().get_axes()[0].set_ylabel("Temp. Média (°C)"); ax_bar.get_figure().get_axes()[1].set_ylabel("Chuva Média (mm)");
    plt.suptitle("Clusterização Climática e Características das Zonas", fontsize=18); plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_plots_dir, 'mapa_e_stats_zonas_climaticas.png'), dpi=300); plt.close(fig)
    print(" -> Mapa e estatísticas de zonas climáticas salvos.")

def analise_climatica_por_estado(geotiff_dir, estados_aoi, output_csv_path):
    """
    Calcula e salva as estatísticas climáticas anuais para cada estado do Brasil.
    """
    print("[Análise 5/7] Gerando estatísticas climáticas por estado...")
    temp_files = sorted(glob.glob(os.path.join(geotiff_dir, 'temperatura_media_*.tif')))
    anos = [os.path.basename(f).split('_')[-1].replace('.tif', '') for f in temp_files]
    data_por_estado = []
    # Loop pelos anos
    for ano in tqdm(anos, desc="Processando anos para cada estado"):
        temp_file = os.path.join(geotiff_dir, f'temperatura_media_{ano}.tif'); rain_file = os.path.join(geotiff_dir, f'pluviosidade_total_{ano}.tif')
        with rasterio.open(temp_file) as src_temp, rasterio.open(rain_file) as src_rain:
            estados_temp_reproj = estados_aoi.to_crs(src_temp.crs)
            estados_rain_reproj = estados_aoi.to_crs(src_rain.crs)
            # Loop pelos estados
            for index, estado in estados_temp_reproj.iterrows():
                geom = [estado.geometry]; sigla = estado['SIGLA_UF']
                # Extrai a temperatura para o estado
                temp_data, _ = mask(src_temp, geom, crop=True, nodata=np.nan)
                temp_media = np.nanmean(temp_data[temp_data > -100]) if not np.all(np.isnan(temp_data)) else np.nan
                # Extrai a pluviosidade para o estado
                rain_data, _ = mask(src_rain, [estados_rain_reproj.iloc[index].geometry], crop=True, nodata=np.nan)
                chuva_media = np.nanmean(rain_data[rain_data > 0]) if not np.all(np.isnan(rain_data)) else np.nan
                data_por_estado.append({'Ano': int(ano), 'Estado': estado['NM_UF'], 'Sigla': sigla, 'Temp_Media_C': temp_media, 'Chuva_Media_Anual_mm': chuva_media})
    df_estados = pd.DataFrame(data_por_estado)
    df_estados.to_csv(output_csv_path, index=False, decimal=',', sep=';')
    print(f" -> Estatísticas por estado salvas."); return df_estados

# --- 4. BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    
    # Carrega os shapefiles necessários.
    try:
        brasil_aoi = gpd.read_file(os.path.join("./natural_earth_110m_countries/", "ne_110m_admin_0_countries.shp"))
        brasil_aoi = brasil_aoi[brasil_aoi['ADMIN'] == 'Brazil']
        estados_aoi = gpd.read_file(os.path.join("./BR_UF_2022/", "BR_UF_2022.shp"))
    except Exception as e: print(f"Erro ao carregar shapefiles: {e}."); exit()

    # ETAPA CRUCIAL: Reprojeta a geometria do Brasil para o CRS de cada tipo de dado.
    # Isso é necessário para que a função `mask` funcione corretamente.
    brasil_aoi_reprojected = {}
    with rasterio.open(glob.glob(os.path.join(INPUT_GEOTIFF_DIR, 'temperatura*.tif'))[0]) as src:
        brasil_aoi_reprojected['temp'] = brasil_aoi.to_crs(src.crs)
    with rasterio.open(glob.glob(os.path.join(INPUT_GEOTIFF_DIR, 'pluviosidade*.tif'))[0]) as src:
        brasil_aoi_reprojected['rain'] = brasil_aoi.to_crs(src.crs)

    # Executa a cadeia de análises na ordem correta.
    df_climate_stats = gerar_estatisticas_climaticas(INPUT_GEOTIFF_DIR, brasil_aoi_reprojected, os.path.join(OUTPUT_CSV_DIR, 'estatisticas_climaticas_anuais.csv'))
    analise_tendencia_climatica(df_climate_stats, OUTPUT_PLOTS_DIR)
    analise_anomalias_climaticas(df_climate_stats, OUTPUT_PLOTS_DIR)
    
    # Exemplo de mapa de anomalia espacial para o ano de 2023.
    ano_recente = '2023' # Mude aqui para analisar outro ano.
    mapa_anomalia_espacial(INPUT_GEOTIFF_DIR, brasil_aoi, ano_recente, 'temperatura_media', f'Anomalia de Temperatura Média em {ano_recente} vs Média do Período', 'coolwarm', 'Diferença de Temp. (°C)', OUTPUT_PLOTS_DIR)
    mapa_anomalia_espacial(INPUT_GEOTIFF_DIR, brasil_aoi, ano_recente, 'pluviosidade_total', f'Anomalia de Pluviosidade Total em {ano_recente} vs Média do Período', 'BrBG', 'Diferença de Chuva (mm)', OUTPUT_PLOTS_DIR)
    
    clusterizacao_climatica(INPUT_GEOTIFF_DIR, brasil_aoi, OUTPUT_GEOTIFF_DIR, OUTPUT_PLOTS_DIR)
    analise_climatica_por_estado(INPUT_GEOTIFF_DIR, estados_aoi, os.path.join(OUTPUT_CSV_DIR, 'clima_anual_por_estado.csv'))

    print("\n--- Análises climáticas concluídas com sucesso! Verifique a pasta 'analises_climaticas'. ---")