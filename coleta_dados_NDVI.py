# =================================================================================================
# SCRIPT 1: coleta_dados_NDVI.py
#
# OBJETIVO:
# Este script é o ponto de partida do projeto. Sua principal função é baixar, processar e
# realizar uma análise preliminar de dados de NDVI (Índice de Vegetação por Diferença Normalizada)
# do produto MOD13A2 do sensor MODIS para todo o território brasileiro.
#
# PASSOS EXECUTADOS:
# 1.  Baixa os shapefiles do contorno do Brasil e de seus estados, que servirão como Área de Interesse (AOI).
# 2.  Autentica-se na plataforma Earthdata da NASA.
# 3.  Para cada ano especificado, busca e baixa os arquivos HDF do produto MOD13A2 que cobrem o Brasil.
# 4.  Para cada ano, processa os arquivos baixados:
#       a. Extrai a camada de dados específica de NDVI de cada arquivo.
#       b. Recorta (mascara) cada camada usando o contorno do Brasil.
#       c. Combina (faz um mosaico) todas as camadas recortadas para criar uma única imagem de NDVI para o Brasil.
# 5.  Salva os mosaicos anuais como arquivos GeoTIFF.
# 6.  Realiza um conjunto de análises iniciais com os dados processados (estatísticas, tendências, mapas).
# =================================================================================================

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
# Importa as ferramentas necessárias para download, processamento geoespacial, análise de dados e visualização.

import earthaccess                      # Biblioteca principal para buscar, baixar e acessar dados da NASA.
import geopandas as gpd                 # Para trabalhar com dados vetoriais (shapefiles) e manipulações geoespaciais.
import rasterio                         # Biblioteca essencial para ler, escrever e manipular dados raster (imagens de satélite como GeoTIFF).
from rasterio.mask import mask          # Função para recortar um raster com base em uma geometria vetorial (shape).
from rasterio.merge import merge        # Função para combinar múltiplos rasters em um único mosaico.
from rasterio.plot import show          # Função para exibir dados raster em gráficos.
import matplotlib.pyplot as plt         # Biblioteca para criar gráficos e visualizações estáticas.
import numpy as np                      # Para computação numérica, especialmente para trabalhar com os arrays (pixels) das imagens.
import os                               # Para interagir com o sistema operacional (criar pastas, verificar caminhos).
import urllib.request                   # Para fazer requisições web e baixar arquivos (como os shapefiles).
import zipfile                          # Para descompactar arquivos .zip.
from datetime import datetime           # Para trabalhar com datas (não usado diretamente, mas bom para referência temporal).
from shapely.geometry import mapping    # Para converter geometrias do GeoPandas para um formato que o Rasterio entende.
import pandas as pd                     # Para criar e manipular tabelas de dados (DataFrames), usadas para as estatísticas.
from tqdm import tqdm                   # Para criar barras de progresso visuais em loops demorados.
import warnings                         # Para controlar a exibição de avisos do sistema.
from scipy import stats                 # Biblioteca científica, usada aqui para calcular a regressão linear (tendência).

# --- 2. CONFIGURAÇÕES INICIAIS ---

# Suprime um aviso específico do Rasterio que ocorre ao abrir arquivos HDF.
# O HDF é um formato contêiner, e o arquivo principal não tem georreferenciamento, apenas os subdatasets.
# Isso é esperado e podemos ignorar o aviso para manter a saída limpa.
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Define um estilo visual mais agradável para os gráficos gerados pelo Matplotlib.
plt.style.use('seaborn-v0_8-whitegrid')
# Altera o "backend" de plotagem para 'Agg'. Isso impede que o Matplotlib tente abrir janelas de interface gráfica,
# o que é crucial para rodar o script em servidores ou ambientes sem monitor, salvando os gráficos diretamente em arquivo.
plt.switch_backend('Agg')

# ======================= 3. FUNÇÕES DE SETUP E PROCESSAMENTO DE DADOS =======================

def carregar_aoi(tipo='pais'):
    """
    Carrega a Área de Interesse (AOI) a partir de um shapefile.
    Se o arquivo não existir localmente, ele será baixado da web.

    Args:
        tipo (str): O tipo de AOI a ser carregado. 'pais' para o Brasil, 'estados' para os estados brasileiros.

    Returns:
        gpd.GeoDataFrame: Um GeoDataFrame contendo a geometria da AOI.
    """
    if tipo == 'pais':
        print("Definindo a Área de Interesse (AOI) - Brasil...")
        # Fonte para o contorno dos países
        shapefile_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        zip_path = "./ne_110m_admin_0_countries.zip"
        shapefile_dir = "./natural_earth_110m_countries/"
        shapefile_path = os.path.join(shapefile_dir, "ne_110m_admin_0_countries.shp")
        nome_unico = 'Brazil'
        coluna_filtro = 'ADMIN'
    elif tipo == 'estados':
        print("Definindo a Área de Interesse (AOI) - Estados do Brasil...")
        # Fonte oficial e atualizada do IBGE para os limites estaduais
        shapefile_url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/Brasil/BR/BR_UF_2022.zip"
        zip_path = "./BR_UF_2022.zip"
        shapefile_dir = "./BR_UF_2022/"
        shapefile_path = os.path.join(shapefile_dir, "BR_UF_2022.shp")
    else:
        print("Erro: Tipo de AOI inválido. Escolha 'pais' ou 'estados'.")
        return None

    # Verifica se o shapefile já existe. Se não, baixa e descompacta.
    if not os.path.exists(shapefile_path):
        print(f"Shapefile '{os.path.basename(shapefile_path)}' não encontrado. Baixando...")
        # Define um User-Agent para simular um navegador e evitar bloqueios por erro 403 (Forbidden).
        headers = {'User-Agent': 'Mozilla/5.0'}
        request = urllib.request.Request(shapefile_url, headers=headers)
        try:
            # Baixa o arquivo .zip
            with urllib.request.urlopen(request) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            # Extrai o conteúdo do .zip para a pasta de destino
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(shapefile_dir)
            print("Download e descompactação concluídos.")
            # Remove o arquivo .zip baixado para limpar o diretório.
            os.remove(zip_path)
        except urllib.error.URLError as e:
            print(f"Erro de URL ao tentar baixar o arquivo: {e.reason}")
            return None
    else:
        print("Shapefile já existe localmente.")

    # Lê o shapefile para um GeoDataFrame.
    gdf = gpd.read_file(shapefile_path)
    if tipo == 'pais':
        # Filtra o GeoDataFrame para manter apenas a geometria do Brasil.
        aoi = gdf[gdf[coluna_filtro] == nome_unico]
        if aoi.empty:
            print(f"Erro: Não foi possível encontrar '{nome_unico}' no shapefile.")
            return None
        print("Shapefile do Brasil carregado.")
        return aoi
    
    print("Shapefile dos estados carregado.")
    return gdf


def processar_modis_ndvi(brasil_aoi, years):
    """
    Baixa e processa os dados de NDVI do MODIS para os anos especificados.

    Args:
        brasil_aoi (gpd.GeoDataFrame): GeoDataFrame com o contorno do Brasil.
        years (range): Um intervalo de anos para processar (ex: range(2019, 2025)).

    Returns:
        list: Uma lista de dicionários, onde cada dicionário contém os dados NDVI processados
              para um ano ('year', 'ndvi', 'transform', 'crs').
    """
    # Etapa de Autenticação na NASA
    try:
        # Tenta fazer login usando credenciais salvas em cache (.netrc) ou de forma interativa.
        auth = earthaccess.login(strategy="interactive", persist=True)
        if not auth.authenticated:
            print("Falha no login no Earthdata. Verifique seu arquivo .netrc ou credenciais interativas.")
            return []
    except Exception as e:
        print(f"Erro ao tentar fazer login no Earthdata: {e}")
        return []

    all_data = [] # Lista para armazenar os dados processados de todos os anos.
    # Itera sobre cada ano com uma barra de progresso.
    for year in tqdm(years, desc="Processando Anos"):
        # Obtém as coordenadas da caixa delimitadora (bounding box) do Brasil.
        minx, miny, maxx, maxy = brasil_aoi.total_bounds
        bounding_box = (minx, miny, maxx, maxy)

        try:
            # Busca os dados no portal da NASA
            results = earthaccess.search_data(
                short_name='MOD13A2',      # Nome curto do produto MODIS NDVI de 16 dias com 1km de resolução.
                bounding_box=bounding_box, # Filtro espacial.
                temporal=(f"{year}-01-01", f"{year}-12-31"), # Filtro temporal.
            )
            print(f"\n[ANO {year}] Foram encontrados {len(results)} arquivos. Iniciando download e processamento...")
            if not results:
                print(f"Aviso: Nenhum dado MODIS NDVI encontrado para o ano {year}.")
                continue

            # Baixa os arquivos encontrados para um diretório local.
            local_dir = f"./dados_satelite/modis_{year}"
            os.makedirs(local_dir, exist_ok=True)
            filepaths = earthaccess.download(results, local_path=local_dir)

            srcs_to_mosaic = [] # Lista para armazenar os rasters abertos em memória, prontos para o mosaico.
            # Itera sobre cada arquivo HDF baixado para o ano atual.
            for hdf_file in tqdm(filepaths, desc=f"Processando arquivos de {year}", leave=False):
                try:
                    # 1. Abre o arquivo HDF e identifica o caminho para o subdataset de NDVI.
                    with rasterio.open(hdf_file) as src:
                        # O nome do subdataset de NDVI geralmente termina com "1 km 16 days NDVI".
                        ndvi_sds_path = [s for s in src.subdatasets if 'NDVI' in s.split(':')[-1]][0]

                    # 2. Abre o subdataset de NDVI.
                    with rasterio.open(ndvi_sds_path) as src_ndvi:
                        # 3. Reprojeta a AOI do Brasil para o mesmo sistema de coordenadas (CRS) do raster.
                        brasil_reprojetado = brasil_aoi.to_crs(src_ndvi.crs)
                        # 4. Recorta (mascara) o raster usando a geometria do Brasil.
                        out_image, out_transform = mask(src_ndvi, [mapping(geom) for geom in brasil_reprojetado.geometry], crop=True, all_touched=True)
                        
                        # Se a imagem resultante do recorte não tiver dados válidos, pula para a próxima.
                        if np.all(out_image == src_ndvi.nodata): continue

                        # 5. Salva o raster recortado em um arquivo em memória para evitar I/O de disco.
                        out_meta = src_ndvi.meta.copy()
                        out_meta.update({"driver": "MEM", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})

                        memfile = rasterio.io.MemoryFile().open(**out_meta)
                        memfile.write(out_image)
                        srcs_to_mosaic.append(memfile) # Adiciona o arquivo em memória à lista para o mosaico.
                except IndexError:
                     print(f"Aviso: Nenhum subdataset NDVI encontrado em {os.path.basename(hdf_file)}. Pulando.")
                except Exception as e:
                    print(f"Erro ao processar {os.path.basename(hdf_file)}: {str(e)}. Pulando.")

            # Se houver rasters válidos para combinar
            if srcs_to_mosaic:
                print(f"Combinando {len(srcs_to_mosaic)} tiles em um mosaico para o ano {year}...")
                # 6. Combina todos os rasters em memória em um único mosaico.
                # O método 'max' é ideal para compósitos de NDVI, pois seleciona o pixel com o maior valor
                # de NDVI ao longo do tempo, o que geralmente representa a vegetação mais saudável e com menos nuvens.
                mosaic, out_trans = merge(srcs_to_mosaic, method='max')
                
                # 7. Converte os valores brutos para o valor real de NDVI.
                ndvi_raw = mosaic[0].astype(float)
                ndvi_raw[ndvi_raw == -3000] = np.nan # Substitui o valor 'no data' por NaN.
                ndvi = ndvi_raw * 0.0001 # Aplica o fator de escala do produto MODIS.
                
                # Armazena os dados processados para o ano.
                all_data.append({'year': year, 'ndvi': ndvi, 'transform': out_trans, 'crs': srcs_to_mosaic[0].crs})
                print(f"Mosaico para o ano {year} processado com sucesso.")
                # Fecha todos os arquivos em memória para liberar recursos.
                for src in srcs_to_mosaic: src.close()
            else:
                print(f"Nenhum dado válido foi processado para o ano {year}.")
        except Exception as e:
            print(f"Erro geral ao processar o ano {year}: {str(e)}")

    return all_data


# ============================= 4. FUNÇÕES DE ANÁLISE =============================
# Estas funções pegam os dados processados e geram os produtos finais: CSVs, gráficos e mapas.

def analise_estatisticas_detalhadas(all_data, output_dir):
    """Calcula estatísticas descritivas para o NDVI de cada ano e salva em um CSV."""
    print("\n[Análise 1/5] Gerando estatísticas detalhadas anuais...")
    df_data = []
    for data in all_data:
        ndvi = data['ndvi']
        # Calcula as estatísticas usando as funções `nan*` do NumPy, que ignoram valores NaN.
        df_data.append({
            'Ano': data['year'], 'NDVI Médio': np.nanmean(ndvi), 'NDVI Mediana': np.nanmedian(ndvi),
            'Desvio Padrão': np.nanstd(ndvi), 'NDVI Máximo': np.nanmax(ndvi), 'NDVI Mínimo': np.nanmin(ndvi)
        })
    df_stats = pd.DataFrame(df_data)
    # Salva o DataFrame em um arquivo CSV formatado para o padrão brasileiro.
    output_path = os.path.join(output_dir, 'csv_data', 'estatisticas_anuais_ndvi.csv')
    df_stats.to_csv(output_path, index=False, decimal=',', sep=';')
    print(f" -> Estatísticas salvas em: {output_path}")
    return df_stats


def analise_tendencia_geral(df_stats, output_dir):
    """Calcula e plota a tendência linear do NDVI ao longo dos anos."""
    print("\n[Análise 2/5] Gerando análise de tendência temporal...")
    anos, ndvi_medio = df_stats['Ano'], df_stats['NDVI Médio']
    # Usa a função `linregress` da SciPy para obter os parâmetros da regressão linear.
    slope, intercept, r_value, p_value, std_err = stats.linregress(anos, ndvi_medio)
    
    # Cria o gráfico de tendência.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(anos, ndvi_medio, marker='o', linestyle='-', label='NDVI Médio Anual')
    ax.plot(anos, intercept + slope*anos, 'r', label=f'Linha de Tendência (R²={r_value**2:.2f})')
    ax.set_title('Tendência do NDVI Médio para o Brasil', fontsize=16)
    ax.set_xlabel('Ano'); ax.set_ylabel('NDVI Médio'); ax.legend(); ax.grid(True)
    
    output_path_png = os.path.join(output_dir, 'plots', 'tendencia_ndvi_brasil.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Gráfico de tendência salvo em: {output_path_png}")
    
    # Cria um relatório de texto com a interpretação dos resultados estatísticos.
    report = (f"Análise de Tendência do NDVI (Regressão Linear)\n--------------------------------------------------\n"
              f"Período: {anos.min()} - {anos.max()}\nSlope (inclinação): {slope:.6f}\n"
              f"  - Um slope positivo indica uma tendência de aumento da vegetação (greening).\n"
              f"  - Um slope negativo indica uma tendência de diminuição da vegetação (browning).\n"
              f"P-valor: {p_value:.4f}\n  - Geralmente, um p-valor < 0.05 indica que a tendência é estatisticamente significativa.\n"
              f"R-quadrado: {r_value**2:.4f}\n  - Indica a proporção da variância no NDVI que é explicada pela passagem do tempo.\n")
    output_path_txt = os.path.join(output_dir, 'csv_data', 'relatorio_tendencia.txt')
    with open(output_path_txt, 'w', encoding='utf-8') as f: f.write(report)
    print(f" -> Relatório de tendência salvo em: {output_path_txt}")


def analise_anomalias_anuais(df_stats, output_dir):
    """Cria um gráfico de barras mostrando o desvio do NDVI de cada ano em relação à média geral."""
    print("\n[Análise 3/5] Gerando gráfico de anomalias anuais...")
    overall_mean = df_stats['NDVI Médio'].mean()
    df_stats['Anomalia'] = df_stats['NDVI Médio'] - overall_mean
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Colore as barras: verde se a anomalia for positiva (ano mais verde que a média), vermelho se for negativa.
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in df_stats['Anomalia']]
    ax.bar(df_stats['Ano'], df_stats['Anomalia'], color=colors)
    ax.set_title('Anomalia do NDVI Anual em Relação à Média do Período', fontsize=16)
    ax.set_xlabel('Ano'); ax.set_ylabel('Diferença da Média'); ax.axhline(0, color='grey', lw=1)
    
    output_path = os.path.join(output_dir, 'plots', 'anomalias_ndvi_anuais.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Gráfico de anomalias salvo em: {output_path}")


def analise_mapa_de_diferenca(all_data, brasil_aoi, output_dir):
    """Cria um mapa que mostra a mudança no NDVI (pixel a pixel) entre o primeiro e o último ano."""
    print("\n[Análise 4/5] Gerando mapa de diferença (change detection)...")
    if len(all_data) < 2:
        print(" -> Análise de diferença requer pelo menos dois anos de dados. Pulando.")
        return

    primeiro_ano, ultimo_ano = all_data[0], all_data[-1]
    
    # É preciso garantir que os dois rasters estejam perfeitamente alinhados (mesma projeção, transformação e tamanho).
    # A função `reproject` do rasterio faz isso, alinhando o raster do primeiro ano ao do último.
    ndvi_primeiro_reproj = np.empty(shape=ultimo_ano['ndvi'].shape, dtype=rasterio.float32)
    rasterio.warp.reproject(
        source=primeiro_ano['ndvi'], destination=ndvi_primeiro_reproj, src_transform=primeiro_ano['transform'],
        src_crs=primeiro_ano['crs'], dst_transform=ultimo_ano['transform'], dst_crs=ultimo_ano['crs'],
        resampling=rasterio.warp.Resampling.bilinear)
    # Calcula a diferença pixel a pixel.
    diferenca_ndvi = ultimo_ano['ndvi'] - ndvi_primeiro_reproj
    
    # Salva o resultado como um novo GeoTIFF.
    out_meta = {'driver': 'GTiff', 'height': diferenca_ndvi.shape[0], 'width': diferenca_ndvi.shape[1], 'count': 1, 
                'dtype': str(diferenca_ndvi.dtype), 'crs': ultimo_ano['crs'], 'transform': ultimo_ano['transform'], 'nodata': np.nan}
    output_path_geotiff = os.path.join(output_dir, 'geotiff_maps', f'diferenca_ndvi_{primeiro_ano["year"]}-{ultimo_ano["year"]}.tif')
    with rasterio.open(output_path_geotiff, 'w', **out_meta) as dst: dst.write(diferenca_ndvi, 1)
    print(f" -> GeoTIFF de diferença salvo em: {output_path_geotiff}")
    
    # Cria o mapa visual.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Define os limites de cor com base no 95º percentil para evitar que valores extremos dominem a paleta.
    lim = np.nanpercentile(np.abs(diferenca_ndvi), 95)
    # Usa um mapa de cores divergente (RdBu): azul para aumento de NDVI, vermelho para diminuição.
    show(diferenca_ndvi, transform=ultimo_ano['transform'], ax=ax, cmap='RdBu', vmin=-lim, vmax=lim)
    brasil_aoi.to_crs(ultimo_ano['crs']).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    ax.set_title(f'Diferença de NDVI: {ultimo_ano["year"]} vs {primeiro_ano["year"]}', fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
    output_path_png = os.path.join(output_dir, 'plots', f'mapa_diferenca_ndvi_{primeiro_ano["year"]}-{ultimo_ano["year"]}.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Mapa de diferença salvo em: {output_path_png}")


def analise_regional_por_estado(all_data, estados_aoi, output_dir):
    """Calcula o NDVI médio para cada estado, para cada ano, e gera produtos visuais."""
    print("\n[Análise 5/5] Iniciando análise regional por estado (pode levar alguns minutos)...")
    if estados_aoi is None:
        print(" -> Shapefile dos estados não foi carregado. Pulando análise regional.")
        return
        
    resultados_estados = []
    # Loop pelos anos.
    for data in tqdm(all_data, desc="Processando anos por estado"):
        year, ndvi_raster, transform, crs = data['year'], data['ndvi'], data['transform'], data['crs']
        estados_reprojetado = estados_aoi.to_crs(crs)
        
        # Identifica os nomes das colunas de sigla e nome no shapefile do IBGE.
        sigla_col = 'SIGLA_UF' if 'SIGLA_UF' in estados_reprojetado.columns else 'SIGLA'
        nome_col = 'NM_UF' if 'NM_UF' in estados_reprojetado.columns else 'NM_ESTADO'

        # Loop pelos estados.
        for _, estado in tqdm(estados_reprojetado.iterrows(), total=estados_reprojetado.shape[0], desc=f" {year} ", leave=False):
            try:
                # Recorta o raster nacional usando a geometria de cada estado.
                out_image, _ = mask(rasterio.io.MemoryFile().write(ndvi_raster, 1, transform=transform, crs=crs), [mapping(estado.geometry)], crop=True, all_touched=True)
                
                # Calcula a média do NDVI para a área do estado.
                mean_ndvi_estado = np.nanmean(out_image[out_image > -9999])
                if not np.isnan(mean_ndvi_estado):
                    resultados_estados.append({'Ano': year, 'Estado': estado[nome_col], 'Sigla': estado[sigla_col], 'NDVI_Medio': mean_ndvi_estado})
            except Exception: pass # Ignora estados que possam causar erros (ex: sem sobreposição com os dados).

    if not resultados_estados:
        print(" -> Nenhum resultado para a análise de estados."); return

    # Salva os resultados em um CSV.
    df_estados = pd.DataFrame(resultados_estados)
    output_path_csv = os.path.join(output_dir, 'csv_data', 'ndvi_medio_por_estado.csv')
    df_estados.to_csv(output_path_csv, index=False, decimal=',', sep=';', encoding='utf-8-sig')
    print(f" -> Dados de NDVI por estado salvos em: {output_path_csv}")

    # Gera um gráfico de linha para alguns estados selecionados.
    estados_selecionados = ['AM', 'MT', 'SP', 'BA', 'RS'] # Estados de diferentes biomas para comparação.
    df_plot = df_estados[df_estados['Sigla'].isin(estados_selecionados)]
    fig, ax = plt.subplots(figsize=(12, 7))
    for estado, group in df_plot.groupby('Estado'): ax.plot(group['Ano'], group['NDVI_Medio'], marker='o', linestyle='-', label=estado)
    ax.set_title('Tendência do NDVI Médio para Estados Selecionados', fontsize=16)
    ax.set_xlabel('Ano'); ax.set_ylabel('NDVI Médio'); ax.legend(title='Estado'); ax.grid(True)
    output_path_plot = os.path.join(output_dir, 'plots', 'tendencia_ndvi_estados_selecionados.png')
    plt.savefig(output_path_plot, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f" -> Gráfico de tendência por estado salvo em: {output_path_plot}")

    # Gera um mapa coroplético (de cores) do NDVI por estado para o último ano.
    df_ultimo_ano = df_estados[df_estados['Ano'] == df_estados['Ano'].max()]
    mapa_plot = pd.merge(estados_aoi, df_ultimo_ano, left_on='SIGLA_UF', right_on='Sigla')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    mapa_plot.plot(column='NDVI_Medio', cmap='RdYlGn', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                   legend_kwds={'label': "NDVI Médio", 'orientation': "horizontal"})
    ax.set_title(f'NDVI Médio por Estado - {df_estados["Ano"].max()}', fontsize=16); ax.set_axis_off()
    output_path_mapa = os.path.join(output_dir, 'plots', f'mapa_ndvi_estados_{df_estados["Ano"].max()}.png')
    plt.savefig(output_path_mapa, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f" -> Mapa de NDVI por estado salvo em: {output_path_mapa}")


# ================================ 5. FUNÇÃO PRINCIPAL (MAIN) ================================

if __name__ == "__main__":
    # --- Setup Inicial ---
    # Define e cria as pastas de saída para organizar os resultados.
    output_dir_base = "resultados"
    subfolders = ['plots', 'csv_data', 'geotiff_maps']
    for folder in subfolders:
        path = os.path.join(output_dir_base, folder)
        os.makedirs(path, exist_ok=True)

    # Carrega as geometrias do Brasil e dos estados.
    brasil_aoi = carregar_aoi('pais')
    estados_aoi = carregar_aoi('estados')
    if brasil_aoi is None:
        print("Não foi possível carregar a AOI do Brasil. Encerrando o script.")
        exit()
    
    # --- Execução do Processamento ---
    # Define o período de interesse.
    years = range(2019, 2025)
    # Chama a função principal de download e processamento.
    all_data = processar_modis_ndvi(brasil_aoi, years)

    # --- Execução das Análises ---
    if all_data:
        print("\n--- INICIANDO ANÁLISES COM OS DADOS PROCESSADOS ---")
        # Salva os mosaicos anuais em disco como arquivos GeoTIFF.
        print("Salvando mosaicos anuais em formato GeoTIFF...")
        for data in tqdm(all_data, desc="Salvando GeoTIFFs"):
            path = os.path.join(output_dir_base, 'geotiff_maps', f'mosaico_ndvi_brasil_{data["year"]}.tif')
            with rasterio.open(path, 'w', driver='GTiff', height=data['ndvi'].shape[0], width=data['ndvi'].shape[1],
                count=1, dtype=str(data['ndvi'].dtype), crs=data['crs'], transform=data['transform'], nodata=np.nan) as dst:
                dst.write(data['ndvi'], 1)
        
        # Chama sequencialmente todas as funções de análise.
        df_stats = analise_estatisticas_detalhadas(all_data, output_dir_base)
        analise_tendencia_geral(df_stats, output_dir_base)
        analise_anomalias_anuais(df_stats, output_dir_base)
        analise_mapa_de_diferenca(all_data, brasil_aoi, output_dir_base)
        analise_regional_por_estado(all_data, estados_aoi, output_dir_base)

        print("\n--- Processamento e análises concluídos com sucesso! Verifique a pasta 'resultados'. ---")
    else:
        print("\nNenhum dado foi processado com sucesso. Verifique os logs de erro. Encerrando o script.")