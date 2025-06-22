# =================================================================================================
# SCRIPT 5: processa_dados_climaticos.py
#
# OBJETIVO:
# Este script é a etapa de processamento dos dados climáticos brutos baixados pelo SCRIPT 3.
# Ele lê os arquivos de Temperatura (MODIS LST, formato HDF4) e Pluviosidade (GPM IMERG, formato HDF5),
# realiza os cálculos necessários e gera dois produtos principais:
#
# 1.  Mosaicos GeoTIFF de TEMPERATURA MÉDIA ANUAL em graus Celsius.
# 2.  Mosaicos GeoTIFF de PLUVIOSIDADE TOTAL ANUAL em milímetros.
#
# COMO FUNCIONA:
# - Para a Temperatura:
#   1. Agrupa os arquivos HDF por ano.
#   2. Para cada ano, primeiro calcula a extensão geográfica total de todos os "tiles" (cenas).
#   3. Em seguida, para cada período de 8 dias, cria um mosaico de todos os tiles.
#   4. Converte os valores de Kelvin para Celsius e acumula a soma e a contagem de observações válidas para cada pixel.
#   5. Ao final do ano, calcula a média dividindo a soma pela contagem e salva o resultado como um GeoTIFF.
#
# - Para a Pluviosidade:
#   1. Agrupa os arquivos HDF5 por ano.
#   2. Para cada ano, itera sobre os arquivos mensais.
#   3. Lê os dados de precipitação (taxa em mm/hora) de cada arquivo.
#   4. Converte a taxa para o total mensal (taxa * horas no mês).
#   5. Acumula a soma dos totais mensais para cada pixel.
#   6. Ao final do ano, salva a soma total como um GeoTIFF com georreferenciamento padrão.
# =================================================================================================

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---

import rasterio                         # Para ler, escrever e manipular dados raster (GeoTIFF, e subdatasets de HDF).
from rasterio.merge import merge        # Função específica para combinar múltiplos rasters em um único mosaico.
import numpy as np                      # Para computação numérica e manipulação de arrays (os pixels das imagens).
import glob                             # Para encontrar arquivos que correspondem a um padrão (ex: todos os .hdf em uma pasta).
import os                               # Para interagir com o sistema operacional (criar pastas, manipular caminhos).
from tqdm import tqdm                   # Para criar barras de progresso visuais em loops demorados.
import geopandas as gpd                 # Usado aqui para carregar a área de interesse (shapefile do Brasil).
import re                               # Módulo de Expressões Regulares, usado para extrair informações (como o ano) dos nomes dos arquivos.
import h5py                             # Biblioteca para ler e escrever arquivos no formato HDF5 (usado pelo GPM IMERG).

# --- 2. FUNÇÃO PARA PROCESSAR TEMPERATURA (MODIS LST) ---

def processar_temperatura_anual(input_dir, output_dir, brasil_aoi):
    """
    Processa os dados brutos de MODIS LST (HDF4) para gerar mosaicos de temperatura média anual em Celsius.

    Args:
        input_dir (str): Pasta contendo os arquivos .hdf de temperatura baixados.
        output_dir (str): Pasta de destino para salvar os GeoTIFFs anuais processados.
        brasil_aoi (gpd.GeoDataFrame): GeoDataFrame do Brasil (usado para referência, não para recorte aqui).
    """
    print("\n--- Processando Temperatura Média Anual (LST) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Encontra todos os arquivos .hdf no diretório de entrada.
    files = sorted(glob.glob(os.path.join(input_dir, '*.hdf')))
    if not files:
        print("Aviso: Nenhum arquivo de temperatura (.hdf) encontrado.")
        return

    # Agrupa os caminhos dos arquivos por ano usando um dicionário.
    files_por_ano = {}
    for f in files:
        # Usa uma expressão regular para encontrar o padrão '.A<ano>' no nome do arquivo.
        match = re.search(r'\.A(\d{4})', f)
        if match:
            ano = match.group(1)
            if ano not in files_por_ano:
                files_por_ano[ano] = []
            files_por_ano[ano].append(f)

    # Itera sobre cada ano que possui arquivos.
    for ano, ano_files in files_por_ano.items():
        output_path = os.path.join(output_dir, f'temperatura_media_{ano}.tif')
        # Verifica se o arquivo final já foi processado para permitir a retomada do script.
        if os.path.exists(output_path):
            print(f"Arquivo de temperatura para o ano {ano} já existe. Pulando.")
            continue
        
        print(f"Processando ano: {ano} ({len(ano_files)} arquivos)")
        
        soma_anual = None      # Array para acumular a soma das temperaturas de cada pixel.
        contagem_valida = None # Array para contar quantas observações válidas cada pixel teve.
        profile = None         # Dicionário para guardar os metadados do GeoTIFF de saída.

        # Etapa 1: Calcular a extensão geográfica total (bounding box) de todos os tiles do ano.
        # Isso é crucial para criar um "canvas" de mosaico que comporte todas as imagens sem cortar nenhuma.
        print("Calculando extensão geográfica total para o ano...")
        all_bounds = []
        for f in tqdm(ano_files, desc=f"Lendo metadados de {ano}"):
            try:
                # O caminho para o subdataset de Temperatura Diurna dentro do arquivo HDF4.
                sds_path = f'HDF4_EOS:EOS_GRID:"{f}":MODIS_Grid_8Day_1km_LST:LST_Day_1km'
                with rasterio.open(sds_path) as src:
                    all_bounds.append(src.bounds)
            except rasterio.errors.RasterioIOError:
                print(f"Aviso: não foi possível ler o arquivo {f}. Pulando na etapa de metadados.")
                continue

        if not all_bounds:
            print(f"Nenhum arquivo válido encontrado para o ano {ano}. Pulando para o próximo ano.")
            continue
            
        # Calcula as coordenadas mínimas e máximas de todos os tiles.
        min_x = min(b.left for b in all_bounds)
        min_y = min(b.bottom for b in all_bounds)
        max_x = max(b.right for b in all_bounds)
        max_y = max(b.top for b in all_bounds)
        total_bounds = (min_x, min_y, max_x, max_y)
        
        # Etapa 2: Agrupar arquivos pela data (o produto é de 8 dias, então múltiplos tiles têm a mesma data).
        files_por_data = {}
        for f in ano_files:
            data_str = os.path.basename(f).split('.')[1] # Extrai a data (ex: A2020001)
            if data_str not in files_por_data:
                files_por_data[data_str] = []
            files_por_data[data_str].append(f)
        
        # Etapa 3: Iterar sobre cada data, criar o mosaico e acumular os valores.
        with tqdm(total=len(files_por_data), desc=f"Criando mosaicos e média para {ano}") as pbar:
            for data, tile_files in files_por_data.items():
                datasets_to_mosaic = []
                for f in tile_files:
                    try:
                        sds_path = f'HDF4_EOS:EOS_GRID:"{f}":MODIS_Grid_8Day_1km_LST:LST_Day_1km'
                        datasets_to_mosaic.append(rasterio.open(sds_path))
                    except rasterio.errors.RasterioIOError:
                        continue # Pula arquivos corrompidos ou ilegíveis.
                
                if not datasets_to_mosaic:
                    pbar.update(1)
                    continue

                # Cria o mosaico para a data atual, usando a extensão total calculada anteriormente.
                mosaic, out_trans = merge(datasets_to_mosaic, bounds=total_bounds)
                
                # Na primeira iteração, captura os metadados (profile) para o arquivo de saída.
                if profile is None:
                    profile = datasets_to_mosaic[0].profile
                    profile.update(transform=out_trans, height=mosaic.shape[1], width=mosaic.shape[2], driver='GTiff')

                # Fecha todos os datasets abertos para liberar memória.
                for ds in datasets_to_mosaic:
                    ds.close()

                # Converte os valores do MODIS LST (Kelvin * 50) para Graus Celsius.
                # A fórmula é: (Valor * 0.02) - 273.15
                temp_celsius = (mosaic * 0.02) - 273.15
                # Define valores inválidos (no data) como NaN.
                temp_celsius[temp_celsius < -100] = np.nan

                # Inicializa os arrays de soma e contagem na primeira iteração válida.
                if soma_anual is None:
                    soma_anual = np.zeros_like(mosaic, dtype=np.float32)
                    contagem_valida = np.zeros_like(mosaic, dtype=np.int16)

                # Acumula os valores: soma os valores de temperatura e incrementa a contagem de pixels válidos.
                soma_anual += np.nan_to_num(temp_celsius, nan=0.0) # Converte NaN para 0 antes de somar.
                contagem_valida += (~np.isnan(temp_celsius)).astype(int) # Adiciona 1 onde a temp não era NaN.
                
                pbar.update(1)
        
        if soma_anual is None:
            print(f"Não foi possível processar nenhum dado para o ano {ano}.")
            continue

        # Etapa 4: Calcula a média anual dividindo a soma pela contagem.
        # `where=contagem_valida!=0` evita a divisão por zero. Onde for 0, o resultado será NaN.
        media_anual = np.divide(soma_anual, contagem_valida, out=np.full_like(soma_anual, np.nan), where=contagem_valida!=0)
        
        # Etapa 5: Salva o resultado como um GeoTIFF.
        print(f"Salvando GeoTIFF de temperatura para {ano}...")
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(media_anual.astype(rasterio.float32))

# --- 3. FUNÇÃO PARA PROCESSAR PLUVIOSIDADE (GPM IMERG) ---

def processar_pluviosidade_anual(input_dir, output_dir, brasil_aoi):
    """
    Processa os dados brutos de GPM IMERG (HDF5) para gerar mosaicos de pluviosidade total anual em mm.
    
    Args:
        input_dir (str): Pasta contendo os arquivos .HDF5 de pluviosidade baixados.
        output_dir (str): Pasta de destino para salvar os GeoTIFFs anuais processados.
        brasil_aoi (gpd.GeoDataFrame): GeoDataFrame do Brasil.
    """
    print("\n--- Processando Pluviosidade Total Anual (GPM IMERG) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(input_dir, '*.HDF5')))
    if not files:
        print("Aviso: Nenhum arquivo de pluviosidade (.HDF5) encontrado.")
        return

    # Agrupa os arquivos por ano.
    files_por_ano = {}
    for f in files:
        # A data no nome do arquivo GPM está em formato diferente (ex: .20200101-S...).
        match = re.search(r'\.(\d{8})-S', f)
        if match:
            ano = match.group(1)[:4]
            if ano not in files_por_ano:
                files_por_ano[ano] = []
            files_por_ano[ano].append(f)

    # Itera sobre cada ano.
    for ano, ano_files in files_por_ano.items():
        output_path = os.path.join(output_dir, f'pluviosidade_total_{ano}.tif')
        if os.path.exists(output_path):
            print(f"Arquivo de pluviosidade para o ano {ano} já existe. Pulando.")
            continue
            
        print(f"Processando ano: {ano} ({len(ano_files)} arquivos)")
        
        soma_anual = None
        profile = None

        # Itera sobre os arquivos mensais do ano.
        for f in tqdm(ano_files, desc=f"Somando totais mensais para {ano}"):
            # Usa a biblioteca h5py para ler o formato HDF5.
            with h5py.File(f, 'r') as hdf:
                # Acessa o dataset de precipitação dentro da estrutura do HDF5.
                precip_data = hdf['/Grid/precipitation'][:]
                # O array original do GPM vem em (longitude, latitude). Transpomos para (latitude, longitude) para o padrão geoespacial.
                precip_data = precip_data.T
                
                # Converte a taxa de chuva (mm/hora) para o total mensal (mm).
                from calendar import monthrange
                match = re.search(r'\.(\d{8})-S', f)
                mes = int(match.group(1)[4:6])
                
                dias_no_mes = monthrange(int(ano), mes)[1]
                horas_no_mes = dias_no_mes * 24
                
                precip_total_mes = precip_data * horas_no_mes
                
                # Acumula a soma mensal no total anual.
                if soma_anual is None:
                    soma_anual = precip_total_mes
                else:
                    soma_anual += precip_total_mes

        if soma_anual is not None:
            # Os dados GPM são globais e têm um georreferenciamento padrão.
            # Podemos definir o profile manualmente.
            transform = rasterio.transform.from_origin(-180, 90, 0.1, 0.1) # Origem no canto superior esquerdo, resolução de 0.1 grau.
            profile = {
                'driver': 'GTiff', 'crs': 'EPSG:4326', 'transform': transform,
                'width': soma_anual.shape[1], 'height': soma_anual.shape[0],
                'count': 1, 'dtype': 'float32', 'nodata': -9999.0 # Valor padrão para 'no data'.
            }
            
            print(f"Salvando GeoTIFF de pluviosidade para {ano}...")
            with rasterio.open(output_path, 'w', **profile) as dst:
                # Os dados de HDF5 podem ter uma dimensão extra. `np.squeeze` remove dimensões de tamanho 1,
                # garantindo que o array seja 2D, como esperado por `rasterio.write`.
                soma_anual_2d = np.squeeze(soma_anual)
                dst.write(soma_anual_2d.astype(rasterio.float32), 1)

# --- 4. BLOCO DE EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    
    # Define e cria os diretórios de saída.
    output_dir_base = "resultados_climaticos"
    output_dir_geotiff = os.path.join(output_dir_base, 'geotiff_maps')
    os.makedirs(output_dir_geotiff, exist_ok=True)
    
    # Carrega o shapefile do Brasil (para manter a consistência, embora não seja usado para recorte aqui).
    try:
        shp_path = os.path.join("./natural_earth_110m_countries/", "ne_110m_admin_0_countries.shp")
        brasil_aoi = gpd.read_file(shp_path)
        brasil_aoi = brasil_aoi[brasil_aoi['ADMIN'] == 'Brazil']
    except Exception as e:
        print(f"Erro ao carregar shapefile do Brasil: {e}")
        exit()

    # Chama as funções de processamento para cada tipo de dado.
    processar_temperatura_anual("./dados_temperatura", output_dir_geotiff, brasil_aoi)
    processar_pluviosidade_anual("./dados_pluviosidade", output_dir_geotiff, brasil_aoi)
    
    print("\nPré-processamento de todos os dados climáticos foi concluído!")