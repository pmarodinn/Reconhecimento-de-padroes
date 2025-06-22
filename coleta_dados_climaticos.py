# =================================================================================================
# SCRIPT 4: coleta_dados_climaticos.py
#
# OBJETIVO:
# Este script é responsável por baixar todos os dados climáticos brutos necessários para a análise.
# Utilizando a biblioteca `earthaccess`, ele se conecta aos servidores da NASA para baixar dois
# conjuntos de dados principais para o território brasileiro e para um período de tempo específico:
#
# 1.  Temperatura da Superfície Terrestre (LST): Do produto MOD11A2 do sensor MODIS.
# 2.  Pluviosidade (Precipitação): Do produto GPM IMERG.
#
# COMO FUNCIONA:
# 1.  Define um período de análise (datas de início e fim).
# 2.  Carrega um shapefile do Brasil para definir a área geográfica de busca (bounding box).
# 3.  Realiza a autenticação na plataforma Earthdata da NASA.
# 4.  Chama uma função genérica de download para cada conjunto de dados, passando os parâmetros
#     específicos do produto (nome, versão) e a área de interesse.
# 5.  Os arquivos são salvos em diretórios locais separados ('dados_temperatura', 'dados_pluviosidade').
# =================================================================================================

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---

import earthaccess      # Biblioteca principal para buscar e baixar dados do repositório da NASA (Earthdata).
import os               # Fornece funções para interagir com o sistema operacional, como criar diretórios (pastas).
import geopandas as gpd # Usado para ler e manipular arquivos de dados geoespaciais (neste caso, o shapefile do Brasil).

# --- 2. FUNÇÃO DE DOWNLOAD ---

def baixar_dados(short_name, version, start_date, end_date, output_dir, brasil_aoi):
    """
    Função genérica para buscar e baixar dados da NASA para uma área e período específicos.

    Args:
        short_name (str): O nome curto do produto de dados da NASA (ex: "MOD11A2").
        version (str): A versão do produto de dados (ex: "061").
        start_date (str): A data de início da busca (formato "AAAA-MM-DD").
        end_date (str): A data de fim da busca (formato "AAAA-MM-DD").
        output_dir (str): O caminho da pasta onde os arquivos baixados serão salvos.
        brasil_aoi (gpd.GeoDataFrame): O GeoDataFrame contendo a geometria do Brasil,
                                       usado para definir a área de busca.
    """
    print(f"\nBuscando por '{short_name}' de {start_date} a {end_date}...")
    
    # Cria a pasta de destino para os downloads, se ela ainda não existir.
    # O `os.makedirs` cria pastas de forma recursiva e não gera erro se a pasta já existir.
    os.makedirs(output_dir, exist_ok=True)
        
    # Extrai as coordenadas geográficas da "caixa delimitadora" (bounding box)
    # que envolve todo o território do Brasil a partir do shapefile.
    bounds = brasil_aoi.total_bounds
    
    # Utiliza a função `search_data` do earthaccess para encontrar os arquivos de dados.
    results = earthaccess.search_data(
        short_name=short_name,              # Filtra pelo nome do produto.
        cloud_hosted=True,                  # Prioriza dados hospedados na nuvem da NASA, que geralmente são mais rápidos para baixar.
        bounding_box=(bounds[0], bounds[1], bounds[2], bounds[3]), # Filtra espacialmente pelos limites do Brasil.
        temporal=(start_date, end_date),    # Filtra pelo período de tempo definido.
        version=version                     # Filtra pela versão específica do produto.
    )
    
    # Verifica se a busca retornou algum resultado.
    if not results:
        print(f" -> Nenhum dado encontrado para o produto '{short_name}' no período especificado.")
        # Encerra a execução desta função se nada for encontrado.
        return

    print(f"Encontrados {len(results)} arquivos. Iniciando o download para a pasta '{output_dir}'...")
    
    # Inicia o download dos arquivos encontrados para a pasta local especificada.
    # A biblioteca `earthaccess` gerencia a conexão e o download de cada arquivo, mostrando uma barra de progresso.
    earthaccess.download(results, local_path=output_dir)
    
    print("Download concluído.")

# --- 3. BLOCO DE EXECUÇÃO PRINCIPAL ---
# Este bloco `if __name__ == "__main__":` garante que o código dentro dele
# só será executado quando o script for rodado diretamente.
if __name__ == "__main__":
    
    # --- A. CONFIGURAÇÃO DO PERÍODO DE ANÁLISE ---
    START_DATE = "2019-01-01"
    END_DATE = "2024-12-31"

    # --- B. CARREGAMENTO DA ÁREA DE INTERESSE (AOI) ---
    # É necessário carregar o contorno do Brasil para que a busca de dados
    # seja limitada geograficamente ao nosso país.
    try:
        # Caminho para o shapefile. Certifique-se de que a pasta 'natural_earth_110m' está no mesmo diretório do script.
        # NOTA: Este caminho parece diferente do usado no SCRIPT 1. Verifique se a pasta é 'natural_earth_110m' ou 'natural_earth_110m_countries'.
        shp_path = os.path.join("./natural_earth_110m_countries/", "ne_110m_admin_0_countries.shp")
        brasil_aoi = gpd.read_file(shp_path)
        # Filtra o GeoDataFrame para manter apenas a linha correspondente ao Brasil.
        brasil_aoi = brasil_aoi[brasil_aoi['ADMIN'] == 'Brazil']
        if brasil_aoi.empty:
            raise FileNotFoundError("Não foi possível encontrar 'Brazil' na coluna 'ADMIN' do shapefile.")
    except Exception as e:
        print(f"ERRO CRÍTICO: Não foi possível carregar o shapefile do Brasil. Detalhes: {e}")
        print("Verifique se o caminho para o shapefile está correto e se o arquivo não está corrompido.")
        # Encerra o script, pois a AOI é essencial para continuar.
        exit()

    # --- C. AUTENTICAÇÃO NA PLATAFORMA EARTHDATA ---
    # O `earthaccess` precisa de autenticação para baixar dados da NASA.
    # Ele buscará por um arquivo .netrc em seu diretório home ou pedirá o login interativo.
    print("Realizando autenticação no NASA Earthdata...")
    auth = earthaccess.login()
    if not auth.authenticated:
        print("ERRO: Autenticação falhou. O script não pode continuar.")
        exit()
    print("Autenticação bem-sucedida.")
    
    # --- D. DOWNLOAD DOS CONJUNTOS DE DADOS ---
    
    # 1. Baixar dados de Temperatura da Superfície Terrestre (MODIS LST)
    # Produto: MOD11A2 - Fornece médias de temperatura de 8 dias com resolução de 1km.
    baixar_dados(
        short_name="MOD11A2",           # Nome curto do produto de Temperatura
        version="061",                  # Versão recomendada para o produto MODIS
        start_date=START_DATE, 
        end_date=END_DATE, 
        output_dir="./dados_temperatura", # Pasta de destino para os dados de temperatura
        brasil_aoi=brasil_aoi
    )
    
    # 2. Baixar dados de Pluviosidade (GPM IMERG)
    # Produto: GPM_3IMERGM - Fornece estimativas de precipitação mensais.
    baixar_dados(
        short_name="GPM_3IMERGM",        # Nome curto do produto de Precipitação Mensal
        version="07",                   # A versão mais recente e recomendada do IMERG
        start_date=START_DATE, 
        end_date=END_DATE, 
        output_dir="./dados_pluviosidade", # Pasta de destino para os dados de pluviosidade
        brasil_aoi=brasil_aoi
    )
    
    print("\nColeta de todos os dados climáticos foi concluída.")