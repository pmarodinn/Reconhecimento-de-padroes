# =================================================================================================
# SCRIPT 2: analise_NDVI.py
#
# OBJETIVO:
# Este script é o coração da análise de NDVI (Índice de Vegetação por Diferença Normalizada).
# Ele lê os mosaicos GeoTIFF de NDVI anuais (previamente criados), calcula estatísticas,
# gera visualizações (gráficos e mapas) e realiza análises mais avançadas, como
# clusterização e previsão, para entender a dinâmica da vegetação no Brasil ao longo do tempo.
#
# PASSOS EXECUTADOS:
# 1.  Calcula estatísticas descritivas (média, mediana, etc.) para o NDVI de cada ano.
# 2.  Analisa a tendência temporal do NDVI médio para o Brasil.
# 3.  Visualiza as anomalias anuais de NDVI.
# 4.  Cria um mapa de diferença para comparar o primeiro e o último ano da série.
# 5.  Gera análises específicas por estado (gráficos de tendência e mapas).
# 6.  Aplica um algoritmo de Machine Learning (KMeans) para clusterizar o território em zonas de NDVI similar.
# 7.  Treina um modelo de Machine Learning (Random Forest) para prever o NDVI do próximo ano.
# 8.  Gera gráficos de tendência individuais para cada estado.
# 9.  Classifica a vegetação em categorias (densa, rasteira, etc.) com base nos valores de NDVI.
# =================================================================================================

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
# Importa as ferramentas necessárias para cada etapa da análise.

import geopandas as gpd                  # Para manipulação de dados geoespaciais (shapefiles).
import rasterio                          # Para leitura, escrita e manipulação de dados raster (GeoTIFFs).
from rasterio.plot import show           # Função específica para exibir dados raster com Matplotlib.
import matplotlib.pyplot as plt          # Biblioteca principal para criar gráficos e visualizações.
import numpy as np                       # Biblioteca para computação numérica, essencial para trabalhar com arrays (imagens raster).
import os                                # Para interagir com o sistema operacional, como criar pastas e manipular caminhos de arquivos.
import pandas as pd                      # Para manipulação e análise de dados tabulares (nossas estatísticas).
from tqdm import tqdm                    # Para criar barras de progresso, úteis em loops longos.
from scipy import stats                  # Biblioteca de computação científica, usada aqui para regressão linear (cálculo de tendência).
import glob                              # Para encontrar arquivos que correspondem a um padrão (ex: todos os .tif em uma pasta).
from shapely.geometry import mapping     # Para converter geometrias do GeoPandas para um formato que o Rasterio entende.
from sklearn.cluster import KMeans       # Algoritmo de Machine Learning para agrupamento não supervisionado (clusterização).
from sklearn.ensemble import RandomForestRegressor # Algoritmo de Machine Learning para tarefas de regressão (previsão).
from sklearn.model_selection import train_test_split # Ferramenta para dividir dados em conjuntos de treino e teste.
from sklearn.metrics import mean_squared_error   # Métrica para avaliar a performance do modelo de regressão.
import joblib                            # Para salvar e carregar modelos de Machine Learning treinados.

# --- 2. DEFINIÇÃO DAS FUNÇÕES DE ANÁLISE ---
# O código é organizado em funções para tornar cada análise modular, reutilizável e fácil de entender.

def gerar_estatisticas_anuais(geotiff_dir, output_path_csv):
    """
    Lê todos os GeoTIFFs de NDVI anual, calcula estatísticas descritivas para cada um e salva em um CSV.

    Args:
        geotiff_dir (str): Caminho para a pasta contendo os mosaicos NDVI anuais.
        output_path_csv (str): Caminho completo do arquivo CSV onde as estatísticas serão salvas.

    Returns:
        pd.DataFrame: Um DataFrame do Pandas com as estatísticas anuais.
    """
    print("[Análise 1/7] Gerando estatísticas detalhadas anuais...")
    # Encontra todos os arquivos .tif na pasta de entrada e os ordena (garante a ordem cronológica).
    geotiff_files = sorted(glob.glob(os.path.join(geotiff_dir, 'mosaico_ndvi_brasil_*.tif')))
    if not geotiff_files:
        print(" -> ATENÇÃO: Nenhum arquivo GeoTIFF encontrado na pasta especificada. Pulando esta etapa.")
        return None

    df_data = []
    # Itera sobre cada arquivo GeoTIFF encontrado, com uma barra de progresso (tqdm).
    for f in tqdm(geotiff_files, desc="Calculando estatísticas anuais"):
        # Extrai o ano do nome do arquivo.
        year = os.path.basename(f).split('_')[-1].replace('.tif', '')
        with rasterio.open(f) as src:
            # Lê a primeira banda do raster como um array NumPy.
            ndvi = src.read(1)
            # Converte valores 'no data' (ex: -9999) para NaN (Not a Number), para que sejam ignorados nos cálculos.
            ndvi[ndvi < -1] = np.nan
            # Calcula as estatísticas, ignorando os valores NaN, e adiciona à lista.
            df_data.append({
                'Ano': int(year),
                'NDVI Médio': np.nanmean(ndvi),
                'NDVI Mediana': np.nanmedian(ndvi),
                'Desvio Padrão': np.nanstd(ndvi),
                'NDVI Máximo': np.nanmax(ndvi),
                'NDVI Mínimo': np.nanmin(ndvi)
            })
    
    # Converte a lista de dicionários em um DataFrame do Pandas.
    df_stats = pd.DataFrame(df_data)
    # Salva o DataFrame em um arquivo CSV formatado para o padrão brasileiro (vírgula decimal, ponto e vírgula separador).
    df_stats.to_csv(output_path_csv, index=False, decimal=',', sep=';')
    print(f" -> Estatísticas salvas em: {output_path_csv}")
    return df_stats


def analise_tendencia_geral(df_stats, output_dir_plots, output_dir_csv):
    """
    Calcula e plota a tendência linear do NDVI médio ao longo dos anos para o Brasil.

    Args:
        df_stats (pd.DataFrame): DataFrame contendo as estatísticas anuais de NDVI.
        output_dir_plots (str): Pasta para salvar o gráfico gerado.
        output_dir_csv (str): Pasta para salvar arquivos de dados (não usado aqui, mas mantido por padrão).
    """
    print("[Análise 2/7] Gerando análise de tendência temporal...")
    if df_stats is None or df_stats.empty:
        print(" -> DataFrame de estatísticas vazio. Pulando análise de tendência.")
        return

    # Extrai os dados de ano e NDVI médio do DataFrame.
    anos, ndvi_medio = df_stats['Ano'], df_stats['NDVI Médio']
    # Usa a função linregress da biblioteca SciPy para calcular a regressão linear.
    slope, intercept, r_value, p_value, std_err = stats.linregress(anos, ndvi_medio)
    
    # Inicia a criação do gráfico.
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid') # Define um estilo visual agradável.
    # Plota o NDVI médio anual como uma linha com marcadores.
    ax.plot(anos, ndvi_medio, marker='o', linestyle='-', label='NDVI Médio Anual')
    # Plota a linha de tendência calculada pela regressão. O R² (r_value**2) indica o quão bem a linha se ajusta aos dados.
    ax.plot(anos, intercept + slope*anos, 'r', label=f'Linha de Tendência (R²={r_value**2:.2f})')
    # Adiciona títulos e legendas.
    ax.set_title('Tendência do NDVI Médio para o Brasil', fontsize=16)
    ax.set_xlabel('Ano'); ax.set_ylabel('NDVI Médio'); ax.legend(); ax.grid(True)
    # Salva a figura em alta resolução.
    plt.savefig(os.path.join(output_dir_plots, 'tendencia_ndvi_brasil.png'), dpi=300, bbox_inches='tight')
    plt.close(fig) # Fecha a figura para liberar memória.
    print(f" -> Gráfico de tendência salvo.")


def analise_anomalias_anuais(df_stats, output_dir_plots):
    """
    Calcula e plota as anomalias de NDVI, mostrando o desvio de cada ano em relação à média do período completo.

    Args:
        df_stats (pd.DataFrame): DataFrame com as estatísticas anuais.
        output_dir_plots (str): Pasta para salvar o gráfico de anomalias.
    """
    print("[Análise Bônus] Gerando gráfico de anomalias anuais...")
    if df_stats is None or df_stats.empty:
        print(" -> DataFrame de estatísticas vazio. Pulando análise de anomalias.")
        return

    # Calcula a média de NDVI de todo o período.
    overall_mean = df_stats['NDVI Médio'].mean()
    # Calcula a anomalia de cada ano (valor do ano - média geral).
    df_stats['Anomalia'] = df_stats['NDVI Médio'] - overall_mean
    
    # Criação do gráfico de barras.
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    # Define as cores das barras: verde para anomalias positivas (acima da média) e vermelho para negativas.
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in df_stats['Anomalia']]
    ax.bar(df_stats['Ano'], df_stats['Anomalia'], color=colors)
    # Customização do gráfico.
    ax.set_title('Anomalia do NDVI Anual em Relação à Média do Período', fontsize=16)
    ax.set_xlabel('Ano'); ax.set_ylabel('Diferença da Média'); ax.axhline(0, color='grey', lw=1) # Linha de base no zero.
    plt.savefig(os.path.join(output_dir_plots, 'anomalias_ndvi_anuais.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Gráfico de anomalias salvo.")


def analise_mapa_de_diferenca(geotiff_dir, brasil_aoi, output_dir_plots, output_dir_geotiff):
    """
    Calcula a diferença de NDVI pixel a pixel entre o primeiro e o último ano e gera um mapa.

    Args:
        geotiff_dir (str): Pasta com os mosaicos GeoTIFF.
        brasil_aoi (gpd.GeoDataFrame): Shapefile do contorno do Brasil para o mapa.
        output_dir_plots (str): Pasta para salvar a imagem PNG do mapa.
        output_dir_geotiff (str): Pasta para salvar o novo GeoTIFF com os dados da diferença.
    """
    print("[Análise 3/7] Gerando mapa de diferença...")
    geotiff_files = sorted(glob.glob(os.path.join(geotiff_dir, 'mosaico_ndvi_brasil_*.tif')))
    if len(geotiff_files) < 2:
        print(" -> Análise de diferença requer pelo menos dois GeoTIFFs. Pulando.")
        return

    # Abre o primeiro e o último arquivo da lista ordenada.
    with rasterio.open(geotiff_files[0]) as src_primeiro, rasterio.open(geotiff_files[-1]) as src_ultimo:
        primeiro_ano_str = os.path.basename(geotiff_files[0]).split('_')[-1].replace('.tif', '')
        ultimo_ano_str = os.path.basename(geotiff_files[-1]).split('_')[-1].replace('.tif', '')
        
        # Subtrai os arrays de NDVI para encontrar a diferença.
        diferenca_ndvi = src_ultimo.read(1) - src_primeiro.read(1)

        # Prepara para salvar o resultado como um novo GeoTIFF, herdando os metadados (projeção, etc.).
        out_meta = src_ultimo.meta.copy()
        output_path_geotiff = os.path.join(output_dir_geotiff, f'diferenca_ndvi_{primeiro_ano_str}-{ultimo_ano_str}.tif')
        with rasterio.open(output_path_geotiff, 'w', **out_meta) as dst:
            dst.write(diferenca_ndvi, 1)
        print(f" -> GeoTIFF de diferença salvo.")
        
        # Criação do mapa visual.
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.style.use('default') # Usa estilo padrão para mapas.
        # Define os limites da paleta de cores para realçar as diferenças, ignorando outliers extremos.
        lim = np.nanpercentile(np.abs(diferenca_ndvi), 95)
        # Mostra o raster de diferença usando uma paleta de cores divergente (Vermelho-Azul).
        show(diferenca_ndvi, transform=src_ultimo.transform, ax=ax, cmap='RdBu', vmin=-lim, vmax=lim)
        # Adiciona o contorno do Brasil sobre o mapa.
        brasil_aoi.to_crs(src_ultimo.crs).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        ax.set_title(f'Diferença de NDVI: {ultimo_ano_str} vs {primeiro_ano_str}', fontsize=16)
        ax.set_xticks([]); ax.set_yticks([]) # Remove os eixos para um mapa mais limpo.
        plt.savefig(os.path.join(output_dir_plots, f'mapa_diferenca_ndvi_{primeiro_ano_str}-{ultimo_ano_str}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f" -> Mapa de diferença salvo.")


def analises_regionais_plots(path_csv_estados, estados_aoi, output_dir_plots):
    """
    Gera visualizações de NDVI por estado: um gráfico de linhas para estados selecionados e um mapa coroplético.

    Args:
        path_csv_estados (str): Caminho para o CSV com NDVI médio por estado.
        estados_aoi (gpd.GeoDataFrame): Shapefile com a geometria dos estados.
        output_dir_plots (str): Pasta para salvar os gráficos.
    """
    print("[Análise 4/7] Gerando gráficos e mapas por estado...")
    if not os.path.exists(path_csv_estados):
        print(f" -> Arquivo de dados estaduais não encontrado em {path_csv_estados}. Pulando.")
        return
    if estados_aoi is None:
        print(" -> Shapefile dos estados não carregado. Pulando.")
        return

    # Carrega os dados de NDVI por estado.
    df_estados = pd.read_csv(path_csv_estados, sep=';', decimal=',')

    # --- Gráfico 1: Linhas de tendência para estados selecionados ---
    estados_selecionados = ['AM', 'MT', 'SP', 'BA', 'RS'] # Exemplos de estados de biomas diferentes.
    df_plot = df_estados[df_estados['Sigla'].isin(estados_selecionados)]
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    # Plota uma linha para cada estado no subconjunto.
    for estado, group in df_plot.groupby('Estado'):
        ax.plot(group['Ano'], group['NDVI_Medio'], marker='o', linestyle='-', label=estado)
    ax.set_title('Tendência do NDVI Médio para Estados Selecionados', fontsize=16)
    ax.set_xlabel('Ano'); ax.set_ylabel('NDVI Médio'); ax.legend(title='Estado'); ax.grid(True)
    plt.savefig(os.path.join(output_dir_plots, 'tendencia_ndvi_estados_selecionados.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Gráfico de tendência por estado salvo.")

    # --- Gráfico 2: Mapa Coroplético (de cores) ---
    # Filtra os dados para mostrar apenas o último ano disponível.
    df_ultimo_ano = df_estados[df_estados['Ano'] == df_estados['Ano'].max()]
    # Junta os dados de NDVI com as geometrias dos estados.
    mapa_plot = pd.merge(estados_aoi, df_ultimo_ano, left_on='SIGLA_UF', right_on='Sigla')
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.style.use('default')
    # Plota o mapa, colorindo cada estado de acordo com seu 'NDVI_Medio'.
    mapa_plot.plot(column='NDVI_Medio', cmap='RdYlGn', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                   legend_kwds={'label': "NDVI Médio", 'orientation': "horizontal"})
    ax.set_title(f'NDVI Médio por Estado - {df_estados["Ano"].max()}', fontsize=16)
    ax.set_axis_off() # Remove os eixos.
    plt.savefig(os.path.join(output_dir_plots, f'mapa_ndvi_estados_{df_estados["Ano"].max()}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Mapa de NDVI por estado salvo.")


def clusterizacao_ndvi(geotiff_dir, output_dir_geotiff, output_dir_plots):
    """
    Usa o algoritmo K-Means para agrupar pixels com valores de NDVI semelhantes, criando um mapa de zonas.

    Args:
        geotiff_dir (str): Pasta com os mosaicos GeoTIFF.
        output_dir_geotiff (str): Pasta para salvar o GeoTIFF dos clusters.
        output_dir_plots (str): Pasta para salvar a imagem PNG do mapa de clusters.
    """
    print("[Análise 5/7] Realizando clusterização espacial do NDVI...")
    
    geotiff_files = sorted(glob.glob(os.path.join(geotiff_dir, 'mosaico_ndvi_brasil_*.tif')))
    if not geotiff_files:
        print(" -> Nenhum GeoTIFF encontrado para clusterização.")
        return

    # Usa o ano do meio da série como representativo para a clusterização.
    mid_index = len(geotiff_files) // 2
    with rasterio.open(geotiff_files[mid_index]) as src:
        profile = src.profile
        ndvi_data = src.read(1)
        transform = src.transform

    # Prepara os dados para o K-Means: remove NaNs e formata como uma lista de pontos.
    valid_mask = ~np.isnan(ndvi_data)
    X = ndvi_data[valid_mask].reshape(-1, 1)

    # Inicializa e treina o modelo K-Means para encontrar 5 clusters.
    # n_init=10 executa o algoritmo 10 vezes com centróides diferentes e escolhe o melhor resultado.
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
    
    # Cria um raster em branco para armazenar os resultados da clusterização.
    clusters = np.zeros_like(ndvi_data, dtype=float)
    clusters[:] = np.nan
    # Atribui o rótulo do cluster a cada pixel válido.
    clusters[valid_mask] = kmeans.labels_

    # Salva o raster de clusters como um novo GeoTIFF.
    cluster_path = os.path.join(output_dir_geotiff, "ndvi_clusters.tif")
    profile.update(dtype=rasterio.float32) # Atualiza o tipo de dado no perfil do raster.
    with rasterio.open(cluster_path, 'w', **profile) as dst:
        dst.write(clusters.astype(rasterio.float32), 1)

    # Plota o mapa de clusters.
    fig, ax = plt.subplots(figsize=(10, 10))
    show(clusters, transform=transform, cmap='tab10', ax=ax) # 'tab10' é uma boa paleta para dados categóricos.
    ax.set_title('Clusters de NDVI - Classificação Espacial', fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(os.path.join(output_dir_plots, 'mapa_clusters_ndvi.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Mapa de clusters salvo.")


def prever_ndvi(df_stats, output_dir_csv):
    """
    Treina um modelo de Machine Learning (Random Forest) para prever o NDVI médio do próximo ano.

    Args:
        df_stats (pd.DataFrame): DataFrame com as estatísticas anuais.
        output_dir_csv (str): Pasta para salvar o modelo treinado e o relatório.
    """
    print("[Análise 6/7] Treinando modelo de previsão do NDVI...")
    if df_stats is None or len(df_stats) < 2:
        print(" -> DataFrame de estatísticas insuficiente para previsão. Pulando.")
        return

    # Define as variáveis: 'Ano' (X) para prever 'NDVI Médio' (y).
    X = df_stats[['Ano']]
    y = df_stats['NDVI Médio']

    # Divide os dados em treino e teste (os últimos 20% dos dados são para teste).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Inicializa e treina o modelo. RandomForest é um modelo robusto e bom para pequenas séries.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Avalia o modelo usando o conjunto de teste, se houver.
    if not X_test.empty:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f" -> Erro Quadrático Médio (MSE) no conjunto de teste: {mse:.6f}")
    else:
        mse = -1 # Sinaliza que não houve teste.
        print(" -> Não houve conjunto de teste para calcular o MSE (poucos dados).")

    # Salva o modelo treinado para uso futuro.
    joblib.dump(model, os.path.join(output_dir_csv, 'modelo_random_forest.pkl'))

    # Usa o modelo treinado para prever o NDVI do próximo ano.
    next_year = int(df_stats['Ano'].max()) + 1
    predicted_next = model.predict([[next_year]])

    # Gera um relatório simples com os resultados.
    report = f"""
Relatório de Previsão de NDVI com Random Forest
================================================
- Modelo treinado com sucesso.
- Ano da Previsão: {next_year}
- Valor de NDVI Médio Previsto: {predicted_next[0]:.4f}
- Erro Quadrático Médio (MSE) no conjunto de teste: {'Não aplicável' if mse == -1 else f'{mse:.6f}'}
"""
    with open(os.path.join(output_dir_csv, 'relatorio_ml.txt'), 'w') as f:
        f.write(report)
    print(f" -> Modelo de ML e relatório de previsão salvos.")


def tendencias_por_estado(path_csv_estados, output_dir_plots):
    """
    Gera um gráfico de tendência linear individual para cada estado.

    Args:
        path_csv_estados (str): Caminho para o CSV com dados de NDVI por estado.
        output_dir_plots (str): Pasta para salvar os gráficos de tendência.
    """
    print("[Análise 7/7] Calculando tendências regionais por estado...")
    if not os.path.exists(path_csv_estados):
        print(f" -> Arquivo de dados estaduais não encontrado em {path_csv_estados}. Pulando.")
        return

    df_estados = pd.read_csv(path_csv_estados, sep=';', decimal=',')
    estados_selecionados = df_estados['Sigla'].unique()
    print(f" -> Gerando gráficos de tendência para {len(estados_selecionados)} estados...")

    # Itera sobre cada estado único.
    for sigla in tqdm(estados_selecionados, desc="Gerando tendências estaduais"):
        df_estado = df_estados[df_estados['Sigla'] == sigla]
        # Pula se não houver pelo menos 2 pontos de dados para a regressão.
        if len(df_estado) < 2:
            continue

        # Calcula a regressão linear para o estado atual.
        anos, valores = df_estado['Ano'], df_estado['NDVI_Medio']
        slope, intercept, r_value, p_value, _ = stats.linregress(anos, valores)

        # Plota o gráfico para o estado.
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.style.use('seaborn-v0_8-whitegrid')
        # A legenda mostra o R² e o p-value (significância estatística da tendência).
        ax.plot(anos, valores, marker='o', label=f'R²={r_value**2:.2f}, p-value={p_value:.3f}')
        ax.plot(anos, intercept + slope*anos, 'r--')
        ax.set_title(f'Tendência do NDVI no Estado de {sigla}', fontsize=14)
        ax.legend()
        plt.savefig(os.path.join(output_dir_plots, f'tendencia_estado_{sigla}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(" -> Tendências regionais geradas.")


def mapa_classes_vegetacao(geotiff_dir, brasil_aoi, output_dir_geotiff, output_dir_plots):
    """
    Classifica o NDVI em categorias de vegetação (ex: solo exposto, vegetação densa) e gera um mapa temático.

    Args:
        geotiff_dir (str): Pasta com os mosaicos GeoTIFF.
        brasil_aoi (gpd.GeoDataFrame): Shapefile do contorno do Brasil.
        output_dir_geotiff (str): Pasta para salvar o GeoTIFF classificado.
        output_dir_plots (str): Pasta para salvar a imagem PNG do mapa.
    """
    print("[Análise Bônus] Gerando mapa de classes de vegetação via NDVI...")

    geotiff_files = sorted(glob.glob(os.path.join(geotiff_dir, 'mosaico_ndvi_brasil_*.tif')))
    if not geotiff_files:
        print(" -> Nenhum GeoTIFF encontrado para análise de classes.")
        return

    # Usa o último ano para a classificação.
    with rasterio.open(geotiff_files[-1]) as src:
        ndvi = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile

    # Define os limiares para cada classe de vegetação.
    # 0=Sem dados, 1=Solo/Água, 2=Rasteira, 3=Média, 4=Densa.
    classes = np.zeros_like(ndvi, dtype=np.uint8) # Começa com 0 para 'NoData'.
    classes[ndvi < 0.2] = 1 # Solo Exposto/Água
    classes[(ndvi >= 0.2) & (ndvi < 0.4)] = 2 # Vegetação Rasteira/Pastagem
    classes[(ndvi >= 0.4) & (ndvi < 0.6)] = 3 # Vegetação Média/Floresta Aberta
    classes[ndvi >= 0.6] = 4 # Vegetação Densa
    classes[np.isnan(ndvi)] = 0 # Garante que pixels NaN sejam classificados como 'NoData'.

    # Salva o raster classificado como um novo GeoTIFF.
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    out_path = os.path.join(output_dir_geotiff, 'classes_vegetacao_ndvi.tif')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(classes, 1)

    # Cria o mapa visual com uma legenda customizada.
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('viridis', 5) # Paleta de cores com 5 cores distintas.
    show(classes, transform=transform, ax=ax, cmap=cmap)
    
    brasil_aoi.to_crs(crs).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    ax.set_title('Classes de Vegetação via NDVI', fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
    
    # Cria os elementos da legenda manualmente para corresponder às nossas classes.
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap(0/4.), label='Solo Exposto/Água'),
                       Patch(facecolor=cmap(1/4.), label='Vegetação Rasteira'),
                       Patch(facecolor=cmap(2/4.), label='Vegetação Média'),
                       Patch(facecolor=cmap(3/4.), label='Vegetação Densa'),
                       Patch(facecolor='grey', label='Sem Dados')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.savefig(os.path.join(output_dir_plots, 'mapa_classes_vegetacao.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(" -> Mapa de classes de vegetação gerado.")


# --- 3. BLOCO DE EXECUÇÃO PRINCIPAL ---
# Este bloco é executado quando o script é chamado diretamente.
if __name__ == "__main__":
    # Define as pastas de saída padrão.
    output_dir_base = "resultados"
    output_dir_geotiff = os.path.join(output_dir_base, 'geotiff_data') # Renomeado para clareza
    output_dir_csv = os.path.join(output_dir_base, 'csv_data')
    output_dir_plots = os.path.join(output_dir_base, 'plots')
    # Cria todas as pastas de saída, se não existirem.
    for d in [output_dir_geotiff, output_dir_csv, output_dir_plots]:
        os.makedirs(d, exist_ok=True)

    print("\n--- INICIANDO GERAÇÃO DE ANÁLISES A PARTIR DOS DADOS PROCESSADOS ---")

    # Carrega os shapefiles vetoriais necessários para as análises espaciais.
    try:
        # Carrega o shapefile dos países para obter o contorno do Brasil.
        paises_shp_path = os.path.join("./natural_earth_110m_countries/", "ne_110m_admin_0_countries.shp")
        brasil_aoi = gpd.read_file(paises_shp_path)
        brasil_aoi = brasil_aoi[brasil_aoi['ADMIN'] == 'Brazil']
        # Carrega o shapefile dos estados do Brasil.
        estados_shp_path = os.path.join("./BR_UF_2022/", "BR_UF_2022.shp")
        estados_aoi = gpd.read_file(estados_shp_path)
    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar shapefiles: {e}")
        print("Por favor, certifique-se que os shapefiles 'ne_110m_admin_0_countries' e 'BR_UF_2022' estão nas pastas corretas.")
        brasil_aoi, estados_aoi = None, None
        exit() # Encerra o script se os arquivos essenciais não forem encontrados.

    # --- Execução Sequencial das Análises ---
    # Chama cada função de análise na ordem lógica.
    
    df_stats = gerar_estatisticas_anuais(output_dir_geotiff, os.path.join(output_dir_csv, 'estatisticas_anuais_ndvi.csv'))
    analise_tendencia_geral(df_stats, output_dir_plots, output_dir_csv)
    analise_anomalias_anuais(df_stats, output_dir_plots)
    analise_mapa_de_diferenca(output_dir_geotiff, brasil_aoi, output_dir_plots, output_dir_geotiff)
    analises_regionais_plots(os.path.join(output_dir_csv, 'ndvi_medio_por_estado.csv'), estados_aoi, output_dir_plots)
    clusterizacao_ndvi(output_dir_geotiff, output_dir_geotiff, output_dir_plots)
    prever_ndvi(df_stats, output_dir_csv)
    tendencias_por_estado(os.path.join(output_dir_csv, 'ndvi_medio_por_estado.csv'), output_dir_plots)
    mapa_classes_vegetacao(output_dir_geotiff, brasil_aoi, output_dir_geotiff, output_dir_plots)

    print("\n--- Análises concluídas com sucesso! Verifique a pasta 'resultados'. ---")