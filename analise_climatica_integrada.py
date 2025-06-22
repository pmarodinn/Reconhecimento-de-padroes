# =================================================================================================
# SCRIPT 7: analise_integrada_final.py
#
# OBJETIVO:
# Este script é a etapa final da análise, onde os dados de vegetação (NDVI) e os dados
# climáticos (temperatura e chuva) são integrados. O objetivo principal é encontrar e
# visualizar as correlações entre as condições climáticas e a saúde da vegetação em
# diferentes escalas espaciais: nacional, regional e estadual.
#
# PASSOS EXECUTADOS:
# 1. Carrega os dados de NDVI e clima previamente processados.
# 2. Realiza uma análise de correlação em nível de Brasil.
# 3. Agrega os dados por região (Norte, Nordeste, etc.) e analisa as correlações.
# 4. Calcula a correlação para cada estado e gera um mapa temático do Brasil.
# =================================================================================================

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
# Importa as bibliotecas necessárias para manipulação de dados, visualização e análises estatísticas.

import pandas as pd  # Biblioteca fundamental para manipulação e análise de dados tabulares (planilhas, CSVs).
import matplotlib.pyplot as plt  # Biblioteca para a criação de gráficos e visualizações estáticas.
import seaborn as sns  # Construída sobre o Matplotlib, oferece visualizações estatísticas mais atraentes e complexas.
import os  # Fornece uma maneira de usar funcionalidades dependentes do sistema operacional, como criar pastas.
from scipy.stats import pearsonr  # Função da biblioteca SciPy para calcular o coeficiente de correlação de Pearson.
import geopandas as gpd  # Extensão do Pandas que permite trabalhar com dados geoespaciais e criar mapas.

# --- 2. CONFIGURAÇÃO DOS CAMINHOS E DIRETÓRIOS ---
# Define os caminhos de entrada (onde os dados estão) e de saída (onde os resultados serão salvos).
# Isso torna o código mais organizado e fácil de manter.

# Diretórios de entrada
INPUT_CSV_NDVI_DIR = "resultados/csv_data"  # Pasta com os CSVs gerados pela análise de NDVI.
INPUT_CSV_CLIMA_DIR = "analises_climaticas/csv_data"  # Pasta com os CSVs gerados pela análise climática.
SHAPEFILE_DIR = "./BR_UF_2022/"  # Pasta contendo o mapa vetorial dos estados do Brasil.

# Diretórios de saída
OUTPUT_DIR_BASE = "analise_final_integrada"  # Pasta principal para salvar todos os resultados deste script.
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_DIR_BASE, 'plots')  # Subpasta para salvar os gráficos e mapas.
OUTPUT_CSV_DIR = os.path.join(OUTPUT_DIR_BASE, 'csv_data')  # Subpasta para salvar as tabelas de dados integrados.

# Cria os diretórios de saída caso eles não existam.
# O `exist_ok=True` evita que um erro seja lançado se a pasta já existir.
for d in [OUTPUT_PLOTS_DIR, OUTPUT_CSV_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 3. FUNÇÕES DE ANÁLISE ---
# O código é modularizado em funções para separar as diferentes etapas da análise (nacional, regional, estadual).
# Isso melhora a legibilidade e a reutilização do código.

def analise_correlacao_nacional(df_ndvi, df_clima, output_dir, output_csv_dir):
    """
    Analisa e plota a correlação entre NDVI e variáveis climáticas em nível nacional.
    
    Args:
        df_ndvi (pd.DataFrame): DataFrame com os dados de NDVI médio anual para o Brasil.
        df_clima (pd.DataFrame): DataFrame com os dados climáticos médios anuais para o Brasil.
        output_dir (str): Caminho da pasta para salvar os gráficos.
        output_csv_dir (str): Caminho da pasta para salvar o CSV integrado.
    """
    print("\n[Análise 1/3] Gerando análise de correlação em nível nacional...")

    # Junta as tabelas de NDVI e clima usando a coluna 'Ano' como chave.
    df_integrado = pd.merge(df_ndvi, df_clima, on='Ano')
    
    # Salva a tabela de dados integrados em um arquivo CSV para referência futura.
    df_integrado.to_csv(
        os.path.join(output_csv_dir, 'estatisticas_integradas_nacional.csv'),
        index=False, decimal=',', sep=';'
    )
    
    # --- GRÁFICO 1: SÉRIES TEMPORAIS NORMALIZADAS ---
    # Normaliza os dados (coloca todos na mesma escala, de 0 a 1) para permitir a comparação visual das tendências.
    df_norm = df_integrado.drop(columns=['Ano']).apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df_norm['Ano'] = df_integrado['Ano']  # Readiciona a coluna 'Ano'

    # Criação do gráfico de linhas
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid') # Define um estilo visual para o gráfico.
    ax.plot(df_norm['Ano'], df_norm['NDVI Médio'], marker='o', linestyle='-', label='NDVI', color='green', linewidth=2.5)
    ax.plot(df_norm['Ano'], df_norm['Temp_Media_C'], marker='s', linestyle='--', label='Temperatura', color='red')
    ax.plot(df_norm['Ano'], df_norm['Chuva_Media_Anual_mm'], marker='^', linestyle=':', label='Pluviosidade', color='blue')
    
    # Customização do gráfico (títulos, eixos, legenda)
    ax.set_title('Séries Temporais Normalizadas: NDVI vs. Clima (Brasil, 2019-2024)', fontsize=16)
    ax.set_xlabel('Ano')
    ax.set_ylabel('Valor Normalizado (0 a 1)')
    ax.legend()
    ax.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'series_temporais_normalizadas.png'), dpi=300, bbox_inches='tight')
    plt.close(fig) # Fecha a figura para liberar memória.
    print(" -> Gráfico de séries temporais normalizadas salvo.")

    # --- GRÁFICO 2: DIAGRAMAS DE DISPERSÃO COM CORRELAÇÃO ---
    # Cria dois subplots (um ao lado do outro) para visualizar a relação direta entre as variáveis.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Correlação NDVI vs. Temperatura
    corr_temp, _ = pearsonr(df_integrado['Temp_Media_C'], df_integrado['NDVI Médio'])
    sns.regplot(ax=axes[0], x='Temp_Media_C', y='NDVI Médio', data=df_integrado, color='red')
    axes[0].set_title(f'NDVI vs. Temperatura\nCorrelação de Pearson (r) = {corr_temp:.2f}', fontsize=14)
    axes[0].set_xlabel('Temperatura Média Anual (°C)')
    axes[0].set_ylabel('NDVI Médio Anual')
    
    # Correlação NDVI vs. Pluviosidade
    corr_rain, _ = pearsonr(df_integrado['Chuva_Media_Anual_mm'], df_integrado['NDVI Médio'])
    sns.regplot(ax=axes[1], x='Chuva_Media_Anual_mm', y='NDVI Médio', data=df_integrado, color='blue')
    axes[1].set_title(f'NDVI vs. Pluviosidade\nCorrelação de Pearson (r) = {corr_rain:.2f}', fontsize=14)
    axes[1].set_xlabel('Pluviosidade Média Anual (mm)')
    axes[1].set_ylabel('') # Remove o label do eixo Y para um visual mais limpo

    plt.tight_layout() # Ajusta o espaçamento entre os subplots.
    plt.savefig(os.path.join(output_dir, 'correlacao_ndvi_clima.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(" -> Gráficos de correlação salvos.")

def analise_correlacao_regional(path_csv_ndvi, path_csv_clima, output_dir, output_csv_dir):
    """
    Analisa a correlação entre NDVI e clima, agrupando os dados por região do Brasil.
    """
    print("\n[Análise 2/3] Gerando análise de correlação por Região...")
    
    # Carrega os dados estaduais de NDVI e Clima
    df_ndvi_estados = pd.read_csv(path_csv_ndvi, sep=';', decimal=',')
    df_clima_estados = pd.read_csv(path_csv_clima, sep=';', decimal=',')
    
    # Define um dicionário para mapear cada sigla de estado à sua respectiva região.
    mapa_regioes = {
        'Norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'],
        'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
        'Centro-Oeste': ['DF', 'GO', 'MT', 'MS'],
        'Sudeste': ['ES', 'MG', 'RJ', 'SP'],
        'Sul': ['PR', 'RS', 'SC']
    }

    # Função auxiliar para encontrar a região de uma sigla
    def get_regiao(sigla):
        for regiao, siglas in mapa_regioes.items():
            if sigla in siglas: return regiao
        return None

    # Aplica a função para criar a coluna 'Regiao' em ambos os DataFrames.
    df_ndvi_estados['Regiao'] = df_ndvi_estados['Sigla'].apply(get_regiao)
    df_clima_estados['Regiao'] = df_clima_estados['Sigla'].apply(get_regiao)

    # Junta os dados estaduais de NDVI e Clima.
    df_integrado_estados = pd.merge(df_ndvi_estados, df_clima_estados, on=['Ano', 'Sigla', 'Estado', 'Regiao'])
    df_integrado_estados.to_csv(
        os.path.join(output_csv_dir, 'estatisticas_integradas_estadual.csv'),
        index=False, decimal=',', sep=';'
    )

    # Agrupa os dados por Região e Ano, calculando a média para cada variável.
    df_regional = df_integrado_estados.groupby(['Regiao', 'Ano']).agg({
        'NDVI_Medio': 'mean', 
        'Temp_Media_C': 'mean', 
        'Chuva_Media_Anual_mm': 'mean'
    }).reset_index()

    # Cria gráficos de dispersão (regplots) para cada região usando FacetGrid.
    # FacetGrid é uma forma poderosa de criar múltiplos gráficos baseados em subconjuntos de dados.
    g = sns.FacetGrid(df_regional, col="Regiao", col_wrap=3, hue="Regiao", sharex=False, sharey=False, height=4)
    g.map(sns.regplot, "Temp_Media_C", "NDVI_Medio")
    g.add_legend()
    g.fig.suptitle('Correlação NDVI vs. Temperatura por Região', y=1.02, fontsize=16)
    plt.savefig(os.path.join(output_dir, 'correlacao_regional_temperatura.png'), dpi=300, bbox_inches='tight')
    plt.close()

    g = sns.FacetGrid(df_regional, col="Regiao", col_wrap=3, hue="Regiao", sharex=False, sharey=False, height=4)
    g.map(sns.regplot, "Chuva_Media_Anual_mm", "NDVI_Medio")
    g.add_legend()
    g.fig.suptitle('Correlação NDVI vs. Pluviosidade por Região', y=1.02, fontsize=16)
    plt.savefig(os.path.join(output_dir, 'correlacao_regional_pluviosidade.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" -> Gráficos de correlação regional salvos.")
    
    # Retorna o DataFrame integrado para ser usado na próxima função.
    return df_integrado_estados

def mapa_correlacao_estadual(df_integrado_estados, estados_aoi, output_dir):
    """
    Gera um mapa do Brasil colorindo os estados pela força da correlação NDVI x Chuva.
    
    Args:
        df_integrado_estados (pd.DataFrame): DataFrame com dados integrados por estado.
        estados_aoi (gpd.GeoDataFrame): GeoDataFrame com a geometria dos estados do Brasil.
        output_dir (str): Caminho para salvar o mapa.
    """
    print("\n[Análise 3/3] Gerando mapa de correlação espacial por estado...")
    
    # --- INÍCIO DA CORREÇÃO E ROBUSTEZ ---
    # Calcula a correlação para cada estado individualmente.
    correlacoes = []
    # Itera sobre os dados agrupados por 'Sigla' de cada estado.
    for sigla, group in df_integrado_estados.groupby('Sigla'):
        
        # PASSO CRÍTICO: Remove linhas com dados ausentes (NaN) ANTES de calcular a correlação.
        # Isso evita o erro "ValueError: x and y must have the same length." se houver anos faltantes.
        clean_group = group.dropna(subset=['Chuva_Media_Anual_mm', 'NDVI_Medio'])
        
        # Garante que há dados suficientes para uma correlação minimamente significativa.
        # A correlação de Pearson com apenas 2 pontos sempre será +1 ou -1, o que não tem sentido.
        if len(clean_group) > 2:
            corr_chuva, _ = pearsonr(clean_group['Chuva_Media_Anual_mm'], clean_group['NDVI_Medio'])
            correlacoes.append({'Sigla': sigla, 'Correlacao_NDVI_Chuva': corr_chuva})
    
    # Converte a lista de dicionários em um DataFrame.
    df_corr = pd.DataFrame(correlacoes)

    # Verificação de segurança: se df_corr estiver vazio (nenhum estado tinha dados suficientes),
    # avisa o usuário e encerra a função para evitar um erro na etapa de merge.
    if df_corr.empty:
        print(" -> ATENÇÃO: Não foi possível calcular a correlação para nenhum estado.")
        print(" -> Verifique se os arquivos CSV de entrada possuem dados para pelo menos 3 anos por estado.")
        print(" -> Mapa de correlação espacial não será gerado.")
        return
    # --- FIM DA CORREÇÃO ---

    # Junta os dados de correlação com o shapefile dos estados.
    # O merge é feito entre a 'SIGLA_UF' do shapefile e a 'Sigla' do nosso DataFrame de correlações.
    mapa_corr = pd.merge(estados_aoi, df_corr, left_on='SIGLA_UF', right_on='Sigla')
    
    # --- CRIAÇÃO DO MAPA ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    mapa_corr.plot(
        column='Correlacao_NDVI_Chuva', # Coluna usada para colorir os estados.
        cmap='YlGn',                     # Paleta de cores (Amarelo-Verde), boa para valores positivos.
        linewidth=0.8,                   # Espessura da borda dos estados.
        ax=ax,                           # Eixo onde o mapa será desenhado.
        edgecolor='0.7',                 # Cor da borda dos estados (cinza claro).
        legend=True,                     # Mostra a legenda de cores.
        missing_kwds={                   # Configuração para estados sem dados.
            "color": "lightgrey",
            "label": "Dados insuficientes"
        },
        legend_kwds={                    # Configuração da legenda.
            'label': "Correlação de Pearson (r) entre NDVI e Chuva",
            'orientation': "horizontal"
        }
    )
    
    # Customização final do mapa.
    ax.set_title('Dependência da Vegetação à Chuva por Estado (2019-2024)', fontsize=16)
    ax.set_axis_off() # Remove os eixos X e Y para um visual de mapa limpo.
    
    plt.savefig(os.path.join(output_dir, 'mapa_correlacao_chuva_estados.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(" -> Mapa de correlação espacial salvo.")

# --- 4. BLOCO DE EXECUÇÃO PRINCIPAL ---
# Este bloco `if __name__ == "__main__":` garante que o código dentro dele só será executado
# quando o script for rodado diretamente (e não quando for importado por outro script).

if __name__ == "__main__":
    
    # Bloco try-except para tratamento de erros, especialmente se os arquivos de entrada não forem encontrados.
    try:
        # Define os caminhos completos para os arquivos de dados necessários.
        path_ndvi_nacional = os.path.join(INPUT_CSV_NDVI_DIR, 'estatisticas_anuais_ndvi.csv')
        path_clima_nacional = os.path.join(INPUT_CSV_CLIMA_DIR, 'estatisticas_climaticas_anuais.csv')
        
        # Carrega os dados nacionais.
        df_ndvi_nacional = pd.read_csv(path_ndvi_nacional, sep=';', decimal=',')
        df_clima_nacional = pd.read_csv(path_clima_nacional, sep=';', decimal=',')
        
        # Define os caminhos para os dados estaduais.
        path_ndvi_estadual = os.path.join(INPUT_CSV_NDVI_DIR, 'ndvi_medio_por_estado.csv')
        path_clima_estadual = os.path.join(INPUT_CSV_CLIMA_DIR, 'clima_anual_por_estado.csv')
        
        # Verifica se os arquivos estaduais existem antes de prosseguir.
        if not os.path.exists(path_ndvi_estadual) or not os.path.exists(path_clima_estadual):
             raise FileNotFoundError("Arquivos de dados estaduais não encontrados.")
             
        # Carrega o arquivo shapefile do Brasil, que contém as geometrias dos estados.
        estados_aoi = gpd.read_file(os.path.join(SHAPEFILE_DIR, "BR_UF_2022.shp"))

    except FileNotFoundError as e:
        # Se um arquivo não for encontrado, exibe uma mensagem de erro amigável e encerra o script.
        print(f"\nERRO CRÍTICO: Arquivo de estatísticas não encontrado. Detalhes: {e}")
        print("Por favor, certifique-se de que os scripts anteriores ('analise_ndvi.py' e 'analise_climatica.py') foram executados com sucesso.")
        exit() # Encerra o programa.

    # --- Execução Sequencial das Análises ---
    # Chama cada função de análise na ordem correta.
    
    # 1. Análise Nacional
    analise_correlacao_nacional(df_ndvi_nacional, df_clima_nacional, OUTPUT_PLOTS_DIR, OUTPUT_CSV_DIR)
    
    # 2. Análise Regional (retorna dados que serão usados na análise estadual)
    df_integrado_estados = analise_correlacao_regional(path_ndvi_estadual, path_clima_estadual, OUTPUT_PLOTS_DIR, OUTPUT_CSV_DIR)
    
    # 3. Análise e Mapeamento Estadual
    mapa_correlacao_estadual(df_integrado_estados, estados_aoi, OUTPUT_PLOTS_DIR)
    
    print("\n--- Análise integrada concluída com sucesso! Verifique a pasta 'analise_final_integrada'. ---")