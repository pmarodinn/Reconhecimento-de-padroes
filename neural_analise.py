# =================================================================================================
# SCRIPT 3: neural_analise.py (VERSÃO MULTI-REGIÃO E COMENTADA)
#
# OBJETIVO:
# Este é o script de análise mais avançado do projeto. Ele utiliza uma rede neural profunda
# (Deep Learning) para detectar anomalias na série temporal de NDVI. O objetivo não é
# apenas ver a mudança entre dois anos, mas identificar locais cuja trajetória de
# vegetação ao longo de todo o período (2019-2024) foi "inesperada" ou "anormal".
# Isso permite detectar focos de mudança abrupta, como desmatamento, incêndios ou
# grandes obras.
#
# A análise é realizada de forma independente para cada uma das 5 grandes regiões do Brasil,
# treinando um modelo "especialista" para cada realidade de bioma.
#
# Os principais produtos gerados são:
# 1.  Mapas de Erro de Reconstrução (.tif): Um mapa para cada região onde o valor de cada
#     pixel representa a magnitude da anomalia detectada pela rede neural.
# 2.  Mapas de Mudança Significativa (.png, .tif): Um mapa binário para cada região que
#     destaca apenas os pixels com os maiores erros (o top 5%), apontando os hotspots
#     de mudança mais relevantes.
#
# COMO FUNCIONA:
# A metodologia utilizada é um Autoencoder Convolucional LSTM (ConvLSTM) não supervisionado.
#
# - Abordagem Geral (Autoencoder Não Supervisionado):
#   A rede não é ensinada com exemplos de "desmatamento". Em vez disso, ela é treinada
#   para uma tarefa simples: reconstruir a própria sequência de imagens que ela recebe como
#   entrada. Ao ser forçada a aprender a comprimir e descompactar os dados, ela se torna
#   especialista nos padrões temporais e espaciais mais comuns (a "normalidade").
#   A detecção de mudança ocorre ao identificar onde a rede falha: um alto erro de
#   reconstrução indica que o padrão real era tão anômalo que o modelo não conseguiu
#   recriá-lo com base no que aprendeu.
#
# - Etapa 1: Preparação dos Dados Regionais:
#   1. O script carrega todos os mosaicos anuais de NDVI do Brasil.
#   2. Carrega o shapefile das Unidades Federativas e usa a função `geopandas.dissolve`
#      para agrupar os estados e criar as 5 geometrias das grandes regiões.
#   3. Inicia um loop principal, iterando sobre cada uma das 5 regiões.
#   4. Para cada região, recorta a pilha nacional de dados NDVI usando `rasterio.mask.mask`,
#      gerando uma série temporal de imagens exclusiva para aquela região.
#
# - Etapa 2: Treinamento do Modelo Especialista (por Região):
#   1. A série temporal da região é segmentada em pequenos "patches" (recortes 3D de 6x32x32).
#   2. Um novo modelo `ConvLSTMAutoencoder` é instanciado para cada região.
#   3. O modelo é treinado por 50 épocas utilizando os patches daquela região. O objetivo é
#      minimizar o Erro Quadrático Médio (MSE) entre os patches originais e os reconstruídos.
#
# - Etapa 3: Inferência e Cálculo do Mapa de Erro:
#   1. Com o modelo treinado, ele é colocado em modo de avaliação (`model.eval()`).
#   2. O script alimenta toda a série de patches da região no modelo e calcula, para cada
#      pixel, o erro de reconstrução.
#   3. Esses valores de erro são montados em um novo mapa 2D, onde a intensidade de cada
#      pixel corresponde ao quão anômala foi sua evolução temporal.
#
# - Etapa 4: Geração dos Resultados Finais:
#   1. O mapa de erro contínuo é salvo como um GeoTIFF para análises futuras.
#   2. Um limiar estatístico (o percentil 95) é aplicado sobre o mapa de erro para isolar
#      apenas as anomalias mais extremas.
#   3. O resultado é um mapa binário (0 para "normal", 1 para "mudança significativa"), que
#      é salvo como um GeoTIFF e como uma imagem PNG para fácil visualização.
# =================================================================================================

# --- Importação de Bibliotecas ---
# Bibliotecas padrão para manipulação de dados e sistema operacional
import os
import glob
import re
# Bibliotecas científicas e de dados
import numpy as np
import pandas as pd
import geopandas as gpd
# Bibliotecas de geoprocessamento
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
# Bibliotecas de Deep Learning (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# Bibliotecas de visualização e progresso
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. CONFIGURAÇÃO DA ANÁLISE ---
# Esta seção centraliza todos os parâmetros que podem ser ajustados pelo usuário.

# Parâmetros de Treinamento da Rede Neural
EPOCHS = 50          # Número de ciclos completos de treinamento sobre o dataset. Mais épocas podem levar a um modelo mais preciso, mas aumentam o tempo de processamento.
LEARNING_RATE = 1e-4 # Taxa de aprendizado do otimizador. Controla o tamanho do "passo" que o modelo dá para corrigir seus erros.
BATCH_SIZE = 16      # Número de amostras (patches) processadas de uma só vez. Um batch size maior acelera o treino, mas consome mais memória (RAM/VRAM).

# Parâmetros dos Dados de Entrada
PATCH_SIZE = 32      # Tamanho em pixels (altura e largura) dos recortes de imagem que alimentarão a rede.

# Definição das Pastas de Entrada e Saída
GEOTIFF_DIR = "resultados/geotiff_maps"     # Pasta onde os mosaicos anuais de NDVI estão salvos.
SHAPEFILE_DIR = "./BR_UF_2022/"              # Pasta que contém o shapefile dos estados do Brasil.
OUTPUT_DIR = "resultados_neural"            # Pasta principal onde todos os resultados desta análise serão salvos.
os.makedirs(OUTPUT_DIR, exist_ok=True)      # Cria a pasta de saída se ela não existir.

# Configuração do Dispositivo de Hardware (GPU ou CPU)
# O PyTorch pode utilizar a GPU (placa de vídeo, via CUDA) para acelerar massivamente o treinamento.
# Este código detecta automaticamente se uma GPU compatível está disponível e a utiliza. Caso contrário, usa a CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de processamento: {DEVICE}")

# --- 2. FUNÇÕES AUXILIARES E CLASSES DE DADOS ---

def load_and_stack_geotiffs(geotiff_dir):
    """
    Carrega todos os GeoTIFFs anuais de NDVI, os empilha em um único array NumPy
    e retorna o perfil geoespacial do primeiro arquivo.
    """
    print("Carregando e empilhando GeoTIFFs anuais...")
    geotiff_files = sorted(glob.glob(os.path.join(geotiff_dir, 'mosaico_ndvi_brasil_*.tif')))
    if not geotiff_files:
        raise FileNotFoundError("Nenhum GeoTIFF de mosaico encontrado na pasta 'resultados/geotiff_maps'.")
    
    stack = []
    profile = None
    # Itera sobre cada arquivo GeoTIFF encontrado
    for f in geotiff_files:
        with rasterio.open(f) as src:
            stack.append(src.read(1))  # Lê a primeira banda (NDVI) e a adiciona à lista
            if profile is None: 
                profile = src.profile # Salva os metadados (CRS, transform, etc.) do primeiro arquivo
    
    # Converte a lista de arrays 2D em um único array 3D (tempo, altura, largura)
    return np.stack(stack, axis=0), profile, geotiff_files

def create_patches(data, patch_size):
    """
    Segmenta um grande array de série temporal em pequenos 'patches' (recortes) 3D.
    A rede neural aprende com esses patches, não com a imagem inteira de uma vez.
    """
    seq_len, height, width = data.shape
    patches = []
    # Itera sobre a imagem, criando recortes de `patch_size` x `patch_size`
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = data[:, i:i + patch_size, j:j + patch_size]
            # Adiciona o patch à lista apenas se ele contiver dados válidos (não for inteiramente NaN)
            if not np.all(np.isnan(patch)):
                patches.append(patch)
    return np.array(patches)

class SpatiotemporalDataset(Dataset):
    """
    Classe customizada do PyTorch para gerenciar o nosso dataset de patches.
    Esta classe organiza os dados e os prepara para serem carregados pela rede.
    """
    def __init__(self, data_patches):
        # Lida com valores NaN (sem dados), que não podem ser processados pela rede.
        nan_mask = np.isnan(data_patches)
        data_patches[nan_mask] = -1 # Substituição temporária
        
        # Normaliza os dados: o NDVI varia de -1 a 1. A rede funciona melhor com dados entre 0 e 1.
        self.data = (data_patches + 1) / 2
        self.data[nan_mask] = 0 # Define os locais que eram NaN como 0 após a normalização.
        
    def __len__(self):
        # Retorna o número total de amostras (patches) no dataset.
        return len(self.data)

    def __getitem__(self, idx):
        # Retorna uma única amostra (patch) do dataset.
        sample = self.data[idx]
        # Converte o array NumPy para um Tensor do PyTorch e adiciona uma dimensão de "canal",
        # pois as camadas convolucionais esperam um input com 4 dimensões: (batch, canais, altura, largura).
        # Para o autoencoder, a entrada (input) e o alvo (target) são os mesmos.
        sample = torch.from_numpy(sample).unsqueeze(1).float()
        return sample, sample

# --- 3. ARQUITETURA DA REDE NEURAL ---

class ConvLSTMCell(nn.Module):
    """
    Define a célula ConvLSTM, o bloco de construção fundamental do nosso modelo.
    Ela combina a convolução (para análise espacial) com a memória de uma LSTM (para análise temporal).
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        # A camada convolucional processa a entrada atual e o estado oculto anterior juntos.
        # O número de canais de saída é 4x o desejado, pois a LSTM calcula 4 "portões" (gates) internos.
        self.conv = nn.Conv2d(in_channels + out_channels, 4 * out_channels, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        # Concatena a entrada atual (x) e o estado oculto anterior (h_prev) na dimensão dos canais.
        combined = torch.cat([x, h_prev], dim=1)
        # Passa os dados combinados pela camada convolucional de uma só vez.
        gates = self.conv(combined)
        
        # Divide o resultado nos 4 portões da LSTM: Input, Forget, Output, e Gate.
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        # Aplica funções de ativação para regular o fluxo de informação.
        i = torch.sigmoid(i) # Portão de entrada: decide o que é novo para ser lembrado.
        f = torch.sigmoid(f) # Portão de esquecimento: decide o que deve ser esquecido da memória antiga.
        o = torch.sigmoid(o) # Portão de saída: decide qual parte da memória vai para o estado oculto de saída.
        g = torch.tanh(g)    # Portão de "gate": modula a nova informação a ser adicionada.
        
        # Calcula o novo estado da célula (a memória de longo prazo)
        c_next = f * c_prev + i * g
        # Calcula o novo estado oculto (a memória de trabalho/saída)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMAutoencoder(nn.Module):
    """
    Define a arquitetura completa do Autoencoder, unindo o Encoder e o Decoder.
    """
    def __init__(self):
        super(ConvLSTMAutoencoder, self).__init__()
        # Encoder: uma célula ConvLSTM que aprende a comprimir a sequência de imagens
        # de 1 canal de entrada para um estado latente de 16 canais (features).
        self.encoder_cell = ConvLSTMCell(1, 16, 3, padding=1)
        
        # Decoder: outra célula que recebe o estado latente e aprende a reconstruir a sequência.
        # Ela trabalha com 16 canais de features para manter a riqueza da informação.
        self.decoder_cell = ConvLSTMCell(16, 16, 3, padding=1)
        
        # Camada de saída: uma convolução 2D padrão que converte as 16 features do decoder
        # de volta para uma imagem de 1 canal, que é a nossa imagem reconstruída.
        self.output_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # x tem a forma (tempo, batch, canais, altura, largura)
        seq_len = x.size(0)
        batch_size, _, height, width = x.size(1), x.size(2), x.size(3), x.size(4)
        
        # --- Fase do Encoder ---
        # Inicializa os estados oculto (h) e da célula (c) com tensores de zeros.
        h, c = (torch.zeros(batch_size, 16, height, width).to(DEVICE),
                torch.zeros(batch_size, 16, height, width).to(DEVICE))
        # Processa a sequência temporalmente, atualizando os estados a cada passo.
        for t in range(seq_len):
            h, c = self.encoder_cell(x[t], h, c)
        
        # Ao final do loop, 'h' e 'c' contêm a "essência" de toda a sequência de entrada.
        
        # --- Fase do Decoder ---
        outputs = []
        # Usa o estado final do encoder como o estado inicial do decoder.
        h_dec, c_dec = h, c
        
        for t in range(seq_len):
            # O input para cada passo do decoder é o estado oculto final do encoder.
            # Isso força a rede a reconstruir a sequência inteira a partir dessa "memória" comprimida.
            h_dec, c_dec = self.decoder_cell(h, h_dec, c_dec)
            # A camada de convolução final transforma as features do estado oculto em uma imagem.
            output_step = self.output_conv(h_dec)
            outputs.append(output_step)
            
        # Empilha as saídas de cada passo para formar a sequência reconstruída.
        outputs = torch.stack(outputs, dim=0)
        return outputs

# --- 4. BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == "__main__":
    
    # Carrega os dados de NDVI e os metadados geoespaciais.
    full_stack, profile, geotiff_files = load_and_stack_geotiffs(GEOTIFF_DIR)
    
    # Agrupa os estados do Brasil em suas respectivas grandes regiões.
    try:
        print("Agrupando estados por região...")
        estados_shp = gpd.read_file(os.path.join(SHAPEFILE_DIR, "BR_UF_2022.shp"))
        regioes_shp = estados_shp.dissolve(by='NM_REGIAO')
        regioes_shp.reset_index(inplace=True)
        print(f"Regiões encontradas: {regioes_shp['NM_REGIAO'].tolist()}")
    except Exception as e:
        print(f"Erro ao agrupar regiões: {e}. Verifique o shapefile 'BR_UF_2022.shp'.")
        exit()
    
    # Loop principal que executa a análise completa para cada região.
    for index, regiao in regioes_shp.iterrows():
        nome_regiao = regiao['NM_REGIAO']
        geom_regiao = gpd.GeoDataFrame([regiao], crs=estados_shp.crs)

        print(f"\n============================================================")
        print(f"   INICIANDO ANÁLISE PARA A REGIÃO: {nome_regiao}")
        print(f"============================================================")

        # Recorta os dados nacionais para a geometria da região atual.
        try:
            with rasterio.open(geotiff_files[0]) as src:
                raster_crs = src.crs
            geom_regiao_reprojected = geom_regiao.to_crs(raster_crs)
            with rasterio.open(geotiff_files[0]) as src:
                roi_stack, roi_transform = mask(src, geom_regiao_reprojected.geometry, crop=True)
            current_profile = profile.copy()
            current_profile.update(height=roi_stack.shape[1], width=roi_stack.shape[2], transform=roi_transform)
            
            roi_data = []
            for f in geotiff_files:
                with rasterio.open(f) as src:
                    masked_data, _ = mask(src, geom_regiao_reprojected.geometry, crop=True)
                    roi_data.append(masked_data[0])
            roi_data = np.stack(roi_data, axis=0)
        except Exception as e:
            print(f"Erro ao processar a ROI para {nome_regiao}: {e}. Pulando.")
            continue

        # Prepara os dados para o PyTorch
        print(f"Criando patches de {PATCH_SIZE}x{PATCH_SIZE} pixels...")
        patches = create_patches(roi_data, PATCH_SIZE)
        if len(patches) == 0:
            print(f" -> Nenhum patch com dados válidos encontrado para {nome_regiao}. Pulando.")
            continue
        dataset = SpatiotemporalDataset(patches)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Treinamento do Modelo
        print(f"\nIniciando treinamento da rede neural para a Região {nome_regiao}...")
        model = ConvLSTMAutoencoder().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        for epoch in range(EPOCHS):
            model.train()
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} ({nome_regiao})")
            total_loss = 0
            for i, (data, _) in enumerate(loop):
                data = data.permute(1, 0, 2, 3, 4).to(DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loop.set_postfix(loss=total_loss / (i + 1))
        
        print("Treinamento concluído.")
        
        # Inferência e Geração dos Mapas de Saída
        print("Gerando mapa de erro de reconstrução detalhado (pixel a pixel)...")
        model.eval()
        error_map = np.zeros((roi_data.shape[1], roi_data.shape[2]))
        
        with torch.no_grad():
            patches_para_inferencia = create_patches(roi_data, PATCH_SIZE)
            full_dataset = SpatiotemporalDataset(patches_para_inferencia)
            full_dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False)
            patch_idx = 0
            for data, _ in tqdm(full_dataloader, desc=f"Calculando erro ({nome_regiao})"):
                data_permuted = data.permute(1, 0, 2, 3, 4).to(DEVICE)
                reconstructed = model(data_permuted)
                pixel_wise_error = torch.mean((data_permuted - reconstructed) ** 2, dim=0).squeeze().cpu().numpy()
                row = (patch_idx // (error_map.shape[1] // PATCH_SIZE)) * PATCH_SIZE
                col = (patch_idx % (error_map.shape[1] // PATCH_SIZE)) * PATCH_SIZE
                if row + PATCH_SIZE <= error_map.shape[0] and col + PATCH_SIZE <= error_map.shape[1]:
                    error_map[row:row+PATCH_SIZE, col:col+PATCH_SIZE] = pixel_wise_error
                patch_idx += 1
        
        # Salva o mapa de erro (o resultado bruto da análise)
        error_profile = current_profile.copy(); error_profile.update(dtype=rasterio.float32, count=1)
        error_map_path = os.path.join(OUTPUT_DIR, f'mapa_erro_reconstrucao_{nome_regiao}.tif')
        with rasterio.open(error_map_path, 'w', **error_profile) as dst:
            dst.write(error_map.astype(rasterio.float32), 1)
        print(f" -> Mapa de erro salvo em: {error_map_path}")
        
        # Cria e salva o mapa binário de mudanças (apenas as anomalias mais extremas)
        threshold = np.percentile(error_map[error_map > 0], 95)
        change_map = (error_map > threshold).astype(np.uint8)
        
        change_map_path_tif = os.path.join(OUTPUT_DIR, f'mapa_mudancas_{nome_regiao}.tif')
        change_profile = current_profile.copy(); change_profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        with rasterio.open(change_map_path_tif, 'w', **change_profile) as dst:
            dst.write(change_map, 1)
        print(f" -> Mapa de mudanças (GeoTIFF) salvo em: {change_map_path_tif}")
        
        # Salva uma imagem de visualização do mapa de mudanças
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(change_map, cmap='inferno', interpolation='nearest')
        ax.set_title(f'Áreas com Mudanças Significativas Detectadas ({nome_regiao})', fontsize=16)
        ax.set_xticks([]); ax.set_yticks([])
        change_map_path_png = os.path.join(OUTPUT_DIR, f'mapa_mudancas_{nome_regiao}.png')
        plt.savefig(change_map_path_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f" -> Mapa de mudanças (PNG) salvo em: {change_map_path_png}")

    print("\nAnálise neural multi-região concluída!")