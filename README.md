# 🌧️ **Comparação de Previsões de Nowcasting: Pysteps e Rainymotion**

## 📋 **Objetivos**  
Avaliar e comparar a precisão das previsões de chuva em curto prazo geradas pelas bibliotecas Python **Pysteps** e **Rainymotion**.  
Os principais objetivos incluem:  

- 🛠️ **Configuração de parâmetros**: Avaliar como diferentes configurações impactam os resultados.  
- ✅ **Validação de previsões**: Comparar previsões com dados reais de observação para análise de eficácia.  
- ⚡ **Análise de desempenho**: Identificar limitações de tempo de processamento e recursos necessários para aplicações em tempo real.  
- 🌦️ **Condições meteorológicas variadas**: Estudar a consistência dos resultados em diferentes cenários de precipitação (chuva leve e intensa).  

O objetivo final é compreender as forças e limitações de cada biblioteca em cenários distintos.  

---

## 📊 **Dados Utilizados**  

### 🌐 **Radar SPOL**  
- **Período analisado**: 08 de janeiro de 2019, das 00:00 UTC às 23:55 UTC.  
- **Formato dos arquivos**: `.vol` (intervalos de 5 minutos).  
- **Localização do radar**:  
  - **Município**: Biritiba Mirim, São Paulo.  
  - **Coordenadas**: 23.600795°S, 45.97279°W.  
  - **Altitude**: 928 m.  
- **Especificações técnicas**:  
  - Banda S (2.7 a 2.9 GHz).  
  - GATES: 250 m.  
  - Alcance máximo: 240 km.  
  - Azimute: 0° a 360°.  
  - Range: 125 m a 240 km.  

### 🌧️ **Outras Fontes de Dados**  
- **Instituto Nacional de Meteorologia (INMET)**.  
- **Estação meteorológica do Instituto de Astronomia, Geofísica e Ciências Atmosféricas (IAG-USP)**.  

---

## ⚙️ **Processamento dos Dados**  

1. 🧭 **Conversão de Coordenadas**  
   - Conversão de coordenadas polares para cartesianas.  
   - Geração de indicadores CAPPI (Constant Altitude Plan Position Indicator) a 3 km de altitude média.  

2. 🛠️ **Filtragem de Dados**  
   - **Refletividade**: `dbz > 0`.  
   - **Velocidade radial**: `vel > -99`.  
   - **Diferença específica de refletividade**: `-2 ≤ zdr ≤ 6`.  
   - **Coeficiente de correlação**: `rho > 0.9`.  
   - **Taxa de fase específica**: `kdp ≥ -0.5`.  

---

## 🚀 **Instalação das Dependências**  

### 📂 **Montagem do Google Drive (se aplicável)**  
```bash
from google.colab import drive
drive.mount('/content/drive')
```

### 📦 **Instalação de Bibliotecas**  
```bash
# Py-ART para manipulação de dados de radar
pip install git+https://github.com/ARM-DOE/pyart.git

# Biblioteca boto (AWS)
pip install boto

# wradlib para processamento de dados de radar
pip install wradlib

# TINT para segmentação de tempestades
git clone https://github.com/openradar/TINT.git
pip install -e /content/TINT

# Cartopy para visualização geográfica
pip install cartopy

# Biblioteca Rainymotion para previsões de nowcasting
git clone https://github.com/hydrogo/rainymotion.git
cd rainymotion
pip install .

# Biblioteca Pysteps para previsões avançadas de nowcasting
pip install pysteps
```

---

## 📁 **Estrutura do Projeto**  
```plaintext
📦 Projeto Nowcasting
├── 📄 defs.py              # Funções personalizadas para suporte ao código
├── 📄 Nowcasting.ipynb     # Código principal para execução e análise
├── 📂 Dados                # Pasta contendo os dados utilizados no projeto
│   ├── 🌐 Radar            # Arquivos do radar Doppler
│   │   ├── 20190108_0000.vol
│   │   ├── 20190108_0005.vol
│   │   ├── ...             # Mais arquivos de radar (.vol)
│   └── 📄 Estacoes         # Dados de estações meteorológicas (INMET, IAG-USP)
├── 📜 README.md            # Documentação do projeto
```

---

## 📚 **Principais Bibliotecas Utilizadas**  
- 🛰️ **wradlib** e **pyart**: Manipulação de dados de radar meteorológico.  
- 📈 **pysteps** e **rainymotion**: Previsões de nowcasting.  
- 🗺️ **cartopy**: Visualização e mapeamento de dados.  
- 📊 **matplotlib** e **numpy**: Análise e visualização de dados.  

---

## ✍️ **Autor**  

- **Nome**: Leonardo Pedroso  
- **E-mail**: [l.pedroso@usp.br](mailto:l.pedroso@usp.br)  
- **GitHub**: [LeonardoPedros0](https://github.com/LeonardoPedros0)  

---

## 📜 **Referências**  
- Rocha et al., 2023  
- Morales et al., 2016  
