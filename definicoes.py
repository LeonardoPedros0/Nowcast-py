import os
from tqdm.notebook import tqdm
# from alive_progress import alive_bar
import datetime
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import wradlib as wrl
from ipywidgets import interact
import warnings
import pyart
import numpy as np
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Suprimir avisos de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)
from matplotlib.colors import LinearSegmentedColormap

colors = [ '#60C8D5','#3C52A5','#67BD4E','#0B9444','#FEF200','#E1C524','#FDAF40','#E83338','#D82D33','#BF1E2E','#B6549F','#986EB0']#'#ffffff'
mycmap = LinearSegmentedColormap.from_list('Custom', colors, N=256)

### LEITURAS DOS DADOS ###
## LETURA BASE ##
def data_input(rbdict,slice=0):
  refl = rbdict['volume']['scan']['slice'][slice]['slicedata']['rawdata']['data']
  datadepth = float(rbdict['volume']['scan']['slice'][slice]['slicedata']['rawdata']['@depth'])
  datamin = float(rbdict['volume']['scan']['slice'][slice]['slicedata']['rawdata']['@min'])
  datamax = float(rbdict['volume']['scan']['slice'][slice]['slicedata']['rawdata']['@max'])
  refl = datamin + refl * (datamax - datamin) / 2 ** datadepth
  angle_fixed = rbdict['volume']['scan']['slice'][slice]['posangle']
  latitude = float(rbdict['volume']['sensorinfo']['lat'])
  longitude = float(rbdict['volume']['sensorinfo']['lon'])
  altitude = float(rbdict['volume']['sensorinfo']['alt'])
  time = rbdict['volume']['scan']['slice'][slice ]['slicedata']['@time']
  azi = rbdict['volume']['scan']['slice'][slice]['slicedata']['rayinfo']['data']
  azidepth =  float(rbdict['volume']['scan']['slice'][slice]['slicedata']['rayinfo']['@depth'])
  azirange = float(rbdict['volume']['scan']['slice'][slice]['slicedata']['rayinfo']['@rays'])
  anglestep = float(rbdict['volume']['scan']['pargroup']['anglestep'])
  azi =  azi * azirange * anglestep / 2**azidepth

  stoprange = float(rbdict['volume']['scan']['pargroup']['stoprange'])
  rangestep = float(rbdict['volume']['scan']['slice'][slice]['rangestep'])
  r = np.arange(125,refl.shape[1]*rangestep*1000,rangestep*1000)
  return refl, angle_fixed, datamin, datamax, latitude, longitude, altitude, time, azi,r

def data_input_cappi(rbdict,slice=0):
  while slice >=len(rbdict['volume']['scan']['slice']):
    slice -= 1
  refl = rbdict['volume']['scan']['slice'][slice]['slicedata']['rawdata']['data']
  datadepth = float(rbdict['volume']['scan']['slice'][slice]['slicedata']['rawdata']['@depth'])
  datamin = float(rbdict['volume']['scan']['slice'][slice]['slicedata']['rawdata']['@min'])
  datamax = float(rbdict['volume']['scan']['slice'][slice]['slicedata']['rawdata']['@max'])
  refl = datamin + refl * (datamax - datamin) / 2 ** datadepth

  return refl


## LEITURA SIMPLIFICADA ##
def select_data(datahora, fglob):
  arq = [vol for vol in fglob if datahora in vol][0]
  rbdict = wrl.io.read_rainbow(arq, loaddata=True)
  s = data_input(rbdict)
  refl= s[0]
  angle_fixed= s[1]
  datamin= s[2]
  datamax= s[3]
  latitude= s[4]
  longitude= s[5]
  altitude= s[6]
  time= s[7]
  azi= s[8]
  r= s[9]
  return refl,azi,r

## ORGANIZAÇÃO PARA DADOS POLARES ##
def My_Radar(refl, angle_fixed, datamin, datamax, latitude, longitude, altitude, time, azi,r):
  my_radar = {
      "time":
      {'units': 'seconds since 2019-01-08T'+time,
        'standard_name': 'time','long_name': 'time_in_seconds_since_volume_start',
        'calendar': 'gregorian',
        'comment': 'Coordinate variable for time. Time at the center of each ray, in fractional seconds since the global variable time_coverage_start',
        'data': np.arange(1,365*0.2,0.2)
        },
      "range":
      {
          'units': 'meters',
          'standard_name': 'projection_range_coordinate',
          'long_name': 'range_to_measurement_volume',
          'axis': 'radial_range_coordinate',
          'spacing_is_constant': 'true',
          'comment': 'Coordinate variable for range. Range to center of each bin.',
          'data': r,
          'meters_to_center_of_first_gate': r[0],
          'meters_between_gates': r[1]-r[0]
        },
      "fields":
      {
          'reflectivity': {'units': 'dBZ',
                            'standard_name': 'equivalent_reflectivity_factor',
                            'long_name': 'Reflectivity',
                            'valid_max': datamax,
                            'valid_min': datamin,
                            'coordinates': 'elevation azimuth range',
                            #'_FillValue': 0,
                            'data':refl
                            }
        },
      "metadata":{
          'Conventions': '',
          'version': '',
          'title': '',
          'institution': '',
          'references': '',
          'source': '',
          'history': '',
          'comment': '',
          'instrument_name': '',
          'original_container': '',
          #'vcp_pattern': 212
          },
      "scan_type":"ppi",
      "latitude":
      {
          'long_name': 'Latitude',
          'standard_name': 'Latitude',
          'units': 'degrees_north',
          'data':np.array([latitude],np.float32)
        },
      "longitude":
      {
          'long_name': 'Longitude',
          'standard_name': 'Longitude',
          'units': 'degrees_north',
          'data':np.array([longitude],np.float32)
        },
      "altitude":
      {
          'long_name': 'Altitude',
          'standard_name': 'Altitude',
          'units': 'meters',
          'positive': 'up',
          'data': np.array([altitude],np.float32)
        },
      "sweep_number":
      {
          'units': 'count',
          'standard_name': 'sweep_number',
          'long_name': 'Sweep number',
          'data': np.array([0],dtype=int)
      },
      "sweep_mode":
      {
          'units': 'unitless',
          'standard_name': 'sweep_mode',
          'long_name': 'Sweep mode',
          'comment': 'Options are: "sector", "coplane", "rhi", "vertical_pointing", "idle", "azimuth_surveillance", "elevation_surveillance", "sunscan", "pointing", "manual_ppi", "manual_rhi"',
          'data': np.array([b'azimuth_surveillance'])
      },
      "fixed_angle":
      {
          'long_name': 'Target angle for sweep',
          'units': 'degrees',
          'standard_name': 'target_fixed_angle',
          'data': np.array([angle_fixed],np.float32)
      },
      "sweep_start_ray_index":
      {
          'long_name': 'Index of first ray in sweep, 0-based',
          'units': 'count',
          'data': np.array([0],dtype=int)
      },
      "sweep_end_ray_index":
      {
          'long_name': 'Index of last ray in sweep, 0-based',
          'units': 'count',
          'data': np.array([719])
      },
      "azimuth":
      {
          'units': 'degrees',
          'standard_name': 'beam_azimuth_angle',
          'long_name': 'azimuth_angle_from_true_north',
          'axis': 'radial_azimuth_coordinate',
          'comment': 'Azimuth of antenna relative to true north',
          'data': azi
      },
      "elevation":
      {
          'units': 'degrees',
          'standard_name': 'beam_elevation_angle',
          'long_name': 'elevation_angle_from_horizontal_plane',
          'axis': 'radial_elevation_coordinate',
          'comment': 'Elevation of antenna relative to the horizontal plane',
          'data': np.full((360,),angle_fixed,dtype=np.float32)
      }
      }
  return my_radar

## ORGANIZAÇÃO PARA DADOS CARTESIANO ##
def transform_grid(grid, time_data=np.arange(1, 365 * 0.2, 0.2),
                   x_data=np.arange(-240000, 240000, 1000),y_data=np.arange(-240000, 240000, 1000),z_data=np.arange(0, 1),
                   latitude=None, longitude=None, altitude=None, datamax=None, datamin=None,time=None):
    if type(grid) == dict:
      dados_refl = grid['reflectivity'].data
    else:
      dados_refl = grid
    # Create the new grid dictionary
    new_grid = {
        'time': {
            'units': 'seconds since 2019-01-08T' + 'time',
            'standard_name': 'time',
            'long_name': 'time_in_seconds_since_volume_start',
            'calendar': 'gregorian',
            'comment': 'Coordinate variable for time. Time at the center of each ray, in fractional seconds since the global variable time_coverage_start',
            'data': time_data
        },
        'fields': {
            'reflectivity': {
                'units': 'dBZ',
                'standard_name': 'equivalent_reflectivity_factor',
                'long_name': 'Reflectivity',
                'valid_max': datamax,
                'valid_min': datamin,
                'coordinates': 'elevation azimuth range',
                'data': dados_refl
            }
        },
        'metadata': '',
        'origin_latitude': {'data': [latitude]},
        'origin_longitude': {'data': [longitude]},
        'origin_altitude': {'data': [altitude]},
        'x': {'data': x_data},
        'y': {'data': y_data},
        'z': {'data': z_data}
    }

    # Create the new Grid object
    new_grid_obj = pyart.core.Grid(
        time=new_grid['time'],
        fields=new_grid['fields'],
        metadata=new_grid['metadata'],
        origin_latitude=new_grid['origin_latitude'],
        origin_longitude=new_grid['origin_longitude'],
        origin_altitude=new_grid['origin_altitude'],
        x=new_grid['x'],
        y=new_grid['y'],
        z=new_grid['z']
    )

    return new_grid_obj

### PLOTS DO MAPAS ###
## PLOT POLAR ##
def plot_ppi(key):
  """ Plots ppi map of nexrad data given S3 key. """
  plt.clf()
  radar = key
  reflectivity= radar.fields['reflectivity']['data']
  mask = reflectivity <=0
  reflectivity[mask] = np.nan
  fig = plt.figure(figsize=(10, 10))
  ax = plt.axes(projection=ccrs.PlateCarree())
  display = pyart.graph.RadarMapDisplay(radar)
  display.plot_ppi_map('reflectivity', resolution='50m',
                     vmin=0,vmax=60,
                     sweep=0, fig=fig,ax=ax,
                     cmap=mycmap,
                     lat_lines=np.arange(-35, -20, 0.5),
                     lon_lines=np.arange(-53, -42, 1),
                     min_lon=-50,  max_lon=-42,
                     min_lat=-26,max_lat=-21,
                     lon_0=radar.longitude['data'][0],
                     lat_0=radar.latitude['data'][0],
                     )
  del display, radar

## PLOT CARTESIANO ##
def plot_grid_reflectivity(grid, field="reflectivity", level= 0, vmin=0, vmax=60, title=None,
 mycmap=mycmap,lat_lines = np.arange(-26, -21, 0.5),lon_lines = np.arange(-49, -42, 1.5),
 pathname_save =None,opt_title_fig={'t':'','fontweight':'bold','y':0.8},
                           opt_title_cbar={'label':'Refletividade (dBZ)'},
                           radar_center=(-23.600795,-45.97279), radius_km=130):


    # Verifica se grid é uma lista ou um único item
    if not isinstance(grid, list):
      grid = [grid]
    if title is None:
      title = ["Radar reflectivity " + str(i + 1) for i in range(len(grid))]
    n = len(grid)
    cols = 3 if n>1 else 1  # Aumentando o número de colunas
    rows = (n + cols - 1) // cols  # Número de linhas necessárias

    fig = plt.figure(figsize=(12, 6 * rows))
    # Itera sobre cada new_grid e cria subplots
    for i, new_grid in enumerate(grid):
      ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
      ds = new_grid.to_xarray()
      data = ds[field].data

      masked_data = np.ma.masked_outside(np.ma.masked_invalid(data), vmin, vmax)
      ds[field].data = masked_data

      # Plota pcolormesh
      pm = ds[field][0, level].plot.pcolormesh(
        x="lon", y="lat", cmap=mycmap, vmin=vmin, vmax=vmax,
        add_colorbar=False, ax=ax
      )

      # Adiciona features do Cartopy
      ax.add_feature(cfeature.STATES, edgecolor="gray", linewidth=0.5)
      ax.add_feature(cfeature.COASTLINE, edgecolor="gray", linewidth=0.5)

      for lat in lat_lines:
        ax.axhline(lat, color='gray', linestyle='--', linewidth=0.5)

      for lon in lon_lines:
        ax.axvline(lon, color='gray', linestyle='--', linewidth=0.5)

      # Define os ticks de latitude e longitude
      ax.set_xticks(lon_lines, crs=ccrs.PlateCarree())
      ax.set_yticks(lat_lines, crs=ccrs.PlateCarree())

      # Adiciona rótulos aos ticks
      ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}°E' if x >= 0 else f'{-x:.1f}°W'))
      ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}°S' if y < 0 else f'{y:.1f}°N'))

      # Título para cada subplot
      ax.set_title(f"{title[i]}")
      ax.set_xlabel("Longitude")
      ax.set_ylabel("Latitude")
      # Adicionar círculo ao gráfico
      center_lat, center_lon = radar_center
      radius_deg = radius_km / 111.0  # Converter o raio de km para graus
      circle = Circle(
          (center_lon, center_lat),  # Centro (longitude, latitude)
          radius_deg,                # Raio em graus
          color="k",               # Cor do círculo
          fill=False,                # Apenas o contorno
          linewidth=1,             # Espessura da linha
          transform=ccrs.PlateCarree()  # Transformação para coordenadas geográficas
      )
      ax.add_patch(circle)

    # Ajusta o layout
    plt.suptitle(**opt_title_fig)

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.50)

    # Adiciona a colorbar sempre, independentemente do número de gráficos
    cbar = plt.colorbar(pm, ax=fig.get_axes(),
                        orientation='horizontal' if n > 1 else 'vertical',
                        fraction=0.02 if n>1 else 0.035,
                        pad=0.06 if n > 1 else 0.02
                        )
    cbar.set_label(**opt_title_cbar,)  # Ajuste o rótulo conforme necessário
    if pathname_save:
      plt.savefig(pathname_save,bbox_inches='tight')
    plt.show()


### Interpolação ###
def load_lut_3d(filelut, nx, ny):
    nz = 2000 #200
    peso = np.zeros((nx, ny, nz), dtype=float)
    azim = np.zeros((nx, ny, nz), dtype=int)
    nbin = np.zeros((nx, ny, nz), dtype=int)
    elev = np.zeros((nx, ny, nz), dtype=int)
    count_lut = np.zeros((nx, ny), dtype=int)

    # Abre o arquivo LUT para leitura
    print(f'Lendo LUT --> {filelut}')
    with open(filelut, 'r') as file:
        for ii in range(nx):
          for jj in range(ny):
            line = file.readline().strip().split() # Lê uma linha do arquivo para os valores i, j, k
            if len(line): # verifica se é uma linha vazia
              i, j, k = int(line[0]), int(line[1]), int(line[2])
              # Armazena o número de bins no array count_lut
              count_lut[ii, jj] = k
              for kk in range(k):
                # if kk<=nz:
                line = file.readline().strip().split() # Lê os dados de wght, elevn, theta, range
                if len(line) != 4:
                  print(f'Erro na leitura do arquivo LUT\n{line}')
                # print(f'{k} / {kk}\t4->',line)
                wght = float(line[0])
                elevn = int(line[1])
                theta = int(line[2])
                range1 = int(line[3])

                # Preenchendo os arrays com os valores lidos
                elev[ii, jj, kk] = elevn - 1
                peso[ii, jj, kk] = wght
                azim[ii, jj, kk] = theta
                nbin[ii, jj, kk] = range1

    # Retorna os arrays preenchidos
    return peso, azim, nbin, elev, count_lut
import progressbar

def Pseudo_cappi(rbdict,nx, ny,peso, azim, nbin, elev, count_lut,nome_arq=''):

    pos_radar = np.where(count_lut > 0)
    ipos_radar, jpos_radar = pos_radar[:][0], pos_radar[:][1]
    radar_dbz = np.zeros((nx, ny))   # Refletividade do radar (dBZ)
    bar = progressbar.ProgressBar(max_value=len(ipos_radar), widgets=[
        progressbar.FormatLabel(nome_arq),
    ' [', progressbar.Percentage(), '] ',
    progressbar.Bar(),])

    for ii in range(len(ipos_radar)):
        i, j = ipos_radar[ii], jpos_radar[ii]
        k = count_lut[i, j]

        for l in range(k):
            alfa = elev[i, j, l]
            teta = azim[i, j, l]
            gate = nbin[i, j, l]
            wght = peso[i, j, l]
            try:
              dbz = data_input_cappi(rbdict, slice=alfa)[teta,gate]
            except:
              dbz = 0

            if dbz > 0:
                zeta = 10.0 ** (dbz / 10.0)
                radar_dbz[i, j] += zeta * wght


        if radar_dbz[i, j] > 0:
            radar_dbz[i, j] = 10.0 * np.log10(radar_dbz[i, j])
        bar.update(ii+1)

    return radar_dbz

def map_arq(azimuth,elevation):
  dict_mapeamento = {}

  for i,(azimt,elevs) in enumerate(zip(azimuth,elevation)):
    dict_mapeamento.setdefault(elevs, {})[int(azimt // 1)] = i
  return dict_mapeamento

def Pseudo_cappi_by_pyradar(radar= {'dbz':None,'vel':None,'zdr':None,'rho':None,'kdp':None},\
                            nx=120, ny=120,peso=None, azim=None, nbin=None, elev=None, count_lut=None,nome_arq=''):

    pos_radar = np.where(count_lut > 0)
    ipos_radar, jpos_radar = pos_radar[:][0], pos_radar[:][1]

    chuva_mmh = np.zeros((nx, ny))   # Taxa de chuva mm/h
    radar_dbz = np.zeros((nx, ny))   # Refletividade do radar (dBZ)
    radar_zdr = np.zeros((nx, ny))   # Campo ZDR (dBZ)
    radar_kdp = np.zeros((nx, ny))   # Campo KDP (graus/km)

    dict_mapeado = {k:map_arq(v.azimuth['data'],v.elevation['data']) for k,v in radar.items()}
    elevs={var:{chave:valor for chave,valor in enumerate(dict_mapeado[var].keys())} for var in radar.keys()}
    x=0
    for ii in tqdm(range(len(ipos_radar)),desc=nome_arq,leave=True):
        i, j = ipos_radar[ii], jpos_radar[ii]
        k = count_lut[i, j]

        for l in range(k):
            alfa = elev[i, j, l]
            teta = azim[i, j, l]
            gate = nbin[i, j, l]
            wght = peso[i, j, l]

            try:
              dbz = radar['dbz'].fields['reflectivity']['data'].data[dict_mapeado['dbz'][elevs['dbz'][alfa]][teta],gate]
              vel = radar['vel'].fields['velocity']['data'].data[dict_mapeado['vel'][elevs['vel'][alfa]][teta],gate]
              zdr = radar['zdr'].fields['differential_reflectivity']['data'].data[dict_mapeado['zdr'][elevs['zdr'][alfa]][teta],gate]
              rho = radar['rho'].fields['cross_correlation_ratio']['data'].data[dict_mapeado['rho'][elevs['rho'][alfa]][teta],gate]
              kdp = radar['kdp'].fields['specific_differential_phase']['data'].data[dict_mapeado['kdp'][elevs['kdp'][alfa]][teta],gate]

            except KeyError:
              x+=1
              dbz = 0

            if dbz > 0 and vel > -99 and -2 <= zdr <= 6 and rho > 0.9 and kdp >= -0.5:
                zeta = 10.0 ** (dbz / 10.0)
                zeta_zdr = 10.0 ** (zdr / 10.0)
                radar_dbz[i, j] += zeta * wght
                radar_zdr[i, j] += zeta_zdr * wght
                radar_kdp[i, j] += kdp * wght

                rzef = (zeta / 300.0) ** (1.0 / 1.4)
                rkdp = 44.0 * (abs(kdp) ** 0.822)

                if rzef < 6:
                    rain = rzef / (0.4 + 5.0 * (abs(zdr - 1.0) ** 1.3))
                elif 6 <= rzef < 50:
                    rain = rkdp / (0.4 + 3.5 * (abs(zdr - 1.0) ** 1.7))
                else:
                    rain = rkdp

                chuva_mmh[i, j] += rain * wght


        if radar_dbz[i, j] > 0:
            radar_dbz[i, j] = 10.0 * np.log10(radar_dbz[i, j])
            radar_zdr[i, j] = 10.0 * np.log10(radar_zdr[i, j])

    return radar_dbz,chuva_mmh


### NOWCASTING ###
def dt_from_str(date_time_str):
  try:
    year = int(date_time_str[0:4])
    month = int(date_time_str[4:6])
    day = int(date_time_str[6:8])
    hour = int(date_time_str[8:10])
    minute = int(date_time_str[10:12])
    second = int(date_time_str[12:14])
    return datetime.datetime(year, month, day, hour, minute, second)
  except (ValueError, IndexError):
    return None

def localiza_data(dados_pcappi_dbz,data_hora_fim, backsteps,sentido='tras',pysteps=False):
  if sentido == 'frente':
    ini = dados_pcappi_dbz.index([dbz for dbz in dados_pcappi_dbz  if data_hora_fim in dbz][0])
    lista_dados = dados_pcappi_dbz[ini:ini+backsteps]
    lista_name_dados = [os.path.basename(arq) for arq in dados_pcappi_dbz[ini:ini+backsteps]]
  elif sentido == 'tras':
    fim = dados_pcappi_dbz.index([dbz for dbz in dados_pcappi_dbz  if data_hora_fim in dbz][0])
    lista_dados = dados_pcappi_dbz[(fim+1)-backsteps:fim+1]
    lista_name_dados = [os.path.basename(arq) for arq in dados_pcappi_dbz[(fim+1)-backsteps:fim+1]]
  lista_dados.sort()
  arq_dbz = []
  for dbz in lista_dados:
    dado = np.load(dbz)
    dado[dado<=0] = 0
    arq_dbz.append(dado)
  if pysteps:
    return np.array(arq_dbz), lista_name_dados
  else:
    return np.array(arq_dbz)

def Create_metadata(accutime = 5, lista_name_arq=None,unit = 'db'):
  metadata = {'accutime': 5,
    'cartesian_unit': 'degrees',
    'institution': 'NOAA National Severe Storms Laboratory',
    'projection': '+proj=longlat  +ellps=IAU76',
    'threshold': 0.0125,
    'timestamps': np.array([dt_from_str(arq) for arq in lista_name_arq ], dtype=object),
    'transform': None,
    'unit': 'db' if unit == 'dbz' else 'mm/h',
    'x1': -132.5,
    'x2': -136.0,
    'xpixelsize': 1,
    'y1': -25.0,
    'y2': -22.0,
    'yorigin': 'upper',
    'ypixelsize': 0.5,
    'zerovalue': -15.0}
  return metadata


from rainymotion.metrics import MAE, CSI, FAR, POD, HSS, R

def Calc_metricas(list_data1,list_data2):
  """

  Args:
    list_data1: observado
    list_data2: simulado

  Returns: DataFrame contendo os indices e cada steps

  """
  df = pd.DataFrame(columns=['step','csi','far','pod','mae','hss','tss','r'])

  for i,(obs,sim) in enumerate(zip(list_data1,list_data2)):
    csi = CSI(obs,sim)
    far = FAR(obs,sim)
    pod = POD(obs,sim)
    mae = MAE(obs,sim)
    hss = HSS(obs,sim)
    tss = pod - far
    r = R(obs,sim)
    df.loc[len(df)] = [i+1,csi,far,pod, mae,hss,tss,r]
  return df

def ZR_Marshall(entrada,sentido='dbz_to_chuva'):
  if sentido == 'dbz_to_chuva':
    zeta = 10.0 ** (entrada / 10.0)
    res = (zeta / 300.0) ** (1.0 / 1.4)
  elif sentido == 'chuva_to_dbz':
    zeta = 300.0 * (entrada ** 1.4)
    res = 10.0 * np.log10(zeta)
    res[res<0] = 0 #res[res<17] = 0
    res = np.nan_to_num(res, nan=0)
  return res