'''
Function: analysis for WRF 2011 May-Aug outputs, as in Ma CAUSES paper Figure 15.
Date: 20200325
'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas
import matplotlib.dates as mdates
import scipy.stats
from sklearn.linear_model import LinearRegression

label_string1 = "_Morr"
label_string2 = "_Thom"
label_string3 = "_new"

#ds_WRF = xr.open_dataset('/home/qin5/Data/WRF.postprocessing.extract.hourly.nc')
ds_WRF = xr.open_dataset('/home/qin5/Data/WRF.postprocessing.extract.hourly.Morri.nc')
ds_WRF_Thom = xr.open_dataset('/home/qin5/Data/WRF.postprocessing.extract.hourly.Thom.05678.nc')

ds_WRF_XD1 = xr.open_dataset('/home/qin5/Data/WRF_Xiaodong_new/Xiaodong_WRF.postprocessing.extract.hourly.1x1.conserve.nc')
ds_WRF_XD2 = xr.open_dataset('/home/qin5/Data/WRF_Xiaodong_new/Xiaodong_WRF.postprocessing.extract.hourly.T2.1x1.bilinear.nc')

ds_SM_WRF = xr.open_dataset('/home/qin5/Data/WRF.postprocessing.extract.hourly.SMOIS.nc')
ds_SM_WRF_XD = xr.open_dataset('/home/qin5/Data/WRF_Xiaodong_new/Xiaodong_WRF.postprocessing.extract.hourly.SMOIS.1x1.bilinear.nc')

ds_ARMBE2D_05 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110501.000000.nc')
ds_ARMBE2D_06 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110601.000000.nc')
ds_ARMBE2D_07 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110701.000000.nc')
ds_ARMBE2D_08 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110801.000000.nc')

ds_pr_stage4 = xr.open_dataset('/home/qin5/Data/Precip_StageIV/Precip_Stage_IV.2011045678.postprocessing.extract.hourly.nc')

ds_GLEAM = xr.open_dataset('/home/qin5/Data/GLEAM/E_2011_GLEAM.processed.daily.nc')

## Figure 8 in Ma et al 2018, average over 35-38N, 99-96W, consistent with ARMBE2D from Qi Tang
lat_1 = 35.0
lat_2 = 38.0
lon_1 = -99.0
lon_2 = -96.0

### WRF calculate daily mean at ARM SGP site 
RAIN_tot_regrid = ds_WRF['RAIN_tot_regrid']
RAIN_WRF_daily = RAIN_tot_regrid.resample(time='1D').mean(dim='time')
RAIN_WRF_SGP = RAIN_WRF_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
RAIN_WRF_SGP = RAIN_WRF_SGP * 24.0 # from mm/hr to mm/day
RAIN_WRF_SGP.attrs['units'] = "mm/day"
#print(RAIN_WRF_SGP)

#### accumulative rain
RAIN_WRF_ACC = np.asarray([RAIN_WRF_SGP[0:i].values.sum() for i in np.arange(0,122,1)])
RAIN_WRF_ACC = xr.DataArray(RAIN_WRF_ACC, dims=('time'), coords = {'time':RAIN_WRF_SGP.coords['time'] })
RAIN_WRF_ACC.attrs['units'] = "mm"
RAIN_WRF_ACC.attrs['long_name'] = "accumulated total precip"
#print(RAIN_WRF_ACC)

### -------- calculate evaporation from latent heat
Lv_water = 2264705.0 # J/kg

LH_regrid = ds_WRF['LH_regrid']
LH_WRF_daily = LH_regrid.resample(time='1D').mean(dim='time')
LH_WRF_SGP = LH_WRF_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
LH_WRF_SGP_W_m2 = LH_WRF_SGP
LH_WRF_SGP = LH_WRF_SGP * 3600.0*24.0/Lv_water # from W/m2 to mm/day
LH_WRF_SGP.attrs['units'] = "mm/day"
LH_WRF_SGP.attrs['long_name'] = "ET converted from latent heat flux, mm/day"
#print(LH_WRF_SGP)

#### accumulative evaporation
evap_WRF_ACC = np.asarray([LH_WRF_SGP[0:i].values.sum() for i in np.arange(0,122,1)])
evap_WRF_ACC = xr.DataArray(evap_WRF_ACC, dims=('time'), coords = {'time':LH_WRF_SGP.coords['time'] })
evap_WRF_ACC.attrs['units'] = "mm"
evap_WRF_ACC.attrs['long_name'] = "accumulated ET, converted from latent heat flux"
#print(evap_WRF_ACC)

### soil moisture at 5cm depth 
SMOIS_regrid = ds_SM_WRF['SMOIS_regrid'][:,0,:,:]   # depth 0 is 5-cm
SMOIS_WRF_daily = SMOIS_regrid.resample(time='1D').mean(dim='time')
SMOIS_WRF_SGP = SMOIS_WRF_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
SMOIS_WRF_SGP.attrs['units'] = "m3/m3"
#print(SMOIS_WRF_SGP)

### soil moisture at 0.25 depth
SMOIS25_regrid = ds_SM_WRF['SMOIS_regrid'][:,1,:,:]   # depth 0 is 25-cm
SMOIS25_WRF_daily = SMOIS25_regrid.resample(time='1D').mean(dim='time')
SMOIS25_WRF_SGP = SMOIS25_WRF_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
SMOIS25_WRF_SGP.attrs['units'] = "m3/m3"
#print(SMOIS25_WRF_SGP)

### evaporative fraction LH/(SH+LH)
HFX_regrid = ds_WRF['HFX_regrid']
EF_regrid = LH_regrid / (HFX_regrid+LH_regrid)
EF_regrid = EF_regrid.where( (HFX_regrid+LH_regrid) > 10.0)  # to avoid unrealistic values when denominator is too small 

EF_WRF_daily = EF_regrid.resample(time='1D').mean(dim='time')
EF_WRF_SGP = EF_WRF_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
EF_WRF_SGP.attrs['units'] = "unitless"
#print(EF_WRF_SGP)

### Sensible heat flux
HFX_WRF_daily = HFX_regrid.resample(time='1D').mean(dim='time')
HFX_WRF_SGP = HFX_WRF_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
HFX_WRF_SGP.attrs['units'] = "unitless"
#print(EF_WRF_SGP)

### T2m
T2_regrid = ds_WRF['T2_regrid']
T2_WRF_daily = T2_regrid.resample(time='1D').mean(dim='time')
T2_WRF_SGP = T2_WRF_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
#print(T2_WRF_SGP)


### ======================== WRF_Thom
RAIN_tot_regrid_Thom = ds_WRF_Thom['RAIN_tot_regrid']
RAIN_WRF_Thom_daily = RAIN_tot_regrid_Thom.resample(time='1D').mean(dim='time')
RAIN_WRF_Thom_SGP = RAIN_WRF_Thom_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
RAIN_WRF_Thom_SGP = RAIN_WRF_Thom_SGP * 24.0 # from mm/hr to mm/day
RAIN_WRF_Thom_SGP.attrs['units'] = "mm/day"
#print(RAIN_WRF_Thom_SGP)

#### accumulative rain
#RAIN_WRF_Thom_ACC = np.asarray([RAIN_WRF_Thom_SGP[0:i].values.sum() for i in np.arange(0,12,1)])
#--------uncomment
## Note the Zhe WRF simulation WRF_Thompson only goes to 08-27, and the first 9 hours of 08-28.
RAIN_WRF_Thom_ACC = np.asarray([RAIN_WRF_Thom_SGP[0:i].values.sum() for i in np.arange(0,120,1)])
#-----------
RAIN_WRF_Thom_ACC = xr.DataArray(RAIN_WRF_Thom_ACC, dims=('time'), coords = {'time':RAIN_WRF_Thom_SGP.coords['time'] })
RAIN_WRF_Thom_ACC.attrs['units'] = "mm"
RAIN_WRF_Thom_ACC.attrs['long_name'] = "accumulated total precip"

### -------- calculate evaporation from latent heat
LH_regrid_Thom = ds_WRF_Thom['LH_regrid']
LH_WRF_Thom_daily = LH_regrid_Thom.resample(time='1D').mean(dim='time')
LH_WRF_Thom_SGP = LH_WRF_Thom_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
LH_WRF_Thom_SGP_W_m2 = LH_WRF_Thom_SGP
LH_WRF_Thom_SGP = LH_WRF_Thom_SGP * 3600.0*24.0/Lv_water # from W/m2 to mm/day
LH_WRF_Thom_SGP.attrs['units'] = "mm/day"
LH_WRF_Thom_SGP.attrs['long_name'] = "ET converted from latent heat flux, mm/day"

#### accumulative evaporation
#evap_WRF_Thom_ACC = np.asarray([LH_WRF_Thom_SGP[0:i].values.sum() for i in np.arange(0,61,1)])
#------uncomment
evap_WRF_Thom_ACC = np.asarray([LH_WRF_Thom_SGP[0:i].values.sum() for i in np.arange(0,120,1)])
#-------

evap_WRF_Thom_ACC = xr.DataArray(evap_WRF_Thom_ACC, dims=('time'), coords = {'time':LH_WRF_Thom_SGP.coords['time'] })
evap_WRF_Thom_ACC.attrs['units'] = "mm"
evap_WRF_Thom_ACC.attrs['long_name'] = "accumulated ET, converted from latent heat flux"

### soil moisture at 5cm depth 
SMOIS_regrid_Thom = ds_WRF_Thom['SMOIS_regrid'][:,0,:,:]   # depth 0 is 5-cm
SMOIS_WRF_Thom_daily = SMOIS_regrid_Thom.resample(time='1D').mean(dim='time')
SMOIS_WRF_Thom_SGP = SMOIS_WRF_Thom_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
SMOIS_WRF_Thom_SGP.attrs['units'] = "m3/m3"

### soil moisture at 0.25 depth
SMOIS25_regrid_Thom = ds_WRF_Thom['SMOIS_regrid'][:,1,:,:]   # depth 0 is 25-cm
SMOIS25_WRF_Thom_daily = SMOIS25_regrid_Thom.resample(time='1D').mean(dim='time')
SMOIS25_WRF_Thom_SGP = SMOIS25_WRF_Thom_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
SMOIS25_WRF_Thom_SGP.attrs['units'] = "m3/m3"

### evaporative fraction LH/(SH+LH)
HFX_regrid_Thom = ds_WRF_Thom['HFX_regrid']
EF_regrid_Thom = LH_regrid_Thom / (HFX_regrid_Thom+LH_regrid_Thom)
EF_regrid_Thom = EF_regrid_Thom.where( (HFX_regrid_Thom+LH_regrid_Thom) > 10.0)  # to avoid unrealistic values when denominator is too small 

EF_WRF_Thom_daily = EF_regrid_Thom.resample(time='1D').mean(dim='time')
EF_WRF_Thom_SGP = EF_WRF_Thom_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
EF_WRF_Thom_SGP.attrs['units'] = "unitless"

### Sensible heat flux
HFX_WRF_Thom_daily = HFX_regrid_Thom.resample(time='1D').mean(dim='time')
HFX_WRF_Thom_SGP = HFX_WRF_Thom_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
HFX_WRF_Thom_SGP.attrs['units'] = "unitless"

### T2m
T2_regrid_Thom = ds_WRF_Thom['T2_regrid']
T2_WRF_Thom_daily = T2_regrid_Thom.resample(time='1D').mean(dim='time')
T2_WRF_Thom_SGP = T2_WRF_Thom_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')

####=========== WRF new simulations from Xiaodong
RAIN_tot_regrid_XD = ds_WRF_XD1['RAIN_tot_regrid']
RAIN_WRF_XD_daily = RAIN_tot_regrid_XD.resample(time='1D').mean(dim='time')
RAIN_WRF_XD_SGP = RAIN_WRF_XD_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
RAIN_WRF_XD_SGP = RAIN_WRF_XD_SGP * 24.0 # from mm/hr to mm/day
RAIN_WRF_XD_SGP.attrs['units'] = "mm/day"
print(RAIN_WRF_XD_SGP)

#### accumulative rain
#--------uncomment
RAIN_WRF_XD_ACC = np.asarray([RAIN_WRF_XD_SGP[0:i].values.sum() for i in np.arange(0,124,1)])
#-----------
RAIN_WRF_XD_ACC = xr.DataArray(RAIN_WRF_XD_ACC, dims=('time'), coords = {'time':RAIN_WRF_XD_SGP.coords['time'] })
RAIN_WRF_XD_ACC.attrs['units'] = "mm"
RAIN_WRF_XD_ACC.attrs['long_name'] = "accumulated total precip"

### -------- calculate evaporation from latent heat
LH_regrid_XD = ds_WRF_XD1['LH_regrid']
LH_WRF_XD_daily = LH_regrid_XD.resample(time='1D').mean(dim='time')
LH_WRF_XD_SGP = LH_WRF_XD_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
LH_WRF_XD_SGP_W_m2 = LH_WRF_XD_SGP
LH_WRF_XD_SGP = LH_WRF_XD_SGP * 3600.0*24.0/Lv_water # from W/m2 to mm/day
LH_WRF_XD_SGP.attrs['units'] = "mm/day"
LH_WRF_XD_SGP.attrs['long_name'] = "ET converted from latent heat flux, mm/day"

#### accumulative evaporation
#------uncomment
evap_WRF_XD_ACC = np.asarray([LH_WRF_XD_SGP[0:i].values.sum() for i in np.arange(0,124,1)])
#-------

evap_WRF_XD_ACC = xr.DataArray(evap_WRF_XD_ACC, dims=('time'), coords = {'time':LH_WRF_XD_SGP.coords['time'] })
evap_WRF_XD_ACC.attrs['units'] = "mm"
evap_WRF_XD_ACC.attrs['long_name'] = "accumulated ET, converted from latent heat flux"
print(evap_WRF_XD_ACC)

### soil moisture at 5cm depth 
SMOIS_regrid_XD = ds_SM_WRF_XD['SMOIS_regrid'][:,0,:,:]   # depth 0 is 5-cm
SMOIS_WRF_XD_daily = SMOIS_regrid_XD.resample(time='1D').mean(dim='time')
SMOIS_WRF_XD_SGP = SMOIS_WRF_XD_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
SMOIS_WRF_XD_SGP.attrs['units'] = "m3/m3"

### soil moisture at 0.25 depth
SMOIS25_regrid_XD = ds_SM_WRF_XD['SMOIS_regrid'][:,1,:,:]   # depth 0 is 25-cm
SMOIS25_WRF_XD_daily = SMOIS25_regrid_XD.resample(time='1D').mean(dim='time')
SMOIS25_WRF_XD_SGP = SMOIS25_WRF_XD_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
SMOIS25_WRF_XD_SGP.attrs['units'] = "m3/m3"

### evaporative fraction LH/(SH+LH)
HFX_regrid_XD = ds_WRF_XD1['HFX_regrid']
EF_regrid_XD = LH_regrid_XD / (HFX_regrid_XD+LH_regrid_XD)
EF_regrid_XD = EF_regrid_XD.where( (HFX_regrid_XD+LH_regrid_XD) > 10.0)  # to avoid unrealistic values when denominator is too small 
EF_WRF_XD_daily = EF_regrid_XD.resample(time='1D').mean(dim='time')
EF_WRF_XD_SGP = EF_WRF_XD_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
EF_WRF_XD_SGP.attrs['units'] = "unitless"

### Sensible heat flux
HFX_WRF_XD_daily = HFX_regrid_XD.resample(time='1D').mean(dim='time')
HFX_WRF_XD_SGP = HFX_WRF_XD_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
HFX_WRF_XD_SGP.attrs['units'] = "unitless"

### T2m
T2_regrid_XD = ds_WRF_XD2['T2_regrid']
T2_WRF_XD_daily = T2_regrid_XD.resample(time='1D').mean(dim='time')
T2_WRF_XD_SGP = T2_WRF_XD_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')

### ---------------------------
### ARM SGP obs: ARMBE2DGRID from Qi Tang

precip05 = ds_ARMBE2D_05['precip_rate']
precip06 = ds_ARMBE2D_06['precip_rate']
precip07 = ds_ARMBE2D_07['precip_rate']
precip08 = ds_ARMBE2D_08['precip_rate']
precip_05678 = xr.concat([precip05, precip06, precip07, precip08], dim='time')
precip_daily = precip_05678.resample(time='1D').mean('time')
precip_ARM_SGP = precip_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
precip_ARM_SGP = precip_ARM_SGP * 24.0 # from mm/hr to mm/day
precip_ARM_SGP.attrs['units'] = "mm/day"
#print(precip_ARM_SGP)

### accumulative rain
precip_ARM_ACC = np.asarray([precip_ARM_SGP[0:i].values.sum() for i in np.arange(0,123,1)])
precip_ARM_ACC = xr.DataArray(precip_ARM_ACC, dims=('time'), coords = {'time':precip_ARM_SGP.coords['time'] })
precip_ARM_ACC.attrs['units'] = "mm"
precip_ARM_ACC.attrs['long_name'] = "accumulated total precip"
#print(precip_ARM_ACC)

### evaporation converted from latent heat flux
latent05 = -ds_ARMBE2D_05['latent_heat_flux'] # upward means positive
latent06 = -ds_ARMBE2D_06['latent_heat_flux']
latent07 = -ds_ARMBE2D_07['latent_heat_flux']
latent08 = -ds_ARMBE2D_08['latent_heat_flux']
latent_05678 = xr.concat([latent05, latent06, latent07, latent08], dim='time')
latent_daily = latent_05678.resample(time='1D').mean('time')
latent_ARM_SGP = latent_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
latent_ARM_SGP_W_m2 = latent_ARM_SGP
latent_ARM_SGP = latent_ARM_SGP * 3600.0*24.0/Lv_water # from W/m2 to mm/day 
latent_ARM_SGP.attrs['units'] = "mm/day"
#print(latent_ARM_SGP)

### accumulative ET
evap_ARM_ACC = np.asarray([latent_ARM_SGP[0:i].values.sum() for i in np.arange(0,123,1)])
evap_ARM_ACC = xr.DataArray(evap_ARM_ACC, dims=('time'), coords = {'time':latent_ARM_SGP.coords['time'] })
evap_ARM_ACC.attrs['units'] = "mm"
evap_ARM_ACC.attrs['long_name'] = "accumulated total ET, converted from latent heat flux"
#print(evap_ARM_ACC)

### soil moisture at 5-cm
SM05 = ds_ARMBE2D_05['soil_moisture_swats'][:,0,:,:] # 0 layer is 5-cm
SM06 = ds_ARMBE2D_06['soil_moisture_swats'][:,0,:,:]
SM07 = ds_ARMBE2D_07['soil_moisture_swats'][:,0,:,:]
SM08 = ds_ARMBE2D_08['soil_moisture_swats'][:,0,:,:]
SM_05678 = xr.concat([SM05, SM06, SM07, SM08], dim='time')
SM_daily = SM_05678.resample(time='1D').mean('time')
SM_ARM_SGP = SM_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
SM_ARM_SGP.attrs['units'] = "m3/m3"
#print(SM_ARM_SGP)

### soil moisture at 25-cm
SM25_05 = ds_ARMBE2D_05['soil_moisture_swats'][:,2,:,:] # 2 layer is 25-cm
SM25_06 = ds_ARMBE2D_06['soil_moisture_swats'][:,2,:,:]
SM25_07 = ds_ARMBE2D_07['soil_moisture_swats'][:,2,:,:]
SM25_08 = ds_ARMBE2D_08['soil_moisture_swats'][:,2,:,:]
SM25_05678 = xr.concat([SM25_05, SM25_06, SM25_07, SM25_08], dim='time')
SM25_daily = SM25_05678.resample(time='1D').mean('time')
SM25_ARM_SGP = SM25_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
SM25_ARM_SGP.attrs['units'] = "m3/m3"
#print(SM_ARM_SGP)

### soil moisture from ebbr measurements (only 2.5cm)
SM05_ebbr = ds_ARMBE2D_05['soil_moisture_ebbr']
SM06_ebbr = ds_ARMBE2D_06['soil_moisture_ebbr']
SM07_ebbr = ds_ARMBE2D_07['soil_moisture_ebbr']
SM08_ebbr = ds_ARMBE2D_08['soil_moisture_ebbr']
SM_05678_ebbr = xr.concat([SM05_ebbr, SM06_ebbr, SM07_ebbr, SM08_ebbr], dim='time')
SM_daily_ebbr = SM_05678_ebbr.resample(time='1D').mean('time')
SM_ARM_SGP_ebbr = SM_daily_ebbr.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
SM_ARM_SGP_ebbr.attrs['units'] = "m3/m3"

### evaporative fraction = LH/(SH+LH)
sensible05 = -ds_ARMBE2D_05['sensible_heat_flux'] # upward means positive
sensible06 = -ds_ARMBE2D_06['sensible_heat_flux']
sensible07 = -ds_ARMBE2D_07['sensible_heat_flux']
sensible08 = -ds_ARMBE2D_08['sensible_heat_flux']
sensible_05678 = xr.concat([sensible05, sensible06, sensible07, sensible08], dim='time')
EF_obs = latent_05678/(latent_05678+sensible_05678)
EF_obs = EF_obs.where( (latent_05678+sensible_05678) > 10.0)  # to avoid unrealistic values when denominator is too small. 

EF_daily = EF_obs.resample(time='1D').mean('time')
EF_ARM_SGP = EF_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
EF_ARM_SGP.attrs['units'] = "unitless"

### Sensible heat flux
sensible_daily = sensible_05678.resample(time='1D').mean('time')
sensible_ARM_SGP = sensible_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
sensible_ARM_SGP.attrs['units'] = "unitless"

### 2m air temperature 
temp05 = ds_ARMBE2D_05['temp'] 
temp06 = ds_ARMBE2D_06['temp']
temp07 = ds_ARMBE2D_07['temp']
temp08 = ds_ARMBE2D_08['temp']
temp_05678 = xr.concat([temp05, temp06, temp07, temp08], dim='time')
temp_daily = temp_05678.resample(time='1D').mean('time')
temp_ARM_SGP = temp_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
#print(temp_ARM_SGP)

### Stage IV precip dataset
pr_st4 = ds_pr_stage4['precip_st4_regrid'][718:,:,:]  # skip Apr values
#print(pr_st4)

pr_st4_daily = pr_st4.resample(time='1D').mean('time')
pr_st4_ARM_SGP = pr_st4_daily.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
pr_st4_ARM_SGP = pr_st4_ARM_SGP * 24.0 # from mm/hr to mm/day
pr_st4_ARM_SGP.attrs['units'] = "mm/day"
#print(pr_st4_ARM_SGP)

### Add GLEAM Evaporation, which can be convert to LHFLX W/m2
E_a = ds_GLEAM['E_a_regrid'][120:243,:,:]   # May-Aug
E_b = ds_GLEAM['E_b_regrid'][120:243,:,:]

E_a = E_a * 2265000.0 / (3600*24)   # from Evaporation mm/day to W/m2
E_a.attrs['units'] = "W/m2"

E_b = E_b * 2265000.0 / (3600*24)   # from Evaporation mm/day to W/m2
E_b.attrs['units'] = "W/m2"

E_a_ARM_SGP = E_a.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
E_b_ARM_SGP = E_b.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')

### ---------------------------
### Plot ###
x_axis = RAIN_WRF_ACC.coords['time']

### x-axis for datetime64
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
days = mdates.DayLocator()
#dates_fmt = mdates.DateFormatter('%Y-%m-%d')
dates_fmt = mdates.DateFormatter('%m-%d')

fig = plt.figure(figsize=(15,15))
fontsize = 5.5
pos_adjust1 = 0.02

ax1 = fig.add_subplot(4,4,1)
ax1.text(s='Accumulated precip, mm', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax1.transAxes)
ax1.plot(x_axis, RAIN_WRF_ACC.values, 'b-', label='precip, WRF'+label_string1)
ax1.plot(x_axis[0:120], RAIN_WRF_Thom_ACC.values, 'g-', label='precip, WRF'+label_string2)
ax1.plot(x_axis, RAIN_WRF_XD_ACC[0:122].values, 'r-', label='precip, WRF'+label_string3)
## Note that WRF simulation does not have 08-31 data, only 122 values;
## Therefore, also omit 08-31 data in ARMBE when plotting.
ax1.plot(x_axis, precip_ARM_ACC[0:122].values, 'k-', label='precip, ARMBE2D')
ax1.grid()
ax1.legend(loc='upper left',fontsize=fontsize)
## format the ticks
ax1.xaxis.set_major_locator(months)
ax1.xaxis.set_major_formatter(dates_fmt)

### subplot (3,3,2)
ax2 = fig.add_subplot(4,4,2)
ax2.text(s='Accumulated ET (converted from LatentHeatFlux), mm', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax2.transAxes)
ax2.plot(x_axis, evap_WRF_ACC.values, 'b-', label='ET, WRF'+label_string1)
ax2.plot(x_axis[0:120], evap_WRF_Thom_ACC.values, 'g-', label='ET, WRF'+label_string2)
ax2.plot(x_axis, evap_WRF_XD_ACC[0:122].values, 'r-', label='ET, WRF'+label_string3)
#-------
ax2.plot(x_axis, evap_ARM_ACC[0:122].values, 'k-', label='ET, ARMBE2D')
ax2.grid()
ax2.legend(loc='upper left',fontsize=fontsize)
# format the ticks
ax2.xaxis.set_major_locator(months)
ax2.xaxis.set_major_formatter(dates_fmt)

### subplot (3,3,3)
ax3 = fig.add_subplot(4,4,3)
ax3.text(s='P-E (Accumulated), mm', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax3.transAxes)
ax3.plot(x_axis, (RAIN_WRF_ACC.values - evap_WRF_ACC.values), 'b-', label='P-E, WRF'+label_string1)
ax3.plot(x_axis[0:120], (RAIN_WRF_Thom_ACC.values - evap_WRF_Thom_ACC.values), 'g-', label='P-E, WRF'+label_string2)
ax3.plot(x_axis, (RAIN_WRF_XD_ACC[0:122].values - evap_WRF_XD_ACC[0:122].values), 'r-', label='P-E, WRF'+label_string3)
#--------
ax3.plot(x_axis, (precip_ARM_ACC[0:122].values - evap_ARM_ACC[0:122].values), 'k-', label='P-E, ARMBE2D')
ax3.grid()
ax3.legend(loc='lower left',fontsize=fontsize)
# format the ticks
ax3.xaxis.set_major_locator(months)
ax3.xaxis.set_major_formatter(dates_fmt)

### subplot(3,3,4)
ax4 = fig.add_subplot(4,4,4)
ax4.text(s='soil moisture, m3/m3', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax4.transAxes)
ax4.plot(x_axis, SMOIS_WRF_SGP.values, 'b-', label='5cm,WRF'+label_string1)
ax4.plot(x_axis, SMOIS25_WRF_SGP.values, 'b+', label='25cm,WRF'+label_string1)
ax4.plot(x_axis[0:120], SMOIS_WRF_Thom_SGP.values, 'g-', label='5cm,WRF'+label_string2)
ax4.plot(x_axis[0:120], SMOIS25_WRF_Thom_SGP.values, 'g+', label='25cm,WRF'+label_string2)
ax4.plot(x_axis, SMOIS_WRF_XD_SGP[0:122].values, 'r-', label='5cm,WRF'+label_string3)
ax4.plot(x_axis, SMOIS25_WRF_XD_SGP[0:122].values, 'r+', label='25cm,WRF'+label_string3)
ax4.plot(x_axis, SM_ARM_SGP[0:122].values, 'k-', label='SM_swats,5cm')
ax4.plot(x_axis, SM25_ARM_SGP[0:122].values, 'k+', label='SM_swats,25cm')
ax4.plot(x_axis, SM_ARM_SGP_ebbr[0:122].values, 'k--', label='SM_ebbr,2.5cm')
ax4.set_ylim(0.05,0.35)
#ax4.set_yticks([0.0,0.1,0.2,0.3,0.4])
ax4.grid()
ax4.legend(loc='lower left',fontsize=fontsize)
# format the ticks
ax4.xaxis.set_major_locator(months)
ax4.xaxis.set_major_formatter(dates_fmt)

### subplot(3,3,5)
ax5 = fig.add_subplot(4,4,5)
ax5.text(s='EF bias, WRF-obs, (EF=LH/(SH+LH)), unitless', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax5.transAxes)
ax5.plot(x_axis, (EF_WRF_SGP.values - EF_ARM_SGP[0:122].values) , 'b-', label='WRF'+label_string1)
ax5.plot(x_axis[0:120], (EF_WRF_Thom_SGP.values - EF_ARM_SGP[0:120].values) , 'g-', label='WRF'+label_string2)
ax5.plot(x_axis, (EF_WRF_XD_SGP[0:122].values - EF_ARM_SGP[0:122].values) , 'r-', label='WRF'+label_string3)
#ax5.plot(x_axis, EF_WRF_SGP.values , 'b-', label='EF WRF')
#ax5.plot(x_axis, EF_ARM_SGP[0:122].values, 'k-', label='EF obs')
ax5.grid()
ax5.legend(loc='lower left',fontsize=fontsize)
# format the ticks
ax5.xaxis.set_major_locator(months)
ax5.xaxis.set_major_formatter(dates_fmt)
ax5.axhline(linewidth=1.5, color='k')

### suplot(3,3,6)
ax6 = fig.add_subplot(4,4,6)
ax6.text(s='T2 bias, WRF-obs, K', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax6.transAxes)
ax6.plot(x_axis, (T2_WRF_SGP.values - temp_ARM_SGP[0:122].values) , 'b-', label='WRF'+label_string1)
ax6.plot(x_axis[0:120], (T2_WRF_Thom_SGP.values - temp_ARM_SGP[0:120].values) , 'g-', label='WRF'+label_string2)
ax6.plot(x_axis, (T2_WRF_XD_SGP[0:122].values - temp_ARM_SGP[0:122].values) , 'r-', label='WRF'+label_string3)
ax6.grid()
ax6.legend(loc='lower left',fontsize=fontsize)
# format the ticks
ax6.xaxis.set_major_locator(months)
ax6.xaxis.set_major_formatter(dates_fmt)
ax6.axhline(linewidth=1.5, color='k')

### Add precipitation rate
ax7 = fig.add_subplot(4,4,7)
ax7.text(s='Precip rate bias, mm/day', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax7.transAxes)
ax7.plot(x_axis, RAIN_WRF_SGP.values - precip_ARM_SGP[0:122].values, 'b-', label='WRF'+label_string1+'-ARMBE2D')
ax7.plot(x_axis[0:120], RAIN_WRF_Thom_SGP.values - precip_ARM_SGP[0:120].values, 'g-', label='WRF'+label_string2+'-ARMBE2D')
ax7.plot(x_axis, RAIN_WRF_XD_SGP[0:122].values - precip_ARM_SGP[0:122].values, 'r-', label='WRF'+label_string3+'-ARMBE2D')
ax7.grid()
ax7.legend(loc='upper left',fontsize=fontsize)
## format the ticks
ax7.xaxis.set_major_locator(months)
ax7.xaxis.set_major_formatter(dates_fmt)
ax7.axhline(linewidth=1.5, color='k')

### Add latent heat flux
ax8 = fig.add_subplot(4,4,8)
ax8.text(s='Latent heat flux bias, W/m2', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax8.transAxes)
ax8.plot(x_axis, LH_WRF_SGP_W_m2.values - latent_ARM_SGP_W_m2[0:122].values , 'b-', label='WRF'+label_string1+'-ARMBE2D')
ax8.plot(x_axis[0:120], LH_WRF_Thom_SGP_W_m2.values - latent_ARM_SGP_W_m2[0:120].values , 'g-', label='WRF'+label_string2+'-ARMBE2D')
ax8.plot(x_axis, LH_WRF_XD_SGP_W_m2[0:122].values - latent_ARM_SGP_W_m2[0:122].values , 'r-', label='WRF'+label_string3+'-ARMBE2D')
ax8.grid()
ax8.legend(loc='lower left',fontsize=fontsize)
## format the ticks
ax8.xaxis.set_major_locator(months)
ax8.xaxis.set_major_formatter(dates_fmt)
ax8.axhline(linewidth=1.5, color='k')

### Add sensible heat flux
ax9 = fig.add_subplot(4,4,9)
ax9.text(s='Sensible heat flux bias, W/m2', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax9.transAxes)
ax9.plot(x_axis, HFX_WRF_SGP.values - sensible_ARM_SGP[0:122].values , 'b-', label='WRF'+label_string1+'-ARMBE2D')
ax9.plot(x_axis[0:120], HFX_WRF_Thom_SGP.values - sensible_ARM_SGP[0:120].values , 'g-', label='WRF'+label_string2+'-ARMBE2D')
ax9.plot(x_axis, HFX_WRF_XD_SGP[0:122].values - sensible_ARM_SGP[0:122].values , 'r-', label='WRF'+label_string3+'-ARMBE2D')
ax9.grid()
ax9.legend(loc='upper right',fontsize=fontsize)
## format the ticks
ax9.xaxis.set_major_locator(months)
ax9.xaxis.set_major_formatter(dates_fmt)
ax9.axhline(linewidth=1.5, color='k')

### Add WRF precip - Stage IV precip
ax10 = fig.add_subplot(4,4,10)
ax10.text(s='Precip rate bias, mm/day', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax10.transAxes)
ax10.plot(x_axis, RAIN_WRF_SGP.values - pr_st4_ARM_SGP[0:122].values, 'b-', label='WRF'+label_string1+'-StageIV_pr')
ax10.plot(x_axis[0:120], RAIN_WRF_Thom_SGP.values - pr_st4_ARM_SGP[0:120].values, 'g-', label='WRF'+label_string2+'-StageIV_pr')
ax10.plot(x_axis, RAIN_WRF_XD_SGP[0:122].values - pr_st4_ARM_SGP[0:122].values, 'r-', label='WRF'+label_string3+'-StageIV_pr')
ax10.grid()
ax10.legend(loc='upper left',fontsize=fontsize)
## format the ticks
ax10.xaxis.set_major_locator(months)
ax10.xaxis.set_major_formatter(dates_fmt)
ax10.axhline(linewidth=1.5, color='k')

### Add latent heat flux
ax11 = fig.add_subplot(4,4,11)
ax11.text(s='Latent heat flux bias, W/m2', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax11.transAxes)
ax11.plot(x_axis, LH_WRF_SGP_W_m2.values - E_a_ARM_SGP[0:122].values , 'b-', label='WRF'+label_string1+'-GLEAM_E_va')
ax11.plot(x_axis[0:120], LH_WRF_Thom_SGP_W_m2.values - E_a_ARM_SGP[0:120].values , 'g-', label='WRF'+label_string2+'-GLEAM_E_va')
ax11.plot(x_axis, LH_WRF_XD_SGP_W_m2[0:122].values - E_a_ARM_SGP[0:122].values , 'r-', label='WRF'+label_string3+'-GLEAM_E_va')
ax11.grid()
ax11.legend(loc='upper right',fontsize=fontsize)
## format the ticks
ax11.xaxis.set_major_locator(months)
ax11.xaxis.set_major_formatter(dates_fmt)
ax11.axhline(linewidth=1.5, color='k')

### Add latent heat flux
ax12 = fig.add_subplot(4,4,12)
ax12.text(s='Latent heat flux bias, W/m2', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax12.transAxes)
ax12.plot(x_axis, LH_WRF_SGP_W_m2.values - E_b_ARM_SGP[0:122].values , 'b-', label='WRF'+label_string1+'-GLEAM_E_vb')
ax12.plot(x_axis[0:120], LH_WRF_Thom_SGP_W_m2.values - E_b_ARM_SGP[0:120].values , 'g-', label='WRF'+label_string2+'-GLEAM_E_vb')
ax12.plot(x_axis, LH_WRF_XD_SGP_W_m2[0:122].values - E_b_ARM_SGP[0:122].values , 'r-', label='WRF'+label_string3+'-GLEAM_E_vb')
ax12.grid()
ax12.legend(loc='lower left',fontsize=fontsize)
## format the ticks
ax12.xaxis.set_major_locator(months)
ax12.xaxis.set_major_formatter(dates_fmt)
ax12.axhline(linewidth=1.5, color='k')

###
fig.savefig("../Figure/10_WRF_3_sets_vs_ARM_SGP_evolution.png",dpi=600)
plt.show()

#### EF and radiation pathways attribution 
### test on partial_Sensible_heat/partial_T2 in 3 WRF simulations during JJA

## ----- WRF_Morrison
y_wrf_m = HFX_WRF_SGP[31:]
x_wrf_m = T2_WRF_SGP[31:]
WRF_M_cor = scipy.stats.pearsonr(x_wrf_m, y_wrf_m)
print('WRF_Morrison, correlation, pearson r:',WRF_M_cor)

#-- linear regression
x2_wrf_m = T2_WRF_SGP[31:].values.reshape(-1,1)
model = LinearRegression().fit(x2_wrf_m, y_wrf_m)
WRF_M_lr_i = model.intercept_
WRF_M_lr_k = model.coef_
print('WRF_Morrison, partial_HFX / partial_T2, linear regression, slope:',WRF_M_lr_k)

## ----- WRF_Thompson
y_wrf_t = HFX_WRF_Thom_SGP[31:]
x_wrf_t = T2_WRF_Thom_SGP[31:]
WRF_T_cor = scipy.stats.pearsonr(x_wrf_t, y_wrf_t)
print('WRF_Thomspon, correlation, pearson r:',WRF_T_cor)

#print(x_wrf_t)
#print(y_wrf_t)

#-- linear regression
x2_wrf_t = T2_WRF_Thom_SGP[31:].values.reshape(-1,1)
model = LinearRegression().fit(x2_wrf_t, y_wrf_t)
WRF_T_lr_i = model.intercept_
WRF_T_lr_k = model.coef_
print('WRF_Thompson, partial_HFX / partial_T2, linear regression, slope:',WRF_T_lr_k)

## ----- WRF_Xiaodong
y_wrf_XD = HFX_WRF_XD_SGP[31:]
x_wrf_XD = T2_WRF_XD_SGP[31:]
WRF_XD_cor = scipy.stats.pearsonr(x_wrf_XD, y_wrf_XD)
print('WRF_Xiaodong, correlation, pearson r:',WRF_XD_cor)

#print(x_wrf_XD)
#print(y_wrf_XD)

#-- linear regression
x2_wrf_XD = T2_WRF_XD_SGP[31:].values.reshape(-1,1)
model = LinearRegression().fit(x2_wrf_XD, y_wrf_XD)
WRF_XD_lr_i = model.intercept_
WRF_XD_lr_k = model.coef_
print('WRF_Xiaodong, partial_HFX / partial_T2, linear regression, slope:',WRF_XD_lr_k)

#####===============
####============ contribution from EF terms
Gamma_obs = 2.2

denominator_WRF_M = 15.4
denominator_WRF_T = 15.7
denominator_WRF_XD = 6.4

EF_JJA_bias_WRF_M = np.mean(EF_WRF_SGP.values[31:] - EF_ARM_SGP[31:122].values)
sum_LH_SH_JJA_WRF_M = np.mean(HFX_WRF_SGP.values[31:]) + np.mean(LH_WRF_SGP_W_m2.values[31:])
print('EF bias in WRF_Morrison, 2011 JJA:')
print(EF_JJA_bias_WRF_M)
print('(SH_mod + LH_mod) in WRF_Morrison, 2011 JJA:')
print(sum_LH_SH_JJA_WRF_M)
print('EF term contribution to T2m bias, WRF_Morrison, 2011JJA:')
print(-1.0 * EF_JJA_bias_WRF_M * Gamma_obs * sum_LH_SH_JJA_WRF_M / denominator_WRF_M)
print('-------')


EF_JJA_bias_WRF_T = np.mean(EF_WRF_Thom_SGP.values[31:] - EF_ARM_SGP[31:120].values)
sum_LH_SH_JJA_WRF_T = np.mean(HFX_WRF_Thom_SGP.values[31:]) + np.mean(LH_WRF_Thom_SGP_W_m2.values[31:])
print('EF bias in WRF_Thompson, 2011 JJA:')
print(EF_JJA_bias_WRF_T)
print('(SH_mod + LH_mod) in WRF_Thomson, 2011 JJA:')
print(sum_LH_SH_JJA_WRF_T)
print('EF term contribution to T2m bias, WRF_Thomspon, 2011JJA:')
print(-1.0 * EF_JJA_bias_WRF_T * Gamma_obs * sum_LH_SH_JJA_WRF_T / denominator_WRF_T)
print('-------')

EF_JJA_bias_WRF_XD = np.mean(EF_WRF_XD_SGP.values[31:-1] - EF_ARM_SGP[31:].values)
sum_LH_SH_JJA_WRF_XD = np.mean(HFX_WRF_XD_SGP.values[31:-1]) + np.mean(LH_WRF_XD_SGP_W_m2.values[31:-1])
print('EF bias in WRF_Xiaodong, 2011 JJA:')
print(EF_JJA_bias_WRF_XD)
print('(SH_mod + LH_mod) in WRF_Xiaodong, 2011 JJA:')
print(sum_LH_SH_JJA_WRF_XD)
print('EF term contribution to T2m bias, WRF_Xiaodong, 2011JJA:')
print(-1.0 * EF_JJA_bias_WRF_XD * Gamma_obs * sum_LH_SH_JJA_WRF_XD / denominator_WRF_XD)
print('-------')










