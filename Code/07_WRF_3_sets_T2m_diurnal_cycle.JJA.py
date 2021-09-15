'''
Function: analysis for WRF 2011 May-Aug outputs, as in Ma CAUSES paper Figure 8.
Date: 20200325
'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas


ds_WRF = xr.open_dataset('/home/qin5/Data/WRF.postprocessing.extract.hourly.nc')
ds_WRF_Thom = xr.open_dataset('/home/qin5/Data/WRF.postprocessing.extract.hourly.Thom.05678.nc')
ds_WRF_XD = xr.open_dataset('/home/qin5/Data/WRF_Xiaodong_new/Xiaodong_WRF.postprocessing.extract.hourly.T2.1x1.bilinear.nc') 

ds_ARMBE2D_01 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110101.000000.nc')
ds_ARMBE2D_02 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110201.000000.nc')
ds_ARMBE2D_03 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110301.000000.nc')
ds_ARMBE2D_04 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110401.000000.nc')

ds_ARMBE2D_05 = xr.open_dataset('/scratch/CAUSES/CAUSES/obs/ARMBE2DGRID/OLD/2011_old/sgparmbe2dgridX1.c1.20110501.000000.nc')
#ds_ARMBE2D_05 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110501.000000.nc')
ds_ARMBE2D_06 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110601.000000.nc')
ds_ARMBE2D_07 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110701.000000.nc')
ds_ARMBE2D_08 = xr.open_dataset('/home/qin5/Data/ARMBE2DGRID/sgparmbe2dgridX1.c1.20110801.000000.nc')

## Figure 8 in Ma et al 2018, average over 35-38N, 99-96W, consistent with ARMBE2D from Qi Tang
lat_1 = 35.0
lat_2 = 38.0
lon_1 = -99.0
lon_2 = -96.0

### WRF calculate diurnal cycle at ARM SGP site 
T2_regrid = ds_WRF['T2_regrid']

T2_May = T2_regrid[:738,:,:]
T2_JJA = T2_regrid[738:,:,:]

T2_WRF_May = T2_May.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
T2_WRF_JJA = T2_JJA.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')

### Lessons learned: the groupby('time.hour') command requires the time has regular values, just check
print(T2_WRF_May.coords['time'][0:24])
print(T2_WRF_JJA.coords['time'][0:24])

### the time has regular values, so you can use the following command
WRF_May = T2_WRF_May.groupby('time.hour').mean()
WRF_JJA = T2_WRF_JJA.groupby('time.hour').mean()

#### WRF_Thom
T2_regrid_Thom = ds_WRF_Thom['T2_regrid']
T2_JJA_Thom = T2_regrid_Thom[738:,:,:]

T2_WRF_JJA_Thom = T2_JJA_Thom.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
print(T2_WRF_JJA_Thom.coords['time'][0:24])

WRF_JJA_Thom = T2_WRF_JJA_Thom.groupby('time.hour').mean()

### WRF new simulation by Xiaodong
T2_regrid_XD = ds_WRF_XD['T2_regrid']
T2_JJA_XD = T2_regrid_XD[744:,:,:]

T2_WRF_JJA_XD = T2_JJA_XD.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
print(T2_WRF_JJA_XD.coords['time'][0:24])

WRF_JJA_XD = T2_WRF_JJA_XD.groupby('time.hour').mean()


### ARM SGP obs: ARMBE2DGRID from Qi Tang
temp05 = ds_ARMBE2D_05['temp']
temp06 = ds_ARMBE2D_06['temp']
temp07 = ds_ARMBE2D_07['temp']
temp08 = ds_ARMBE2D_08['temp']

temp_0678 = xr.concat([temp06, temp07, temp08], dim='time')

temp_May = temp05.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')
temp_JJA = temp_0678.sel(lat=slice(lat_1, lat_2), lon=slice(lon_1, lon_2)).mean(dim='lat').mean(dim='lon')

### the time coords values are irregular, so you cannot use groupby('time.hour').mean() to 
### calculate diurnal cycle, just use for loops
print(temp_JJA['time'][0:24])

ARM_May = np.zeros(24)
ARM_JJA = np.zeros(24)

for i in np.arange(0,31,1):
    ARM_May = ARM_May + temp_May[0+i*24:i*24+24].values

ARM_May = ARM_May / 31.0

for i in np.arange(0,92,1):
    ARM_JJA = ARM_JJA + temp_JJA[0+i*24:i*24+24].values

ARM_JJA = ARM_JJA / 92.0
print(ARM_May)
print(ARM_JJA)

### do not use the groupby('time.hour').mean() command since the time values is not regular.
##ARM_Jan = temp_Jan.groupby('time.hour').mean()
#ARM_Feb = temp_Feb.groupby('time.hour').mean()
#ARM_Mar = temp_Mar.groupby('time.hour').mean()
#ARM_Apr = temp_Apr.groupby('time.hour').mean()

#ARM_May = temp_May.groupby('time.hour').mean()
#ARM_Jun = temp_Jun.groupby('time.hour').mean()
#ARM_Jul = temp_Jul.groupby('time.hour').mean()
#ARM_Aug = temp_Aug.groupby('time.hour').mean()
#ARM_JJA = temp_JJA.groupby('time.hour').mean()

### WRF bias in May, and JJA ###
#bias_May = WRF_May - ARM_May
#bias_JJA = WRF_JJA - ARM_JJA

#print(WRF_May)
#print(bias_JJA)

### Plot ###
x_axis = WRF_May.coords['hour']

fig = plt.figure(figsize=(8,6))
fontsize = 7
pos_adjust1 = 0.04

ax1 = fig.add_subplot(1,1,1)
ax1.text(s='T2m, WRF, ARMBE2D', x=0, y=1.02, ha='left', va='bottom', \
        fontsize=fontsize, transform=ax1.transAxes)
#ax1.plot(x_axis, bias_May.values, 'r-', label='WRF,May')
#ax1.plot(x_axis, bias_JJA.values, 'r--', label='WRF,JJA')
ax1.plot(x_axis, ARM_JJA, 'k-', label='ARM, JJA')
ax1.plot(x_axis, WRF_JJA, 'b--', label='WRF_Morri, JJA')
ax1.plot(x_axis, WRF_JJA_Thom, 'g--', label='WRF_Thom, JJA')
ax1.plot(x_axis, WRF_JJA_XD, 'r--', label='WRF_new, JJA')

ax1.set_yticks(np.arange(294.0,316.0,2.0))
ax1.set_xticks(np.arange(0.0,24.1,3.0))
ax1.set(xlabel='UTC(hr)', ylabel='T2m, K', title='T2m, WRF vs ARM SGP')
ax1.grid()
ax1.legend(loc='lower right')

#ax2 = fig.add_subplot(2,1,2)
#ax2.text(s='T2m, ARM SGP', x=0, y=1.02, ha='left', va='bottom', \
#        fontsize=fontsize, transform=ax2.transAxes)
##ax2.plot(x_axis, ARM_May, 'k-', label='May')
#ax2.plot(x_axis, ARM_JJA, 'k--', label='JJA')
#ax2.set_yticks(np.arange(285.0,311.0,3.0))
#ax2.set_xticks(np.arange(0.0,24.1,3.0))
#ax2.set(xlabel='UTC(hr)', ylabel='T2m ARM SGP, K')
#ax2.grid()
#ax2.legend(loc='lower right')

print('observation T2m @SGP, 2011 JJA mean')
print(np.mean(ARM_JJA))

print('WRF_Morrison T2m @SGP, 2011 JJA mean')
print(np.mean(WRF_JJA.values))

print('WRF_Thompson T2m @SGP, 2011 JJA mean')
print(np.mean(WRF_JJA_Thom.values))

print('WRF_new T2m @SGP, 2011 JJA mean')
print(np.mean(WRF_JJA_XD.values))

print('WRF_Morrison T2m bias, @SGP, 2011 JJA')
print(np.mean(WRF_JJA.values) - np.mean(ARM_JJA))

print('WRF_Thompson T2m bias, @SGP, 2011 JJA')
print(np.mean(WRF_JJA_Thom.values) - np.mean(ARM_JJA))

print('WRF_new T2m bias, @SGP, 2011 JJA')
print(np.mean(WRF_JJA_XD.values) - np.mean(ARM_JJA))


fig.savefig("../Figure/T2m.WRF_vs_ARM_SGP.png",dpi=600)
plt.show()



