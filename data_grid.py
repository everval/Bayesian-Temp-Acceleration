import os 
os.chdir(r"C:\Users\Bruger\OneDrive\Nicolai_pedersen\Universitet\Kandidat\Semester_8\Projekt")
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import zoom
from matplotlib.ticker import FuncFormatter,MaxNLocator 
from scipy.signal import convolve2d
import pandas as pd
projection = ccrs.Robinson()
def custom_latitude_formatter(lat,pos):
    if lat > 0:
        return f"{int(lat)}N"
    elif lat <= 0:
        return f"{int(lat)}N" 
def custom_longitude_formatter(lon,pos):
    if lon > 0:
        return f"{int(lon)}E"
    elif lon <= 0:
        return f"{int(lon)}E"
def gibbs(N,T,beta,alpha=np.ones(4)*3,mu=np.array([0.052689,0.01346537,0.00022735]),burnin=0.1):
    init = np.concatenate((mu,beta/(alpha-1)))
    res = np.zeros((N,len(init)))
    res[0] = np.array(init)
    x = np.arange(len(T))
    X = np.transpose(np.array([x**0,x**1,x**2]))
    n = len(x)
    for i in range(1,N):
        cov_post = np.linalg.inv(1/res[i-1,3]*np.transpose(X)@X+np.diag(res[i-1,4:7]))
        mu_post = cov_post@(1/res[i-1,3]*np.transpose(X)@T+np.diag(res[i-1,4:7])@np.array(mu))
        res[i,0:3] = np.random.multivariate_normal(mu_post, cov_post)
        res[i,3] = 1/np.random.gamma(alpha[0]+n/2,1/(1/2*np.sum((T-(res[i,0]*np.transpose(X)[0]+res[i,1]*np.transpose(X)[1]+res[i,2]*np.transpose(X)[2]))**2)+beta[0]))
        for j in range(3):
            res[i,j+4] = 1/np.random.gamma(alpha[j+1]+1/2,1/(1/2*(res[i, j]-mu[j])**2+beta[j+1]))
    res = res[np.int64(N*burnin):]
    res = np.transpose(res)
    return res

#all datasets end in present month except berkeley which does not contain 2025 or onwards
data_origin = "noaa"  #can be hadcrut, noaa or berkeley
#Load NetCDF file
dataset = nc.Dataset(f"data_grid_{data_origin}.nc", mode='r')
#Extract data
if data_origin == "hadcrut": #data starts in 1850 and ends in present time
    first_meas = 1850
    lat = dataset.variables['latitude'][:].data
    lon = dataset.variables['longitude'][:].data
    #time = dataset.variables['time'][:].data
    data = dataset.variables['tas_mean'][:].data
    masks = dataset.variables['tas_mean'][:].mask
if data_origin == "noaa": #data starts in 1850 and ends in present time
    first_meas = 1850
    lat = dataset.variables['lat'][:].data
    lon = dataset.variables['lon'][:].data-180 #to synchronize values
    #time = dataset.variables['time'][:].data
    data = dataset.variables['anom'][:].data.squeeze(1)  #remove a dummy-dimension
if data_origin == "berkeley": #data starts in 1850 and ends in 2024
    first_meas = 1850
    lat = dataset.variables['latitude'][:].data.reshape(180//5,5).mean(1)
    lon = dataset.variables['longitude'][:].data.reshape(360//5,5).mean(1)
    #time = dataset.variables['time'][:].data
    data = dataset.variables["temperature"][:].data
    data = np.nanmean(data.reshape(data.shape[0],36,5,72,5),axis=(2,4))
if data_origin == "era5": #data starts in 1940 and ends in present time
    first_meas = 1940
    lat = dataset.variables['latitude'][:].data
    lat = np.flipud(np.mean([lat[i*21-i:(i+1)*21-i] for i in range(36)],axis=1)) #compute running avg. to downscale resolution
    lon = dataset.variables['longitude'][:].data
    lon = np.mean([lon[1:][i*19+i:(i+1)*19+i] for i in range(72)],axis=1)-180 #compute running avg. to downscale resolution
    #time = dataset.variables['valid_time'][:].data
    data = dataset.variables['t2m'][:].data-273.15 #data is in kelvin
    placeholder = np.zeros((data.shape[0],36,72))
    for i,d in enumerate(data):
        d = np.flipud(d)
        for j in range(36):
            for k in range(72):
                placeholder[i,j,k] = np.mean(d[:,:-1][j*21-j:(j+1)*21-j,k*19+k:(k+1)*19+k])
    data = placeholder
dataset.close()

#temp_anomaly is difference in temp. compared to mean  of the january's of 1961-1990
#first datapoint is jan. 1850 - last is feb. 2026
coordinates = np.transpose(np.meshgrid(lat, lon)).reshape(-1, 2)  # longitude, latitude
coordinates = [' '.join(map(str, sublist)) for sublist in coordinates]
dataset = 0 #clear memory

#change baseline to pre-industrial mean
curr_mean = np.zeros_like(data[0])
pre_mean = np.zeros_like(curr_mean)
num = np.zeros_like(curr_mean)
for i in range(12*(1961-first_meas), 12*(1991-first_meas)):  # 111 141
    if data_origin in ["noaa","berkeley","era5"]:
        mask = np.isnan(data[i])
    elif data_origin == "hadcrut":
        mask = masks[i]
    curr_mean += np.where(mask,0,data[i])
    num += np.invert(mask)
curr_mean *= 1/num
if data_origin != "era5":
    for i in range(0, 12*50):
        if data_origin in ["noaa","berkeley"]:
            mask = np.isnan(data[i])
        elif data_origin == "hadcrut":
            mask = masks[i]
        pre_mean += np.where(mask,0,data[i])
        num += np.invert(mask)
    pre_mean *= 1/num
else:
    pre_mean = 0
for i in range(len(data)):
    data[i] -= (pre_mean+curr_mean)

#%% plot average for each point in grid for each year in years
years = [1980,2000,2010,2025]
avg = []
for year in years:
    a = np.zeros_like(data[0])
    num = np.zeros_like(data[0])
    for j in range((year-first_meas)*12, (year+1-first_meas)*12):
        if data_origin in ["noaa","berkeley","era5"]:
            mask = np.isnan(data[j])
        elif data_origin == "hadcrut":
            mask = masks[j]
        a += np.where(mask,0,data[j])
        num += np.invert(mask)
    a *= 1/num
    avg.append(a)
vmin,vmax = np.floor(np.min(avg)),np.ceil(np.max(avg))
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(10,5),subplot_kw={'projection': projection})
row = 0
col = 0
for i in range(len(avg)):
    if i == 2:
        row = 1
        col = 0
    avg[i] = zoom(avg[i],(2,2),order=1)
    img = ax[row,col].imshow(np.flipud(avg[i]),transform=ccrs.PlateCarree(),cmap="jet",vmin=vmin,vmax=vmax)
    ax[row,col].set_title(f"Average Temperature {years[i]:}")
    ax[row,col].coastlines(color='black',linewidth=0.5)
    ax[row,col].add_feature(cfeature.BORDERS,edgecolor='black',linewidth=0.5)
    gl = ax[row,col].gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color="gray")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = FuncFormatter(custom_longitude_formatter)
    gl.yformatter = FuncFormatter(custom_latitude_formatter)
    gl.ylocator = MaxNLocator(nbins=9)
    col += 1
cbar = plt.colorbar(img,ax=ax,shrink=1,aspect=40,orientation='vertical',pad=0.1)
cbar.set_label('Temperature (°C)')
plt.suptitle(data_origin)
#plt.savefig(f"grid_{years[i]:}.pdf",bbox_inches="tight")

#%% extract data from start_year onwards at each point in grid
start_year = 1970
end_years = [1990,1995,2000,2005,2010,2015,2020,2026]  #ends with december data from previous year for every year in end_years
for end_year in end_years:
    coordinates = np.transpose(np.meshgrid(lat, lon)).reshape(-1, 2)  # longitude, latitude
    coordinates = [' '.join(map(str, sublist)) for sublist in coordinates]
    data_dict = dict.fromkeys(coordinates, None)
    for k in range(len(lat)):
        for l in range(len(lon)):
            data_grid = []
            coord = [lat[k], lon[l]]
            coord = f"{coord[0]} {coord[1]}"
            for j in range((start_year-first_meas)*12,len(data)):
                if data_origin == "hadcrut" and masks[j, k, l] == False:
                    data_grid.append(data[j,k,l])
                elif ~np.isnan(data[j,k,l]):
                    data_grid.append(data[j,k,l])
                else:
                    data_grid.append(np.nan)
            data_dict[coord] = data_grid[:(end_year-start_year)*12]
            
    # turn data yearly instead of monthly
    for coord in coordinates:
        new_dat = np.array(data_dict[coord]).reshape(np.int64(len(data_dict[coord])/12),12)
        for k in range(len(new_dat)): # linear interpolate temp. if at most 3 months missing out of a year
            if np.sum(np.isnan(new_dat[k])) < 4:
                x = np.arange(12)
                ind = ~np.isnan(new_dat[k])
                new_dat[k] = np.interp(x,x[ind],new_dat[k][ind])
        data_dict[coord] = list(np.mean(new_dat,axis=1).astype(np.float64))
    with open(fr"results\{data_origin}\data_{data_origin}_{start_year}_{end_year}.txt", "w") as file:
        json.dump(data_dict,file)

    def bootstrapping(data, N):
        x = data[0]
        y = data[1]
        p = data[2]
        res = y-p
        new_data = np.zeros((N, len(x)))
        for i in range(N):
            new_res = np.random.choice(res, len(res))
            new_data[i] = y+new_res
        return new_data

    def poly(x,coeff):
        return np.vander(x,N=len(coeff),increasing=True)@coeff

    #find beta for each point in the grid
    beta = dict.fromkeys(coordinates, None)
    mu = np.zeros(3)
    k = 0
    for coord in coordinates:
        d = data_dict[coord]
        if np.sum(np.isnan(d)) == 0:
            sig_hat = np.zeros(4)
            coeff = np.flip(np.polyfit(np.arange(len(d)),d,2))
            mu += coeff; k += 1
            sig_hat[3] = 1/len(d)*np.sum((d-poly(np.arange(len(d)),coeff))**2)
            new_d = bootstrapping([np.arange(len(d)),d,poly(np.arange(len(d)),coeff)],1000)
            coeffs = np.zeros((1000,3))
            for i in range(1000):
                coeffs[i] = np.flip(np.polyfit(np.arange(len(d)),new_d[i],2))
            sig_hat[0:3] = 1/len(coeffs)*np.sum((coeffs-np.mean(coeffs,axis=0))**2,axis=0)
            b = 2*sig_hat
            beta[coord] = list(b)
    mu /= k
    beta["mu"] = list(mu)
    with open(fr"results\{data_origin}\beta_{data_origin}_{start_year}_{end_year}.txt", "w") as file:
        json.dump(beta, file)

#%% run gibbs to sample from posterior distribution
for end_year in end_years:
    with open(fr"results\{data_origin}\data_{data_origin}_{start_year}_{end_year}.txt", "r") as file:
        data_dict = json.load(file)
    with open(fr"results\{data_origin}\beta_{data_origin}_{start_year}_{end_year}.txt", "r") as file:
        beta = json.load(file)
    post_means = dict.fromkeys(coordinates,None)
    cpi = dict.fromkeys(coordinates,None)
    prob_g_0 = dict.fromkeys(coordinates,None)
    for k in range(len(lat)):
        for l in range(len(lon)):
            coord = [lat[k],lon[l]]
            coord = f"{coord[0]} {coord[1]}"
            d = data_dict[coord]
            if np.sum(np.isnan(d)) == 0:
                b = beta[coord]
                sim = gibbs(100000,d,b,mu=beta["mu"])
                post_means[coord] = list(np.mean(sim,axis=1))
                prob_g_0[coord] = list(np.sum(sim>=0,axis=1)/sim.shape[1])
    with open(fr"results\{data_origin}\post_means_{data_origin}_{start_year}_{end_year}.txt", "w") as file:
        json.dump(post_means,file)
    with open(fr"results\{data_origin}\prob_g_0_{data_origin}_{start_year}_{end_year}.txt", "w") as file:
        json.dump(prob_g_0,file)
        
#%% traceplots for points in grid
data_origin = "era5"
with open(fr"results\{data_origin}\data_{data_origin}_1970_1990.txt","r") as file:
    data = json.load(file)
with open(fr"results\{data_origin}\beta_{data_origin}_1970_1990.txt","r") as file:
    beta = json.load(file)

ks = ["57.5","-2.5","-17.5","-62.5"]
ls = ["87.5","32.5","-152.5","62.5"]  
steps = 100

j = 0
for k,l in zip(ks,ls):  #middle of russia (approx. Tomsk), approx. Rwanda, middle of pacific ocean, southern ocean
    coord = f"{k} {l}"
    d = data[coord]
    if np.sum(np.isnan(d)) == 0:
        b = beta[coord]
        sim = gibbs(1000,d,beta=np.array(b),burnin=0)
        plt.figure()
        fig, axes = plt.subplots(nrows=3,ncols=1,figsize=(6,8),sharex=True)
        for i in range(3):
            axes[i].plot(sim[i][:steps],"o",label=f"Trace of $a_{i:}$")
            axes[i].plot(sim[i][:steps],"b--")
            axes[i].legend(loc="upper right")
        axes[0].set_title(f"Traceplot at: lat = {k}, lon = {l}")
        plt.savefig(f"{j}_a.pdf", bbox_inches="tight")
        plt.figure()
        fig, axes = plt.subplots(nrows=4,ncols=1,figsize=(6,8),sharex=True)
        axes[0].plot(sim[3][:steps],"o",label="Trace of $\sigma^2$")
        axes[0].plot(sim[3][:steps],"b--")
        axes[0].legend(loc="upper right")
        for i in range(4,7):
            axes[i-3].plot(sim[i][:steps],"o",label=f"Trace of $\sigma^2_{i-4:}$")
            axes[i-3].plot(sim[i][:steps],"b--")
            axes[i-3].legend(loc="upper right")
        axes[0].set_title(f"Traceplot at: lat = {k}, lon = {l}")
        plt.savefig(f"{j}_s.pdf",bbox_inches="tight")
        j += 1
#%% plot of fitted polynomial at points in grid
start_year = 1970
end_year = 2026

data_origin = "era5"
with open(fr"results\{data_origin}\data_{data_origin}_1970_1990.txt","r") as file:
    data = json.load(file)
with open(f"results\{data_origin}\post_means_{data_origin}_{start_year}_{end_year}.txt","r") as file:
    post_means = json.load(file)

ks = ["57.5","-2.5","-17.5","-62.5"]
ls = ["87.5","32.5","-152.5","62.5"]  

acc = []
j = 0
for j, (k, l) in enumerate(zip(ks, ls)):
    coord = f"{k} {l}"
    d = np.array(data[coord])
    coeff = post_means[coord][:3]
    acc.append(coeff[2])
    x = np.arange(len(d))
    poly = coeff[0] + coeff[1]*x + coeff[2]*x**2
    plt.figure()
    plt.plot(d, "bo", label="Measurements")
    plt.plot(d, "b--")
    plt.plot(poly, "r", label="Fitted polynomial")
    plt.title(f"lat = {k}, lon = {l}")
    ticks = np.arange(0,len(x),10)
    labs = ticks+start_year
    plt.xticks(ticks,labels=labs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data_and_poly_{coord}.pdf")
#plt.show()