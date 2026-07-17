import numpy as np
import os 
os.chdir(r"C:\Users\Bruger\OneDrive\Nicolai_pedersen\Universitet\Kandidat\Semester_8\Projekt")
import matplotlib.pyplot as plt
from scipy.special import gamma, kv
from scipy.optimize import minimize
import json
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import zoom
from matplotlib.ticker import FuncFormatter,MaxNLocator
projection = ccrs.Robinson()
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from global_land_mask import globe

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
def grf(a):
    def pos(n, m):
        i,j = np.indices((n,m))
        return np.stack((i,j),axis=-1)
    x_known = a.flatten()
    s_known = pos(a.shape[0],a.shape[1]).reshape(-1, 2)
    keep = ~np.isnan(x_known)
    x_known = x_known[keep];s_known = s_known[keep]
    s_unknown = pos(a.shape[0],a.shape[1]).reshape(-1, 2)
    s_unknown = s_unknown[~keep]
    
    def gen_cov_mat(s,sig,nu,l):
        n = s.shape[0]
        cov_matrix = np.zeros((n,n))
        h = s[:, np.newaxis, :] - s[np.newaxis, :, :]
        h = np.linalg.norm(h,axis=-1)
        np.fill_diagonal(h,10)
        cov_matrix = sig**2*(2**(1-nu)/gamma(nu))*((h/l)**nu)*kv(nu,h/l)
        np.fill_diagonal(cov_matrix,sig**2)
        return cov_matrix
    
    def mean(s,mu):
        n = s.shape[0]
        return mu*np.ones(n)
    
    def log_like_simp(params,sig,x,s):
        nu,l = params
        cov_mat = gen_cov_mat(s,sig,nu,l)
        L = np.linalg.cholesky(cov_mat)
        n = x.shape[0]
        L_inv = np.linalg.solve(L,np.eye(n))
        cov_inv = L_inv.T@L_inv
        log_det_cov = 2*np.sum(np.log(np.diag(L)))
        mu = np.sum(cov_inv@x)/np.sum(cov_inv)*np.ones(n)
        return n*np.log((x-mu).T@cov_inv@(x-mu))+log_det_cov
    
    def callback(xk):
        print("Evaluating at:", xk)
    nu,l = minimize(log_like_simp,[1,1],method='SLSQP',args=(1,x_known,s_known),bounds=[(1e-6,2),(1e-6,50)],callback=callback).x
    mu = np.sum(gen_cov_mat(s_known,1,nu,l)@x_known)/np.sum(gen_cov_mat(s_known,1,nu,l))
    sig = 1/len(x_known)*x_known.T@np.linalg.solve(gen_cov_mat(s_known,1,nu,l),np.eye(s_known.shape[0]))@x_known
    
    def pred(x_known,s_known,s_unknown,mu,sig,nu,l):
        s = np.concatenate((s_unknown,s_known))
        n = s_known.shape[0]
        d = s_unknown.shape[0]
        full_cov = gen_cov_mat(s,sig,nu,l)
        cov_un = full_cov[:d,:d]
        cov_kn = full_cov[d:,d:]
        cov_un_kn = full_cov[:d,d:]
        cov_kn_inv = np.linalg.solve(cov_kn,np.eye(n))
        E_un = mean(s_unknown,mu)+cov_un_kn@cov_kn_inv@(x_known-mean(s_known,mu))
        cond_cov = cov_un - cov_un_kn@cov_kn_inv@cov_un_kn.T
        return E_un,np.diag(cond_cov)
    
    x_unknown,cov_unknown = pred(x_known,s_known,s_unknown,mu,sig,nu,l)
    x = np.zeros(len(x_unknown)+len(x_known));x[keep] = x_known;x[~keep] = x_unknown
    return x.reshape(a.shape)

#choose data origin: hadcrut, era5 or noaa
#choose dataset: evaluate probability the acceleration coefficient is greater than 0 or posterior means of temperature
data_origin = "era5"  #can be hadcrut, noaa or era5
#Load NetCDF file
path = f"results\\{data_origin}\\"
fname = nc.Dataset(f"data_grid_{data_origin}.nc", mode='r')
#Extract data
if data_origin == "hadcrut": #data starts in 1850 and ends in present time
    lat = fname.variables['latitude'][:].data
    lon = fname.variables['longitude'][:].data
if data_origin == "noaa": #data starts in 1850 and ends in present time
    lat = fname.variables['lat'][:].data
    lon = fname.variables['lon'][:].data-180 #to synchronize values
if data_origin == "berkeley": #data starts in 1850 and ends in 2024
    lat = fname.variables['latitude'][:].data.reshape(180//5,5).mean(1)
    lon = fname.variables['longitude'][:].data.reshape(360//5,5).mean(1)
if data_origin == "era5": #data starts in 1940 and ends in present time
    lat = fname.variables['latitude'][:].data
    lat = np.flipud(np.mean([lat[i*21-i:(i+1)*21-i] for i in range(36)],axis=1)) #compute running avg. to downscale resolution
    lon = fname.variables['longitude'][:].data
    lon = np.mean([lon[1:][i*19+i:(i+1)*19+i] for i in range(72)],axis=1)-180 #compute running avg. to downscale resolution
fname.close()
dataset = "prob_g_0"   #can be post_means or prob_g_0
if dataset == "prob_g_0":
    title1 = "$P($"
    title2 = "$\geq0)$"
elif dataset == "post_means":
    title1 = "Acceleration coefficient "
    title2 = ""
start_year = 1970
end_years = [1990,1995,2000,2005,2010,2015,2020,2026]
#%% remove data not present in all time-intervals
nans = []
places = []
coeff = np.zeros((len(end_years),len(lat),len(lon)))
for i,end_year in enumerate(end_years):
    place = []
    with open(path+dataset+f"_{data_origin}_{start_year}_{end_year}.txt", "r") as file:
        posts = json.load(file)
    for k in range(len(lat)):
        for l in range(len(lon)):
            coord = [lat[k],lon[l]]
            coord = f"{coord[0]} {coord[1]}"
            try:
                coeff[i,k,l] = posts[coord][2]
            except:
                coeff[i,k,l] = np.nan
                place.append([lat[k],lon[l]])
    places.append(place)
    nans.append(np.isnan(coeff).sum())
base = set(tuple(p) for p in places[0])
places = [[] if i == 0 else [p for p in sub if tuple(p) not in base] for i, sub in enumerate(places)]
for i,end_year in enumerate(end_years):
    for k in range(len(lat)):
        for l in range(len(lon)):
            coord = [lat[k],lon[l]]
            if coord in places[-1]:
                coeff[i,k,l] = np.nan
#% plot before interpolation
for i in range(len(coeff)):
    coeff_zoom = zoom(coeff[i],(4,4),order=1)
    fig,ax = plt.subplots(figsize=(10,5),subplot_kw={'projection':projection})
    if dataset=="prob_g_0":
        vmax = 1
    else:
        vmax = np.max(coeff)
    img = ax.imshow(np.flipud(coeff_zoom),cmap="viridis",transform=ccrs.PlateCarree(),vmax=vmax)#,vmin=vmin,vmax=vmax)
    ax.coastlines(color='black',linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color="gray")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = FuncFormatter(custom_longitude_formatter)
    gl.yformatter = FuncFormatter(custom_latitude_formatter)
    gl.ylocator = MaxNLocator(nbins=9)
    cbar = plt.colorbar(img,shrink=0.8)
    if dataset == "post_means":
        ax.set_title(f"Leading coefficient: {start_year}-{end_years[i]}")
    elif dataset == "prob_g_0":
        ax.set_title(f"Probability leading coefficient greater than 0: {start_year}-{end_years[i]}")
#% interpolate and save data
saves = []
for i in range(len(coeff)):
    x = grf(coeff[i])
    saves.append(list(x))
with open(path+"saves_{dataset}_{data_origin}_{start_year}.txt","w") as file:
    json.dump(saves,file)
#%% plot after interpolation
with open(f"saves_{dataset}_{data_origin}_{start_year}.txt","r") as file:
    coeff = json.load(file)
for i in range(len(coeff)):
    x_zoom = zoom(coeff[i],(4,4),order=1)
    fig,ax = plt.subplots(figsize=(10,5),subplot_kw={'projection':projection})
    if dataset=="prob_g_0":
        vmax = 1
    else:
        vmax = np.max(coeff)
    img = ax.imshow(np.flipud(x_zoom),cmap="viridis",transform=ccrs.PlateCarree(),vmax=vmax)#,vmin=vmin,vmax=vmax)
    ax.coastlines(color='black',linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linestyle="--",linewidth=0.5,color="gray")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = FuncFormatter(custom_longitude_formatter)
    gl.yformatter = FuncFormatter(custom_latitude_formatter)
    gl.ylocator = MaxNLocator(nbins=9)
    cbar = plt.colorbar(img,shrink=0.8)
    if dataset == "post_means":
        ax.set_title(f"$Magnitude\ \gamma(s)$",fontsize=18)
        #plt.savefig(f"lead_coeff_{start_year}_{end_years[i]}.pdf",bbox_inches="tight")
    elif dataset == "prob_g_0":
        ax.set_title(f"$Probability\ \gamma(s)>0$",fontsize=18)
        #plt.savefig(f"prob_g_0_{start_year}_{end_years[i]}.pdf",bbox_inches="tight")
#%% plot of when probability of positive acceleration first exceeds perc
perc = 0.9
data_origin = "hadcrut"
start_year = 1970

with open(f"results\{data_origin}\saves_prob_g_0_{data_origin}_{start_year}.txt","r") as file:
    coeff = json.load(file)
above_perc = np.array([(np.array(c)>perc)*end_years[i] for i,c in enumerate(coeff)])
above_perc = np.transpose(above_perc,(2, 1, 0))
masked = np.where(above_perc>0,above_perc,np.inf) #replace 0's by infinity
min_vals = masked.min(axis=2)
min_vals[min_vals==np.inf] = 0
vals = np.unique(min_vals)
fig,ax = plt.subplots(figsize=(10,5),subplot_kw={'projection':projection})
boundaries = np.concatenate([[vals[0]-0.5],(vals[:-1]+vals[1:])/2,[vals[-1]+0.5]])
cmap = plt.get_cmap("jet", len(vals))
colors = cmap(np.arange(len(vals)))
colors[0] = [1, 1, 1, 1]  # white for "Not exceeded"
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)
norm = mcolors.BoundaryNorm(boundaries,cmap.N)
labels = ["Not exceeded"]+[f"{start_year}-{year}" for year in end_years]
x_zoom = zoom(min_vals,(4,4),order=1)
img = ax.imshow(np.flipud(x_zoom),cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
ax.coastlines(color='black',linewidth=0.5)
ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=0.5)
gl = ax.gridlines(draw_labels=True, linestyle="--",linewidth=0.5,color="gray")
gl.top_labels = False
gl.right_labels = False
gl.xformatter = FuncFormatter(custom_longitude_formatter)
gl.yformatter = FuncFormatter(custom_latitude_formatter)
gl.ylocator = MaxNLocator(nbins=9)
plt.title(data_origin.upper())
handles = [mpatches.Patch(color=cmap(norm(v)),label=lab) for v,lab in zip(vals,labels)]
fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(0.5,-0.1),ncol=int(np.ceil((len(end_years)+1)/2)),fontsize=12)
plt.savefig(f"results\{data_origin}\exceed_times_{data_origin}_{int(perc*100)}.png",bbox_inches="tight")

#%% number of grids exceeding perc probability of positive acceleration across land and ocean
perc = 0.9
data_origin = "noaa"

for end_year in [1990,2000,2010,2020,2026]:
    start_year = 1970
    with open(f"results\{data_origin}\saves_prob_g_0_{data_origin}_{start_year}.txt","r") as file:
        coeff = json.load(file)
    #end_year = 2010
    print(f"Timespan: 1970-{end_year}")
    end_year = int((end_year-1990)/5)
    data = np.flipud(coeff[end_year])
    lon_grid,lat_grid = np.meshgrid(lon,lat)
    land_mask = np.flipud(globe.is_land(lat_grid,lon_grid))
    print(f"Grids with probability exceeding {perc}: {np.sum(data>perc)}/{data.size}")
    print(f"Ocean grids with probability exceeding {perc}: {np.sum(data[land_mask]>perc)}/{np.sum(land_mask)}")
    print(f"Land grids with probability exceeding {perc}: {np.sum(data[~land_mask]>perc)}/{np.sum(~land_mask)}")
    print("")
    print("")