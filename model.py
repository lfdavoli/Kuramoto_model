# %% [markdown]
# # Kuramoto model

# %%
# Libs
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# %% [markdown]
# ### 1. 
# 
# scrivi

# %% [markdown]
# ### 2.

# %%
# function d theta / dt
def dtheta_dt(t,thetas,omegas,K):
    global R
    N = len(thetas)
    sum_s = np.sum(np.sin(thetas))
    sum_c = np.sum(np.cos(thetas))
    psi = np.arctan(sum_s/sum_c)
    r = sum_c/(N*np.cos(psi))
    return omegas + K*r*np.sin(psi*np.ones(N)-thetas)

# %%
N = 10
K = [0,0.5,1,2,5]
omegas = np.random.normal(size=N)
thetas = np.random.uniform(-np.pi,np.pi,size=N)
t_max = 100
t = np.arange(1,t_max+1)

fig,ax = plt.subplots(len(K),1,figsize=(10,30))

for i,k in enumerate(K):
    sol = solve_ivp(dtheta_dt,[0,t_max],thetas,args=(omegas,k),t_eval=np.arange(0,t_max))
    for j in np.arange(0,N):
        ax[i].plot(sol.y[j]/t,label='%2f'%(omegas[j]))
    ax[i].set_title("K = "+str(k))
    ax[i].grid()
    ax[i].legend(loc='upper center',fontsize=15)
print(omegas)

# notice dependence of final state

# %% [markdown]
# ### 3.

# %%
N = 100
K = np.linspace(1,2,25)
omegas = np.random.normal(size=N)
mean_omega = sum(omegas)/N
omegas = omegas - mean_omega
thetas = np.random.uniform(-np.pi,np.pi,size=N)
t_max = 100
t = np.arange(1,t_max+1)

fig,ax = plt.subplots(figsize=(15,10))


for i,k in enumerate(K):
    sol = solve_ivp(dtheta_dt,[0,t_max],thetas,args=(omegas,k),t_eval=np.arange(0,t_max))
    if(i%5==0):
        sum_s = np.sum(np.sin(sol.y),0)
        sum_c = np.sum(np.cos(sol.y),0)
        psi = np.arctan(sum_s/sum_c)
        r = np.abs(sum_c/(N*np.cos(psi)))
        ax.plot(r,label=str(k))
        ax.grid()
    ax.legend()


# %%
N = 100
K = np.linspace(1,2,25)
t_max = 100
N_sim = 100
t = np.arange(1,t_max+1)


fig,ax = plt.subplots(figsize=(15,10))
R = {k:0 for k in K}

for n_sim in np.arange(0,N_sim):
    omegas = np.random.normal(size=N)
    mean_omega = sum(omegas)/N
    omegas = omegas - mean_omega
    thetas = np.random.uniform(-np.pi,np.pi,size=N)

    for i,k in enumerate(K):
        sol = solve_ivp(dtheta_dt,[0,t_max],thetas,args=(omegas,k),t_eval=np.arange(0,t_max))
        y = sol.y[:,50:]
        sum_s = np.sum(np.sin(y),0)
        sum_c = np.sum(np.cos(y),0)
        psi = np.arctan(sum_s/sum_c)
        r = np.mean(np.abs(sum_c/(N*np.cos(psi))))
        R[k]+=r/N_sim

# %%
R_mean = [R[k] for k in K]
plt.plot(R_mean)
plt.grid()

# %%



