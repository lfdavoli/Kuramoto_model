# %% [markdown]
#  # Kuramoto model

# %%
# Libs
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# %% [markdown]
#  ### 1.
# Start from:
# 
# $r(t)e^{-i\Psi(t)} = \frac{1}{N}\sum_{i=1}^{N} e^{-i\theta_i}$
# 
# dividing real and imaginary part:
# 
# $Re:\;\;\; rcos(-\Psi) = \frac{1}{N}\sum_{i=1}^{N} cos(-\theta)$
# 
# $\;\;\;\;\;\;\;\;\;\; rcos(\Psi) = \frac{1}{N}\sum_{i=1}^{N} cos(\theta)$
# 
# $Im:\;\;\; irsin(-\Psi) = \frac{1}{N}\sum_{i=1}^{N} isin(-\theta)$
# 
# $\;\;\;\;\;\;\;\;\;\; -irsin(\Psi) = \frac{1}{N}\sum_{i=1}^{N} -isin(\theta)$
# 
# $\;\;\;\;\;\;\;\;\;\; irsin(\Psi) = \frac{1}{N}\sum_{i=1}^{N} isin(\theta)$
# 
# finally obtaining the original formulation:
# 
# $r(t)e^{i\Psi(t)} = \frac{1}{N}\sum_{i=1}^{N} e^{i\theta_i}$
# 
# Using this result we can work on the actual equation of motion:
# 
# $\dot{\theta_i}=\omega_i+\frac{K}{N}\sum_{j}\frac{1}{2i}[e^{i(\theta_j - \theta_i)}-e^{-i(\theta_j - \theta_i)}]$
# 
# $\dot{\theta_i}=\omega_i+[\frac{K}{2iN}e^{-i\theta_i}\sum_{j}e^{i\theta_j} - \frac{K}{2iN}e^{i\theta_i}\sum_{j}e^{-i\theta_j}]$
# 
# $\dot{\theta_i}=\omega_i+[\frac{K}{2i}e^{-i\theta_i}r(t)e^{i\Psi(t)} - \frac{K}{2i}e^{i\theta_i}r(t)e^{-i\Psi(t)}]$
# 
# $\dot{\theta_i}=\omega_i+Kr(t)sin(\Psi - \theta_i)$
# 

# %% [markdown]
#  ### 2.

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
#  #### Considerations:

# %% [markdown]
#  ### 3.

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
sol


# %%
sol.y[0][0]-sol.y[1][0]


# %% [markdown]
#  ### 4.

# %%
N = 100
K = np.linspace(1,2,25)
t_max = 100
N_sim = 100
t = np.arange(1,t_max+1)

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


# %% [markdown]
#  ### 5.

# %%
N = [100,500,2000,5000,15000]
N_rep = [300,100,50,30,10]
K = np.linspace(1,2,25)
t_max = 100
t = np.arange(1,t_max+1)

R_dict = {n:[0 for k in K] for n in N}

for i,n in enumerate(N):
    N_sim = N_rep[i]

    for n_sim in np.arange(0,N_sim):
        omegas = np.random.normal(size=n)
        mean_omega = sum(omegas)/n
        omegas = omegas - mean_omega
        thetas = np.random.uniform(-np.pi,np.pi,size=n)

        for i,k in enumerate(K):
            sol = solve_ivp(dtheta_dt,[0,t_max],thetas,args=(omegas,k),t_eval=np.arange(0,t_max))
            y = sol.y[:,50:]
            sum_s = np.sum(np.sin(y),0)
            sum_c = np.sum(np.cos(y),0)
            psi = np.arctan(sum_s/sum_c)
            r = np.mean(np.abs(sum_c/(n*np.cos(psi))))
            R_dict[n][i]+=r/N_sim
            


# %%
fig,ax = plt.subplots()
for key in R_dict.keys():
    ax.plot(K,R_dict[key],label='$N_{rotors} = $'+str(key))

ax.legend()
ax.grid()
ax.set_title('Average R as function of K and rotors number')
ax.set_xlabel('K')
ax.set_ylabel('R')


# %% [markdown]
#  ### 6.
#  Assuming normal distribution of omegas (CLT):
# 
#  $p(0) = p(\omega=0)_{K} = \frac{1}{\sqrt{2 \pi}}$
# 
#  $K_c = \frac{2}{\pi p_0}$

# %%
p_0 = 1/np.sqrt(2*np.pi)
K_c = 2/(np.pi*p_0)
K_c


# %%
p_0 = 1/np.sqrt(2*np.pi)
K_c = 2/(np.pi*p_0)

fig,ax = plt.subplots()
for key in R_dict.keys():
    ax.plot(K,R_dict[key],label='$N_{rotors} = $'+str(key))

ax.axvline(x=K_c, label='$K_c$ threshold',color='g')

ax.legend()
ax.grid()
ax.set_title('Average R as function of K and rotors number')
ax.set_xlabel('K')
ax.set_ylabel('R')


# %%






