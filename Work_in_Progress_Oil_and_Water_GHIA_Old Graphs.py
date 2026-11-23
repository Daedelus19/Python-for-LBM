import matplotlib.pyplot as plt
import numpy as np

#########################################
#
# Parâmetros de Malha
#

n = 9
fn = 1
Nx = fn*2**n
Ny = fn*2**n

h = 0.2 #Fração Volumétrica de Óleo

Oil_Ball = True #True
Oil_Seeds = False #False

#########################################
#
# Parâmetros LBM
#

alpha = 0.5 #0.00113
c_s = 1 / np.sqrt(3)
Re_w = 100
Re_oil = alpha*Re_w
Ma = 0.1

U_lid = Ma * c_s;
nu_w = U_lid * (Nx - 1) / Re_w
nu_oil = U_lid * (Nx - 1) / Re_oil

tau_w = (3 * nu_w) + 0.5
tau_oil = (3 * nu_oil) + 0.5

print(tau_w)
print(tau_oil)
print(U_lid)

#########################################
#
# Shan Chen Parameters 
#

G_ww = -5
G_oo = -5
G_wo = 5
G_ow = G_wo
psi0 = 1
rho0 = 1

#########################################
#
# Parâmetros Físicos
#

L = 0.1 # m
V = Re_w/L/1000000
g = 9.81 # m /s ** 2

dx = L / (Nx-1)
dt = U_lid/V
glbm = g*dt**2/dx

print(glbm)
print(dx)
print(dt)

#########################################
#
# Parâmetros de Iteração
#

maxIter = 1001
Iter_Tol = 1e-9
err = 1

#########################################
#
# D2Q9 Arrays
#

cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

#########################################
#
# Inicialização 
#

f_w = np.zeros((9, Ny, Nx))
rho_w = (1-h)*np.ones((Ny, Nx))
f_oil = np.zeros((9, Ny, Nx))
rho_oil = h*np.ones((Ny, Nx))


if Oil_Ball:
    X, Y = np.meshgrid(np.arange(1, Nx+1), np.arange(1, Ny+1))        #Oil Ball 
    r = np.sqrt((X - Nx / 2)**2 + (Y - Ny / 2)**2)
    oil_frac = 0.5 * (1 - np.tanh((r - Nx / 4) / 4))
    rho_w = 1.0 - 0.95 * oil_frac
    rho_oil = 0.05 + 0.95 * oil_frac
    
if Oil_Seeds:
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    n_seed_1d = max(2, n - 4)
    R_seed = Nx // 8
    coords_1d = (np.arange(n_seed_1d) + 0.5) / n_seed_1d * Nx
    np.random.seed(0)
    noise = 0.02 * (np.random.rand(Ny, Nx) - 0.5)
    rho_w = (1 - h) + noise
    rho_oil = h - noise
    oil_frac_seeds = np.zeros((Ny, Nx))
    for xc in coords_1d:
        for yc in coords_1d:
            r = np.sqrt((X - xc)**2 + (Y - yc)**2)
            oil_frac_seeds += 0.5 * (1 - np.tanh((r - R_seed) / 2))
    oil_frac_seeds = np.clip(oil_frac_seeds, 0, 1)
    rho_oil = 0.05 + 0.95 * oil_frac_seeds + (h - 0.05) * (1 - oil_frac_seeds) + noise
    rho_w = 1.05 - rho_oil
    rho_w = np.maximum(rho_w, 0.05)
    rho_oil = np.maximum(rho_oil, 0.05)

ux_w = np.zeros((Ny, Nx))
uy_w = np.zeros((Ny, Nx))
ux_oil = np.zeros((Ny, Nx))
uy_oil = np.zeros((Ny, Nx))

f_w = np.zeros((9, Ny, Nx))
f_oil = np.zeros((9, Ny, Nx))

for i in range(9):
    f_w[i, :, :] = w[i] * rho_w
    f_oil[i, :, :] = w[i] * rho_oil

psi_w = np.zeros((Ny, Nx))
psi_oil = np.zeros((Ny, Nx))

ux_w_old = ux_w.copy()
    
for i in range(9):
    f_w[i, :, :] = w[i] * rho_w
    f_oil[i, :, :] = w[i] * rho_oil
    
oil_mass_history = []
water_mass_history = []
time_history = []

oil_mass_0 = np.sum(rho_oil)
    
#########################################
#
# Main Loop 
#

for it in range(maxIter):

    if it < 1000:
        U_lid_eff = 0
    else:
        U_lid_eff = U_lid
    
    rho_w = np.sum(f_w, axis=0)
    rho_oil = np.sum(f_oil, axis=0)
    
    Fx_w = np.zeros((Ny, Nx))
    Fy_w = np.zeros((Ny, Nx))
    Fx_oil = np.zeros((Ny, Nx))
    Fy_oil = np.zeros((Ny, Nx))
    
    if it % 50 == 0:
        oil_mass_history.append(np.sum(rho_oil))
        water_mass_history.append(np.sum(rho_w))
        time_history.append(it)
    
    rho = rho_w + rho_oil
    
    ux_w = np.sum(f_w * cx[:, None, None], axis=0) / rho_w
    ux_oil = np.sum(f_oil * cx[:, None, None], axis=0) / rho_oil
    
    uy_w = np.sum(f_w * cy[:, None, None], axis=0) / rho_w
    uy_oil = np.sum(f_oil * cy[:, None, None], axis=0) / rho_oil
    
    ux = (np.sum(f_w * cx[:, None, None], axis=0)/tau_w + np.sum(f_oil * cx[:, None, None], axis=0)/tau_oil) / (rho_w/tau_w + rho_oil/tau_oil)
    uy = (np.sum(f_w * cy[:, None, None], axis=0)/tau_w + np.sum(f_oil * cy[:, None, None], axis=0)/tau_oil) / (rho_w/tau_w + rho_oil/tau_oil)
    
    psi_w = psi0*(1-np.exp(-rho_w/rho0))
    psi_oil = psi0*(1-np.exp(-rho_oil/rho0))
    
    for i in range(9):

        psi_w_neighbor = np.roll(psi_w, shift=(cy[i], cx[i]), axis=(0, 1))
        if cy[i] == 1:
            psi_w_neighbor[0, :] = psi_w[0, :]
        if cy[i] == -1:
            psi_w_neighbor[-1, :] = psi_w[-1, :]
        if cx[i] == 1:
            psi_w_neighbor[:, 0] = psi_w[:, 0]
        if cx[i] == -1:
            psi_w_neighbor[:, -1] = psi_w[:, -1]
        
        psi_oil_neighbor = np.roll(psi_oil, shift=(cy[i], cx[i]), axis=(0, 1))
        if cy[i] == 1:
            psi_oil_neighbor[0, :] = psi_oil[0, :]
        if cy[i] == -1:
            psi_oil_neighbor[-1, :] = psi_oil[-1, :]
        if cx[i] == 1:
            psi_oil_neighbor[:, 0] = psi_oil[:, 0]
        if cx[i] == -1:
            psi_oil_neighbor[:, -1] = psi_oil[:, -1]

        Fx_w += w[i] * (G_ww * psi_w_neighbor + G_wo * psi_oil_neighbor) * cx[i]

    Fx_w = -psi_w * Fx_w
        
    for i in range(9):

        psi_w_neighbor = np.roll(psi_w, shift=(cy[i], cx[i]), axis=(0, 1))
        if cy[i] == 1:
            psi_w_neighbor[0, :] = psi_w[0, :]
        if cy[i] == -1:
            psi_w_neighbor[-1, :] = psi_w[-1, :]
        if cx[i] == 1:
            psi_w_neighbor[:, 0] = psi_w[:, 0]
        if cx[i] == -1:
            psi_w_neighbor[:, -1] = psi_w[:, -1]
            
        psi_oil_neighbor = np.roll(psi_oil, shift=(cy[i], cx[i]), axis=(0, 1))
        if cy[i] == 1:
            psi_oil_neighbor[0, :] = psi_oil[0, :]
        if cy[i] == -1:
            psi_oil_neighbor[-1, :] = psi_oil[-1, :]
        if cx[i] == 1:
            psi_oil_neighbor[:, 0] = psi_oil[:, 0]
        if cx[i] == -1:
            psi_oil_neighbor[:, -1] = psi_oil[:, -1]

        Fy_w += w[i] * (G_ww * psi_w_neighbor + G_wo * psi_oil_neighbor) * cy[i]

    Fy_w = -psi_w * Fy_w
    
    for i in range(9):

        psi_w_neighbor = np.roll(psi_w, shift=(cy[i], cx[i]), axis=(0, 1))
        if cy[i] == 1:
            psi_w_neighbor[0, :] = psi_w[0, :]
        if cy[i] == -1:
            psi_w_neighbor[-1, :] = psi_w[-1, :]
        if cx[i] == 1:
            psi_w_neighbor[:, 0] = psi_w[:, 0]
        if cx[i] == -1:
            psi_w_neighbor[:, -1] = psi_w[:, -1]
            
        psi_oil_neighbor = np.roll(psi_oil, shift=(cy[i], cx[i]), axis=(0, 1))
        if cy[i] == 1:
            psi_oil_neighbor[0, :] = psi_oil[0, :]
        if cy[i] == -1:
            psi_oil_neighbor[-1, :] = psi_oil[-1, :]
        if cx[i] == 1:
            psi_oil_neighbor[:, 0] = psi_oil[:, 0]
        if cx[i] == -1:
            psi_oil_neighbor[:, -1] = psi_oil[:, -1]

        Fx_oil += w[i] * (G_oo * psi_oil_neighbor + G_ow * psi_w_neighbor) * cx[i]

    Fx_oil = -psi_oil * Fx_oil
        
    for i in range(9):

        psi_w_neighbor = np.roll(psi_w, shift=(cy[i], cx[i]), axis=(0, 1))
        if cy[i] == 1:
            psi_w_neighbor[0, :] = psi_w[0, :]
        if cy[i] == -1:
            psi_w_neighbor[-1, :] = psi_w[-1, :]
        if cx[i] == 1:
            psi_w_neighbor[:, 0] = psi_w[:, 0]
        if cx[i] == -1:
            psi_w_neighbor[:, -1] = psi_w[:, -1]
            
        psi_oil_neighbor = np.roll(psi_oil, shift=(cy[i], cx[i]), axis=(0, 1))
        if cy[i] == 1:
            psi_oil_neighbor[0, :] = psi_oil[0, :]
        if cy[i] == -1:
            psi_oil_neighbor[-1, :] = psi_oil[-1, :]
        if cx[i] == 1:
            psi_oil_neighbor[:, 0] = psi_oil[:, 0]
        if cx[i] == -1:
            psi_oil_neighbor[:, -1] = psi_oil[:, -1]

        Fy_oil += w[i] * (G_oo * psi_oil_neighbor + G_ow * psi_w_neighbor) * cy[i]

    Fy_oil = -psi_oil * Fy_oil

    ux_eq_w = ux + tau_w*Fx_w/rho_w
    uy_eq_w = uy + tau_w*Fy_w/rho_w                   #-glbm * tu_w
    ux_eq_oil = ux + tau_oil*Fx_oil/rho_oil
    uy_eq_oil = uy + tau_oil*Fy_oil/rho_oil             #-glbm * tu_oil
    
    cu_w = cx[:, None, None] * ux_eq_w[None, :, :] + cy[:, None, None] * uy_eq_w[None, :, :]
    cu_oil = cx[:, None, None] * ux_eq_oil[None, :, :] + cy[:, None, None] * uy_eq_oil[None, :, :]
    
    u2_w = ux_eq_w**2 + uy_eq_w**2
    u2_oil = ux_eq_oil**2 + uy_eq_oil**2
    
    feq_w = w[:, None, None] * rho_w[None, :, :] * (1 + 3*cu_w + 4.5*cu_w**2 - 1.5*u2_w[None, :, :])
    feq_oil = w[:, None, None] * rho_oil[None, :, :] * (1 + 3*cu_oil + 4.5*cu_oil**2 - 1.5*u2_oil[None, :, :])

    f_w = f_w - (f_w - feq_w) / tau_w
    f_oil = f_oil - (f_oil - feq_oil) / tau_oil
    
    for i in range(9):#Streaming
        f_w[i, :, :] = np.roll(f_w[i, :, :], shift=(cy[i], cx[i]), axis=(0, 1))
        f_oil[i, :, :] = np.roll(f_oil[i, :, :], shift=(cy[i], cx[i]), axis=(0, 1))
    
    # Bottom
    f_w[2, 0, :] = f_w[4, 0, :]
    f_w[5, 0, :] = f_w[7, 0, :]
    f_w[6, 0, :] = f_w[8, 0, :]
    
    f_oil[2, 0, :] = f_oil[4, 0, :]
    f_oil[5, 0, :] = f_oil[7, 0, :]
    f_oil[6, 0, :] = f_oil[8, 0, :]
    
    # Left
    f_w[1, :, 0] = f_w[3, :, 0]
    f_w[5, :, 0] = f_w[7, :, 0]
    f_w[8, :, 0] = f_w[6, :, 0]

    f_oil[1, :, 0] = f_oil[3, :, 0]
    f_oil[5, :, 0] = f_oil[7, :, 0]
    f_oil[8, :, 0] = f_oil[6, :, 0]
    
    # Right
    f_w[3, :, -1] = f_w[1, :, -1]
    f_w[6, :, -1] = f_w[8, :, -1]
    f_w[7, :, -1] = f_w[5, :, -1]
    
    f_oil[3, :, -1] = f_oil[1, :, -1]
    f_oil[6, :, -1] = f_oil[8, :, -1]
    f_oil[7, :, -1] = f_oil[5, :, -1]

    # Moving lid
    rho_top_w = np.sum(f_w[:, -1, :], axis=0)
    f_w[4, -1, :] = f_w[2, -1, :]
    f_w[7, -1, :] = f_w[5, -1, :] - 6 * w[7] * rho_top_w * U_lid_eff
    f_w[8, -1, :] = f_w[6, -1, :] + 6 * w[8] * rho_top_w * U_lid_eff
    
    rho_top_oil = np.sum(f_oil[:, -1, :], axis=0)
    f_oil[4, -1, :] = f_oil[2, -1, :]
    f_oil[7, -1, :] = f_oil[5, -1, :] - 6 * w[7] * rho_top_oil * U_lid_eff
    f_oil[8, -1, :] = f_oil[6, -1, :] + 6 * w[8] * rho_top_oil * U_lid_eff
        
    print(f"it={it} psi_w min={psi_w.min():.3f} max={psi_w.max():.3f} NaN={np.isnan(psi_w).sum()}")
    print(f"it={it} psi_oil min={psi_oil.min():.3f} max={psi_oil.max():.3f} NaN={np.isnan(psi_oil).sum()}")
    print(f"it={it} Fx_oil min={Fx_oil.min():.3f} max={Fx_oil.max():.3f} NaN={np.isnan(Fx_oil).sum()}")
    print(f"it={it} Fy_oil min={Fy_oil.min():.3f} max={Fy_oil.max():.3f} NaN={np.isnan(Fy_oil).sum()}")

    if it % 200 == 0 and it > 0:
        print(f"Iter {it}, max ux = {np.max(ux):.4f}")
        err = np.max(np.abs(ux_w - ux_w_old)) / U_lid
        ux_w_old = ux.copy()
    
    if err < Iter_Tol:
        break

#########################################
#
# Final Solution 
#
        
rho_w = np.sum(f_w, axis=0)
rho_oil = np.sum(f_oil, axis=0)
rho = rho_w + rho_oil

ux = (np.sum(f_w * cx[:, None, None], axis=0) + np.sum(f_oil * cx[:, None, None], axis=0)) / rho
uy = (np.sum(f_w * cy[:, None, None], axis=0) + np.sum(f_oil * cy[:, None, None], axis=0)) / rho
phi = rho_oil / (rho)

X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))

#########################################
#
# Plot 
#

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(phi, origin='lower', cmap='RdBu_r', vmin=0, vmax=1)
plt.colorbar(label='Oil fraction')
plt.title(f'Oil distribution (20%)')
plt.xlabel('x'); plt.ylabel('y')
plt.xlim(0,Nx-1)
plt.ylim(0,Ny-1)

plt.subplot(2, 2, 2)
plt.imshow(np.sqrt(ux**2 + uy**2)/U_lid, origin='lower', cmap='viridis')
plt.streamplot(X, Y, ux/U_lid, uy/U_lid, density=1.2, color='white', linewidth=0.6, arrowsize=0.8)
plt.colorbar(label='|U| / U_lid')
plt.title('Bulk Velocity + Streamlines')
plt.xlim(0,Nx-1)
plt.ylim(0,Ny-1)

plt.subplot(2, 2, 3)
plt.imshow(np.sqrt(ux_w**2 + uy_w**2)/U_lid, origin='lower', cmap='viridis')
plt.streamplot(X, Y, ux_w/U_lid, uy_w/U_lid, density=1.2, color='white', linewidth=0.6, arrowsize=0.8)
plt.colorbar(label='|U_w| / U_lid')
plt.title('Water Velocity + Streamlines')
plt.xlim(0,Nx-1)
plt.ylim(0,Ny-1)

plt.subplot(2, 2, 4)
plt.imshow(np.sqrt(ux_oil**2 + uy_oil**2)/U_lid, origin='lower', cmap='viridis')
plt.streamplot(X, Y, ux_oil/U_lid, uy_oil/U_lid, density=1.2, color='white', linewidth=0.6, arrowsize=0.8)
plt.colorbar(label='|U_oil| / U_lid')
plt.title('Oil Velocity + Streamlines')
plt.xlim(0,Nx-1)
plt.ylim(0,Ny-1)

oil_mass_history = np.array(oil_mass_history)
water_mass_history = np.array(water_mass_history)
time_history = np.array(time_history)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
oil_mass_history = np.array(oil_mass_history)
water_mass_history = np.array(water_mass_history)
time_history = np.array(time_history)

plt.plot(time_history, oil_mass_history, label='Oil', color='red', linewidth=2)
plt.plot(time_history, water_mass_history, label='Water', color='blue', linewidth=2)
plt.xlabel('Time step')
plt.ylabel('Total mass')
plt.title('Mass conservation over time')
plt.xlim(1, maxIter)
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 2)
im_w = plt.imshow(rho_w, origin='lower', cmap='viridis')
plt.colorbar(im_w, label=r'$\rho_w$')
plt.title(f'rho_w\nmin={rho_w.min():.3f}  max={rho_w.max():.3f}  std={rho_w.std():.4f}')
plt.xlabel('x'); plt.ylabel('y')
plt.xlim(0, Nx-1)
plt.ylim(0, Ny-1)

plt.subplot(1, 3, 3)
im_oil = plt.imshow(rho_oil, origin='lower', cmap='viridis')
plt.colorbar(im_oil, label=r'$\rho_{oil}$')
plt.title(f'rho_oil\nmin={rho_oil.min():.3f}  max={rho_oil.max():.3f}  std={rho_oil.std():.4f}')
plt.xlabel('x'); plt.ylabel('y')
plt.xlim(0, Nx-1)
plt.ylim(0, Ny-1)

plt.tight_layout()
plt.show()