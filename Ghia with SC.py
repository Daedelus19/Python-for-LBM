import numpy as np
import matplotlib.pyplot as plt

#########################################
#
# Parâmetros de Malha
#

n = 7
fn = 1
Nx = fn*2**n
Ny = fn*2**n

#########################################
#
# Parâmetros LBM
#

c_s = 1 / np.sqrt(3)
Re_w = 100
Re_oil = 0.9*Re_w
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

G = 30
psi0 = 4
rho0 = 200

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

maxIter = 10000
Iter_Tol = 1e-5
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
rho_w = np.ones((Ny, Nx))
ux_w = np.zeros((Ny, Nx))
uy_w = np.zeros((Ny, Nx))

f_oil = np.zeros((9, Ny, Nx))
rho_oil = 0.1*np.ones((Ny, Nx))
ux_oil = np.zeros((Ny, Nx))
uy_oil = np.zeros((Ny, Nx))

psi = np.zeros((Ny, Nx))
Fx = np.zeros((Ny, Nx))
Fy = np.zeros((Ny, Nx))

ux_w_old = ux_w.copy()
    
for i in range(9):
    f_w[i, :, :] = w[i] * rho_w
    f_oil[i, :, :] = w[i] * rho_oil
    
#########################################
#
# Main Loop 
#

for it in range(maxIter):

    rho_w = np.sum(f_w, axis=0)
    rho_oil = np.sum(f_oil, axis=0)
    rho = rho_w + rho_oil
    
    ux_w = np.sum(f_w * cx[:, None, None], axis=0) / rho_w
    ux_oil = np.sum(f_oil * cx[:, None, None], axis=0) / rho_oil
    
    uy_w = np.sum(f_w * cy[:, None, None], axis=0) / rho_w
    uy_oil = np.sum(f_oil * cy[:, None, None], axis=0) / rho_oil
    
    ux = (np.sum(f_w * cx[:, None, None], axis=0)/tau_w + np.sum(f_oil * cx[:, None, None], axis=0)/tau_oil) / (rho_w/tau_w + rho_oil/tau_oil)
    uy = (np.sum(f_w * cy[:, None, None], axis=0)/tau_w + np.sum(f_oil * cy[:, None, None], axis=0)/tau_oil) / (rho_w/tau_w + rho_oil/tau_oil)
    
    psi = psi0*(1-np.exp(-rho/rho0))
    
    Fx = -G*psi*(
    (1/9) * np.roll(psi, -1, axis=1) +
    -(1/9) * np.roll(psi, +1, axis=1) +
    (1/36) * np.roll(psi, shift=(-1,-1), axis=(0,1)) +
    -(1/36) * np.roll(psi, shift=(-1,+1), axis=(0,1)) +
    -(1/36) * np.roll(psi, shift=(+1,+1), axis=(0,1)) +
    (1/36) * np.roll(psi, shift=(+1,-1), axis=(0,1)))
    
    psi = psi0*(1-np.exp(-rho0/rho))

    Fy = -G*psi*(
    (1/9) * np.roll(psi, -1, axis=0) +
    -(1/9) * np.roll(psi, +1, axis=0) +
    (1/36) * np.roll(psi, shift=(-1,-1), axis=(0,1)) +
    (1/36) * np.roll(psi, shift=(-1,+1), axis=(0,1)) +
    -(1/36) * np.roll(psi, shift=(+1,+1), axis=(0,1)) +
    -(1/36) * np.roll(psi, shift=(+1,-1), axis=(0,1)))

    ux_eq_w = ux + tau_w*Fx/rho_w
    uy_eq_w = uy + tau_w*Fy/rho_w                   #-glbm * tu_w
    ux_eq_oil = ux + tau_oil*Fx/rho_oil
    uy_eq_oil = uy + tau_oil*Fy/rho_oil             #-glbm * tu_oil
    
    cu_w = cx[:, None, None] * ux_eq_w[None, :, :] + cy[:, None, None] * uy_eq_w[None, :, :]
    cu_oil = cx[:, None, None] * ux_eq_oil[None, :, :] + cy[:, None, None] * uy_eq_oil[None, :, :]
    
    u2_w = ux_eq_w**2 + uy_eq_w**2
    u2_oil = ux_eq_oil**2 + uy_eq_oil**2
    
    feq_w = w[:, None, None] * rho_w[None, :, :] * (1 + 3*cu_w + 4.5*cu_w**2 - 1.5*u2_w[None, :, :])
    feq_oil = w[:, None, None] * rho_oil[None, :, :] * (1 + 3*cu_oil + 4.5*cu_oil**2 - 1.5*u2_oil[None, :, :])

    f_w = f_w - (f_w - feq_w) / tau_w
    f_oil = f_oil - (f_oil - feq_oil) / tau_oil
    
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
    f_w[7, -1, :] = f_w[5, -1, :] - 6 * w[7] * rho_top_w * U_lid
    f_w[8, -1, :] = f_w[6, -1, :] + 6 * w[8] * rho_top_w * U_lid
    
    rho_top_oil = np.sum(f_oil[:, -1, :], axis=0)
    f_oil[4, -1, :] = f_oil[2, -1, :]
    f_oil[7, -1, :] = f_oil[5, -1, :] - 6 * w[7] * rho_top_oil * U_lid
    f_oil[8, -1, :] = f_oil[6, -1, :] + 6 * w[8] * rho_top_oil * U_lid
    
    for i in range(9):
        f_w[i, :, :] = np.roll(f_w[i, :, :], shift=(cy[i], cx[i]), axis=(0, 1))
        f_oil[i, :, :] = np.roll(f_oil[i, :, :], shift=(cy[i], cx[i]), axis=(0, 1))
        
        print(f"it={it} rho min={rho.min():.3f} max={rho.max():.3f} NaN={np.isnan(rho).sum()}")
        print(f"it={it} psi min={rho.min():.3f} max={psi.max():.3f} NaN={np.isnan(psi).sum()}") 
        print(f"it={it} Fx min={rho.min():.3f} max={Fx.max():.3f} NaN={np.isnan(Fx).sum()}")
        print(f"it={it} Fy min={rho.min():.3f} max={Fy.max():.3f} NaN={np.isnan(Fy).sum()}")

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

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(phi, origin='lower', cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Oil fraction')
plt.title(f'Oil distribution (10%), G={G}')
plt.xlabel('x'); plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.imshow(np.sqrt(ux**2 + uy**2)/U_lid, origin='lower', cmap='viridis')
plt.streamplot(X, Y, ux/U_lid, uy/U_lid, density=1.2, color='white', linewidth=0.6, arrowsize=0.8)
plt.colorbar(label='|U| / U_lid')
plt.title('Velocity + Streamlines')

plt.tight_layout()
plt.show()