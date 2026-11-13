import numpy as np
import matplotlib.pyplot as plt

#########################################
#
# Parâmetros de Malha
#

n = 8
Nx = 2**n
Ny = 2**n

#########################################
#
# Parâmetros LBM
#

c_s = 1 / np.sqrt(3)
Re_w = 100 #(100, 400, 1000, 3200, 5000, 10000)
Re_oil = 0.1*Re_w
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
# Parâmetros de Iteração
#

maxIter = 20000
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
rho_w = np.ones((Ny, Nx))
ux_w = np.zeros((Ny, Nx))
uy_w = np.zeros((Ny, Nx))

f_oil = np.zeros((9, Ny, Nx))
rho_oil = 0.1*np.zeros((Ny, Nx))
ux_oil = np.zeros((Ny, Nx))
uy_oil = np.zeros((Ny, Nx))

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
    ux_w = np.sum(f_w * cx[:, None, None], axis=0) / rho_w
    ux_oil = np.sum(f_oil * cx[:, None, None], axis=0) / rho_w
    uy_w = np.sum(f_w * cy[:, None, None], axis=0) / rho_oil
    uy_oil = np.sum(f_oil * cy[:, None, None], axis=0) / rho_oil

#    cu = cx[:, None, None] * ux[None, :, :] + cy[:, None, None] * uy[None, :, :]
 #   u2 = ux*2 + uy*2
#    feq = w[:, None, None] * rho[None, :, :] * (1 + 3*cu + 4.5*cu**2 - 1.5*u2[None, :, :])

    f_w = f_w - (f_w - feq) / tau_w
    f_oil = f_oil - (f_oil - feq) / tau_oil
    
    for i in range(9):
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
    f_w[7, -1, :] = f_w[5, -1, :] - 6 * w[7] * rho_top_w * U_lid
    f_w[8, -1, :] = f_w[6, -1, :] + 6 * w[8] * rho_top_w * U_lid
    
    rho_top_oil = np.sum(f_oil[:, -1, :], axis=0)
    f_oil[4, -1, :] = f_oil[2, -1, :]
    f_oil[7, -1, :] = f_oil[5, -1, :] - 6 * w[7] * rho_top_oil * U_lid
    f_oil[8, -1, :] = f_oil[6, -1, :] + 6 * w[8] * rho_top_oil * U_lid

    if it % 200 == 0 and it > 0:
        print(f"Iter {it}, max ux = {np.max(ux):.4f}")
        err = np.max(np.abs(ux - ux_w_old)) / U_lid
        ux_w_old = ux.copy()
    
    if err < Iter_Tol:
        break
        
rho = np.sum(f, axis=0)
ux = np.sum(f * cx[:, None, None], axis=0) / rho
uy = np.sum(f * cy[:, None, None], axis=0) / rho
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(np.sqrt(ux**2 + uy**2)/U_lid, origin='lower', cmap='jet')
plt.streamplot(X, Y, ux/U_lid, uy/U_lid, density=1.2, color='white', linewidth=0.6, arrowsize=0.8)
plt.colorbar(label='|U|')
plt.title('Lid-Driven Cavity, Re=1000')

# Ghia 1982 data, Re=1000, u along vertical centreline
y_ghia = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0])
u_ghia = np.array([1.0, 0.65928, 0.57492, 0.51117, 0.46622, 0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.0])

plt.subplot(1, 2, 2)
plt.plot(ux[:, Nx//2]/U_lid, np.linspace(0, 1, Ny), label='LBM')
plt.plot(u_ghia, y_ghia, 'o', label='Ghia et al. 1982')
plt.xlabel('u / U_lid'); plt.ylabel('y / L')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()