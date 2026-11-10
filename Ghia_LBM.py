import numpy as np
import matplotlib.pyplot as plt

Nx = 129
Ny = 129
tau = 0.6
ni = (tau-0.5)/3
Re = 100 #(100, 400, 1000, 3200, 5000, 10000)
U_lid = ni*Re/(Nx-1)
Ma = U_lid*3**(1/2)

print(U_lid)
print(Ma)

maxIter = 10000

cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

f = np.zeros((9, Ny, Nx))
rho = np.ones((Ny, Nx))
ux = np.zeros((Ny, Nx))
uy = np.zeros((Ny, Nx))
for i in range(9):
    f[i, :, :] = w[i] * rho

for it in range(maxIter):

    rho = np.sum(f, axis=0)
    ux = np.sum(f * cx[:, None, None], axis=0) / rho
    uy = np.sum(f * cy[:, None, None], axis=0) / rho

    cu = cx[:, None, None] * ux[None, :, :] + cy[:, None, None] * uy[None, :, :]
    u2 = ux**2 + uy**2
    feq = w[:, None, None] * rho[None, :, :] * (1 + 3*cu + 4.5*cu**2 - 1.5*u2[None, :, :])

    f = f - (f - feq) / tau
    
    for i in range(9):
        f[i, :, :] = np.roll(f[i, :, :], shift=(cy[i], cx[i]), axis=(0, 1))
        
    # Bottom
    f[2, 0, :] = f[4, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[6, 0, :] = f[8, 0, :]
    # Left
    f[1, :, 0] = f[3, :, 0]
    f[5, :, 0] = f[7, :, 0]
    f[8, :, 0] = f[6, :, 0]
    # Right
    f[3, :, -1] = f[1, :, -1]
    f[6, :, -1] = f[8, :, -1]
    f[7, :, -1] = f[5, :, -1]

    # Moving lid
    rho_top = np.sum(f[:, -1, :], axis=0)
    f[4, -1, :] = f[2, -1, :]
    f[7, -1, :] = f[5, -1, :] - 6 * w[7] * rho_top * U_lid
    f[8, -1, :] = f[6, -1, :] + 6 * w[8] * rho_top * U_lid

    if it % 2000 == 0:
        print(f"Iter {it}, max ux = {np.max(ux):.4f}")
        
rho = np.sum(f, axis=0)
ux = np.sum(f * cx[:, None, None], axis=0) / rho
uy = np.sum(f * cy[:, None, None], axis=0) / rho

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(np.sqrt(ux**2 + uy**2), origin='lower', cmap='jet')
plt.colorbar(label='|U|')
plt.title('Lid-Driven Cavity, Re=1000')

# Ghia 1982 data, Re=1000, u along vertical centreline
y_ghia = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0])
u_ghia = np.array([1.0, 0.65928, 0.57492, 0.51117, 0.46622, 0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.0])

plt.subplot(1, 2, 2)
plt.plot(ux[:, Nx//2], np.linspace(0, 1, Ny), label='LBM')
plt.plot(u_ghia, y_ghia, 'o', label='Ghia et al. 1982')
plt.xlabel('u / U_lid'); plt.ylabel('y / L')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()