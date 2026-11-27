import numpy as np
import matplotlib.pyplot as plt

####################################################
#
# Cross Jet Flow - LBM (2D)
#
####################################################

####################################################
#
# Channel and Jet Geometry
#

L = 5
H = 1
W = 0.1

####################################################
#
# Mesh Parameters
#

n = 10
bc = 9

Nx = 2**n

Nc = Nx * W // L
Bc = int(bc * Nc)
Ac = int(Nx - Nc - Bc)
C = int(Bc + Nc)

Ny = H * Nx // L

print(Bc)

####################################################
#
# LBM - D2Q9 
#

c_s = 1 / np.sqrt(3)

Ma = 0.1
U = Ma * c_s;

cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

####################################################
#
# Flow Parameters 
#

Re_Jet = 8000
nu = U * (Nx - 1) / Re_Jet
tau = (3 * nu) + 0.5
#Re_Channel = Re_Jet / 3
#U_Channel = Re_Channel * nu / Ny
U_Channel = U / 3
print(U / U_Channel)

####################################################
#
# Iteration Parameters
#

maxIter = 50001
Iter_Tol = 1e-7
err = 1

####################################################
#
# Initialization
#

f = np.zeros((9, Ny, Nx))
rho = np.ones((Ny, Nx))
ux = np.zeros((Ny, Nx))
uy = np.zeros((Ny, Nx))

ux_old = ux.copy()
    
for i in range(9):
    f[i, :, :] = w[i] * rho
    
####################################################
#
# Main Loop
#

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
        
    # Top
    
    #Top_Wall
    #f[4, -1, :] = f[2, -1, :]
    #f[7, -1, :] = f[5, -1, :]
    #f[8, -1, :] = f[6, -1, :]
    
    #Top_Outlet
    f[4, -1, :] = f[4, -2, :]
    f[7, -1, :] = f[7, -2, :]
    f[8, -1, :] = f[8, -2, :]
    
    # Left
    rho_left = np.sum(f[:, :, 1], axis=0)
    f[1, :, 0] = f[3, :, 0] + 6 * w[1] * rho_left * U_Channel
    f[5, :, 0] = f[7, :, 0] + 6 * w[5] * rho_left * U_Channel
    f[8, :, 0] = f[6, :, 0] + 6 * w[8] * rho_left * U_Channel
    
    # Right
    f[3, :, -1] = f[3, :, -2]
    f[6, :, -1] = f[6, :, -2]
    f[7, :, -1] = f[7, :, -2]

    # Bottom
    # Bounce Back (BC)
    f[2, 0, :Bc] = f[4, 0, :Bc]
    f[5, 0, :Bc] = f[7, 0, :Bc]
    f[6, 0, :Bc] = f[8, 0, :Bc]
    
    # Jet Flow
    rho_bottom = np.sum(f[:, 1, Bc:C], axis=0)    
    f[2, 0, Bc:C] = f[4, 0, Bc:C] + 6 * w[2] * rho_bottom * U
    f[5, 0, Bc:C] = f[7, 0, Bc:C] + 6 * w[5] * rho_bottom * U
    f[6, 0, Bc:C] = f[8, 0, Bc:C] + 6 * w[8] * rho_bottom * U
    
    # Bounce Back (AC)
    f[2, 0, C:] = f[4, 0, C:]
    f[5, 0, C:] = f[7, 0, C:]
    f[6, 0, C:] = f[8, 0, C:]

    if it % 200 == 0 and it > 0:
        print(f"Iter {it}, max ux = {np.max(ux):.4f}")
        err = np.max(np.abs(ux - ux_old)) / U
        ux_old = ux.copy()
    
    if err < Iter_Tol:

        break

####################################################
#
# Finish 
#
        
rho = np.sum(f, axis=0)
ux = np.sum(f * cx[:, None, None], axis=0) / rho
uy = np.sum(f * cy[:, None, None], axis=0) / rho
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))

####################################################
#
# Plot Settings
#

plt.figure(figsize=(10, 4))
plt.imshow(np.sqrt(ux**2 + uy**2)/U, origin='lower', cmap='viridis')
plt.streamplot(X, Y, ux/U, uy/U, density=1.2, color='white', linewidth=0.6, arrowsize=0.8)
plt.colorbar(label='|U|')
plt.title('Cross Jet Flow, Re=1000')
plt.xlim(0, Nx-1)
plt.ylim(0, Ny-1)

plt.tight_layout()
plt.show()