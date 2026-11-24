########################################
#
# Shan Chen - Gas Bubble Model (D2Q9)
#
import numpy as np
import matplotlib.pyplot as plt

########################################
#
# Mesh Parameters
#
n = 8
Nx = 2**n
Ny = 2**n

radius = 20

########################################
#
# Iterations Parameters
#
maxIter = 30000
it_output = 1000
errror = 1
E_tol = 1e-5

########################################
#
# LBM - D2Q9 Parameters
#
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
tau = 1.0
RT = 1/3

########################################
#
# Shan Chen Parameters
#
rho_w0 = 1.95
rho_g0 = 0.15

G = -5.0

########################################
#
# Initialization
#
f = np.zeros((9, Ny, Nx))

X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
r = np.sqrt((X - Nx / 2)**2 + (Y - Ny / 2)**2)
gas_frac = 0.5 * (1 - np.tanh((r - radius) / 4))

# Gas bubble: inside = gas, outside = liquid
rho = rho_w0 * (1 - gas_frac) + rho_g0 * gas_frac

ux = np.zeros((Ny, Nx))
uy = np.zeros((Ny, Nx))

for i in range(9):
    f[i, :, :] = w[i] * rho

########################################
#
# History for plots
#
Dp_list = []
rho_w_list = []
rho_g_list = []
it_list = []

########################################
#
# Main Loop
#
for it in range(maxIter):

    # --- Macroscopic variables ---
    rho = np.sum(f, axis=0)

    # --- Shan-Chen Force ---
    psi = 1 - np.exp(-rho)

    Fx = np.zeros((Ny, Nx))
    Fy = np.zeros((Ny, Nx))
    for i in range(9):
        psi_shift = np.roll(np.roll(psi, shift=cy[i], axis=0), shift=cx[i], axis=1)
        Fx += w[i] * cx[i] * psi_shift
        Fy += w[i] * cy[i] * psi_shift

    Fx = -G * psi * Fx
    Fy = -G * psi * Fy

    # --- Velocity with 1/2 force correction ---
    ux = np.sum(f * cx[:, None, None], axis=0) / rho + Fx / (2.0 * rho)
    uy = np.sum(f * cy[:, None, None], axis=0) / rho + Fy / (2.0 * rho)

    # --- rho_w, rho_g, Dp ---
    # Liquid = outside bubble -> corner box 10x10
    rho_w = np.mean(rho[0:10, 0:10])
    # Gas = inside bubble -> center box 10x10
    rho_g = np.mean(rho[Ny//2-5:Ny//2+5, Nx//2-5:Nx//2+5])

    psi_w = 1 - np.exp(-rho_w)
    psi_g = 1 - np.exp(-rho_g)

    p_w = rho_w*RT + G*psi_w**2/6
    p_g = rho_g*RT + G*psi_g**2/6

    Dp = p_w - p_g

    # --- Save history ---
    if it % it_output == 0:
        Dp_list.append(Dp)
        rho_w_list.append(rho_w)
        rho_g_list.append(rho_g)
        it_list.append(it)
        print(f"Iter {it}, rho_w = {rho_w:.4f}, rho_g = {rho_g:.4f}, Dp = {Dp:.5f}")

    # --- Equilibrium ---
    cu = cx[:, None, None] * ux[None, :, :] + cy[:, None, None] * uy[None, :, :]
    u2 = ux**2 + uy**2
    feq = w[:, None, None] * rho[None, :, :] * (1 + 3*cu + 4.5*cu**2 - 1.5*u2[None, :, :])

    # --- Guo Forcing Term (same as fpop in C++) ---
    cu_F = cx[:, None, None] * Fx[None, :, :] + cy[:, None, None] * Fy[None, :, :]
    # Vectorized form of: w*(1-0.5/tau)*[ (3*(cx-ux)+9*cx*(cx*ux+cy*uy))*Fx + (3*(cy-uy)+9*cy*(...))*Fy ]
    Fpop = w[:, None, None] * (1 - 0.5/tau) * (
        (3*(cx[:, None, None] - ux[None, :, :]) + 9*cx[:, None, None]*cu) * Fx[None, :, :] +
        (3*(cy[:, None, None] - uy[None, :, :]) + 9*cy[:, None, None]*cu) * Fy[None, :, :]
    )

    # --- Collision ---
    f = f - (f - feq) / tau + Fpop

    # --- Streaming ---
    for i in range(9):
        f[i, :, :] = np.roll(f[i, :, :], shift=(cy[i], cx[i]), axis=(0, 1))

########################################
#
# Aftermatch
#
rho = np.sum(f, axis=0)
ux = np.sum(f * cx[:, None, None], axis=0) / rho
uy = np.sum(f * cy[:, None, None], axis=0) / rho

# Final values for titles
rho_w_final = np.mean(rho[0:10, 0:10])
rho_g_final = np.mean(rho[Ny//2-5:Ny//2+5, Nx//2-5:Nx//2+5])
psi_w_f = 1 - np.exp(-rho_w_final)
psi_g_f = 1 - np.exp(-rho_g_final)
p_w_f = rho_w_final*RT + G*psi_w_f**2/6
p_g_f = rho_g_final*RT + G*psi_g_f**2/6
Dp_final = p_w_f - p_g_f

print(f"Final: rho_w = {rho_w_final:.4f}, rho_g = {rho_g_final:.4f}, Dp = {Dp_final:.5f}")
print(f"Check Laplace: sigma_est = Dp * R = {Dp_final * radius:.4f}")

########################################
#
# Plots
#
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
im = plt.imshow(rho, origin='lower', cmap='jet')
plt.colorbar(im, label='Density')
plt.title(f'Density Field\nDp = {Dp_final:.5f}  |  rho_w={rho_w_final:.3f}, rho_g={rho_g_final:.3f}')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 3, 2)
plt.plot(it_list, Dp_list, 'o-', color='black', linewidth=1.5)
plt.axhline(Dp_final, color='red', linestyle='--', label=f'Final Dp = {Dp_final:.5f}')
plt.xlabel('Iteration')
plt.ylabel('Dp = p_w - p_g')
plt.title(f'Laplace Pressure\nFinal Dp = {Dp_final:.5f}')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 3)
center_y = Ny // 2
plt.plot(rho[center_y, :], color='blue', linewidth=1.5)
plt.axhline(rho_w_final, color='red', linestyle='--', label=f'rho_w = {rho_w_final:.3f}')
plt.axhline(rho_g_final, color='green', linestyle='--', label=f'rho_g = {rho_g_final:.3f}')
plt.xlabel('x (through center y=64)')
plt.ylabel('rho')
plt.title(f'Centerline Profile\nDp = {Dp_final:.5f}')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()