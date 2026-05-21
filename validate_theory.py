# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import j1, jn_zeros
from scipy.signal import argrelextrema

NM = 1e-9
MM = 1e-3


def make_single_slit_mask(N, size, width):
    x = np.linspace(-size/2, size/2, N)
    mask = np.abs(x[np.newaxis, :]) <= width/2
    return np.ones((N, N)) * mask


def make_circular_mask(N, size, radius):
    x = np.linspace(-size/2, size/2, N)
    y = np.linspace(-size/2, size/2, N)
    X, Y = np.meshgrid(x, y)
    return (X**2 + Y**2) <= radius**2


def make_double_slit_mask(N, size, slit_width, slit_separation):
    x = np.linspace(-size/2, size/2, N)
    c1 = -slit_separation/2
    c2 = slit_separation/2
    m1 = np.abs(x[np.newaxis,:] - c1) <= slit_width/2
    m2 = np.abs(x[np.newaxis,:] - c2) <= slit_width/2
    return np.ones((N,N)) * (m1 | m2)


def fraunhofer_diffraction(aperture, wavelength, b, aperture_size, N):
    xi = np.linspace(-aperture_size/2, aperture_size/2, N)
    dxi = xi[1] - xi[0]
    A_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))
    x_nat = np.fft.fftshift(np.fft.fftfreq(N, dxi)) * wavelength * b
    intensity = np.abs(A_fft)**2
    return intensity / np.max(intensity), x_nat


def theoretical_slit(x, slit_width, wavelength, b):
    alpha = np.pi * slit_width * x / (wavelength * b)
    return (np.sin(alpha + 1e-10) / (alpha + 1e-10))**2


def theoretical_airy(r, radius, wavelength, b):
    k = 2 * np.pi / wavelength
    rho = k * radius * r / b
    j = j1(rho + 1e-10)
    return (2 * j / (rho + 1e-10))**2


def theoretical_double_slit(x, slit_width, slit_sep, wavelength, b):
    alpha = np.pi * slit_width * x / (wavelength * b)
    beta  = np.pi * slit_sep * x / (wavelength * b)
    return (np.sin(alpha + 1e-10) / (alpha + 1e-10))**2 * np.cos(beta)**2


# =====================================================================
# TEST 1: Single slit - Fraunhofer
# =====================================================================
print("=" * 60)
print("TEST 1: Single slit - Fraunhofer (sinc^2)")
print("=" * 60)

wavelength = 632.8 * NM
b = 2.0
slit_width = 0.1e-3
aperture_size = 2e-3
screen_size_mm = 20.0
N = 1024

aperture = make_single_slit_mask(N, aperture_size, slit_width)
I_num, x_nat = fraunhofer_diffraction(aperture, wavelength, b, aperture_size, N)

mid = N // 2
half = int(screen_size_mm/2 / (x_nat[1] - x_nat[0]) * 1e-3)
half = min(half, N // 2)
x_mm = x_nat[mid-half:mid+half] * 1000
I_num_slice = I_num[mid, mid-half:mid+half]

x_m = x_mm * 1e-3
I_theory = theoretical_slit(x_m, slit_width, wavelength, b)

mse = np.mean((I_num_slice - I_theory)**2)
r2_1 = 1 - mse / np.var(I_theory)
print(f"  MSE  = {mse:.6e}")
print(f"  R^2  = {r2_1:.6f}")
print(f"  max|diff| = {np.max(np.abs(I_num_slice - I_theory)):.6f}")

n_vals = np.arange(1, 6)
x_min_theory = n_vals * wavelength * b / slit_width * 1000
print(f"  Minima (theory, mm):   {np.round(x_min_theory, 3)}")

min_idx = argrelextrema(I_num_slice, np.less, order=10)[0]
x_min_num = x_mm[min_idx]
x_min_num = x_min_num[x_min_num > 0.5]
print(f"  Minima (numerical, mm): {np.round(x_min_num[:6], 3)}")


# =====================================================================
# TEST 2: Circular aperture - Fraunhofer
# =====================================================================
print("\n" + "=" * 60)
print("TEST 2: Circular aperture - Fraunhofer (Airy disk)")
print("=" * 60)

wavelength = 550 * NM
b = 10.0
radius = 0.5e-3
aperture_size = 2e-3
N = 1024

aperture = make_circular_mask(N, aperture_size, radius)
I_num, x_nat = fraunhofer_diffraction(aperture, wavelength, b, aperture_size, N)
mid = N // 2
half = min(int(3e-3 / (x_nat[1] - x_nat[0])), N//2)
x_mm = x_nat[mid-half:mid+half] * 1000
I_num_slice = I_num[mid, mid-half:mid+half]

x_m = np.abs(x_mm * 1e-3)
I_theory = theoretical_airy(x_m, radius, wavelength, b)

mse = np.mean((I_num_slice - I_theory)**2)
r2_2 = 1 - mse / np.var(I_theory)
print(f"  MSE  = {mse:.6e}")
print(f"  R^2  = {r2_2:.6f}")
print(f"  max|diff| = {np.max(np.abs(I_num_slice - I_theory)):.6f}")

first_zero = jn_zeros(1, 1)[0]
theta_airy = first_zero * wavelength / (2 * np.pi * radius)
x_airy_theory = theta_airy * b * 1000
print(f"  First Airy minimum (theory, mm):     {x_airy_theory:.3f}")

min_idx = argrelextrema(I_num_slice, np.less, order=10)[0]
x_min_num = x_mm[min_idx]
x_min_num = x_min_num[x_min_num > 0.1]
print(f"  First Airy minimum (numerical, mm):  {x_min_num[0]:.3f}" if len(x_min_num) else "  minima not found")
print(f"  Relative error: {abs(x_min_num[0] - x_airy_theory)/x_airy_theory*100:.2f}%" if len(x_min_num) else "")


# =====================================================================
# TEST 3: Double slit - Fraunhofer
# =====================================================================
print("\n" + "=" * 60)
print("TEST 3: Double slit - Fraunhofer")
print("=" * 60)

wavelength = 650 * NM
b = 3.0
slit_width = 0.05e-3
slit_sep = 0.2e-3
aperture_size = 0.5e-3
N = 2048

aperture = make_double_slit_mask(N, aperture_size, slit_width, slit_sep)
I_num, x_nat = fraunhofer_diffraction(aperture, wavelength, b, aperture_size, N)
mid = N // 2
half = int(15e-3 / (x_nat[1] - x_nat[0]))
half = min(half, N//2)
x_mm = x_nat[mid-half:mid+half] * 1000
I_num_slice = I_num[mid, mid-half:mid+half]
I_theory = theoretical_double_slit(x_mm * 1e-3, slit_width, slit_sep, wavelength, b)

mse = np.mean((I_num_slice - I_theory)**2)
r2_3 = 1 - mse / np.var(I_theory)
print(f"  MSE  = {mse:.6e}")
print(f"  R^2  = {r2_3:.6f}")
print(f"  max|diff| = {np.max(np.abs(I_num_slice - I_theory)):.6f}")


# =====================================================================
# PLOTS
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# --- Single slit ---
wavelength1 = 632.8 * NM
b1 = 2.0
sw1 = 0.1e-3
ap_size1 = 2e-3
N1 = 1024
ap1 = make_single_slit_mask(N1, ap_size1, sw1)
I1, x1 = fraunhofer_diffraction(ap1, wavelength1, b1, ap_size1, N1)
mid1 = N1//2
half1 = int(10e-3 / (x1[1] - x1[0]))
half1 = min(half1, N1//2)
x_mm1 = x1[mid1-half1:mid1+half1] * 1000
I_num1 = I1[mid1, mid1-half1:mid1+half1]
I_th1 = theoretical_slit(x_mm1 * 1e-3, sw1, wavelength1, b1)
r2_1 = 1 - np.mean((I_num1-I_th1)**2)/np.var(I_th1)

ax = axes[0]
ax.plot(x_mm1, I_num1, 'b-', lw=1.5, label='Numerical')
ax.plot(x_mm1, I_th1, 'r--', lw=1.5, label='Theory (sinc^2)')
ax.set_title('Single slit\nlambda={}nm, b={}m, w={:.2f}mm'.format(632.8, b1, sw1*1000))
ax.set_xlabel('x (mm)'); ax.set_ylabel('I / Imax')
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlim(-10, 10)
ax.text(0.95, 0.95, 'R^2={:.4f}'.format(r2_1), transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

# --- Circular aperture ---
wavelength2 = 550 * NM
b2 = 10.0
r2_val = 0.5e-3
ap_size2 = 2e-3
N2 = 1024
ap2 = make_circular_mask(N2, ap_size2, r2_val)
I2, x2 = fraunhofer_diffraction(ap2, wavelength2, b2, ap_size2, N2)
mid2 = N2//2
half2 = int(3e-3 / (x2[1] - x2[0]))
half2 = min(half2, N2//2)
x_mm2 = x2[mid2-half2:mid2+half2] * 1000
I_num2 = I2[mid2, mid2-half2:mid2+half2]
I_th2 = theoretical_airy(np.abs(x_mm2 * 1e-3), r2_val, wavelength2, b2)
r2_2 = 1 - np.mean((I_num2-I_th2)**2)/np.var(I_th2)

ax2 = axes[1]
ax2.plot(x_mm2, I_num2, 'b-', lw=1.5, label='Numerical')
ax2.plot(x_mm2, I_th2, 'r--', lw=1.5, label='Theory (Airy)')
ax2.set_title('Circular aperture (Airy disk)\nlambda={}nm, b={}m, r={:.2f}mm'.format(550, b2, r2_val*1000))
ax2.set_xlabel('x (mm)'); ax2.set_ylabel('I / Imax')
ax2.legend(); ax2.grid(alpha=0.3)
ax2.text(0.95, 0.95, 'R^2={:.4f}'.format(r2_2), transform=ax2.transAxes, ha='right', va='top', fontsize=10,
         bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

# --- Double slit ---
wavelength3 = 650 * NM
b3 = 3.0
sw3 = 0.05e-3
ss3 = 0.2e-3
ap_size3 = 0.5e-3
N3 = 2048
ap3 = make_double_slit_mask(N3, ap_size3, sw3, ss3)
I3, x3 = fraunhofer_diffraction(ap3, wavelength3, b3, ap_size3, N3)
mid3 = N3//2
half3 = int(15e-3 / (x3[1] - x3[0]))
half3 = min(half3, N3//2)
x_mm3 = x3[mid3-half3:mid3+half3] * 1000
I_num3 = I3[mid3, mid3-half3:mid3+half3]
I_th3 = theoretical_double_slit(x_mm3 * 1e-3, sw3, ss3, wavelength3, b3)
r2_3 = 1 - np.mean((I_num3-I_th3)**2)/np.var(I_th3)

ax3 = axes[2]
ax3.plot(x_mm3, I_num3, 'b-', lw=1.5, label='Numerical')
ax3.plot(x_mm3, I_th3, 'r--', lw=1.5, label='Theory')
ax3.set_title('Double slit\nlambda={}nm, b={}m, w={:.3f}mm, d={:.2f}mm'.format(650, b3, sw3*1000, ss3*1000))
ax3.set_xlabel('x (mm)'); ax3.set_ylabel('I / Imax')
ax3.legend(); ax3.grid(alpha=0.3)
ax3.text(0.95, 0.95, 'R^2={:.4f}'.format(r2_3), transform=ax3.transAxes, ha='right', va='top', fontsize=10,
         bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('validation_results.png', dpi=150)
print("\nPlot saved to validation_results.png")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("SUMMARY:")
print("  Single slit:     R^2 = {:.6f}".format(r2_1))
print("  Circular:        R^2 = {:.6f}".format(r2_2))
print("  Double slit:     R^2 = {:.6f}".format(r2_3))
print("=" * 60)
