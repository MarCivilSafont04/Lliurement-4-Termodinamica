# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:17:49 2026

@author: Usuari
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import lambertw

# Dades experimentals
T = np.array([273.16, 298.16, 323.16, 348.16, 373.16,
              423.16, 473.16, 573.16, 673.16,
              773.16, 873.16, 973.16])

B2 = np.array([-154.74, -130.21, -110.62, -95.04, -82.13,
               -62.10, -46.74, -25.06, -10.77,
               -0.13, 7.95, 14.22])

NA = 6.02214076e23

# Model teòric
def B2_model(T, r0, rho, eps_kb):
    return (2*np.pi/3) * NA * r0**3 * (1 - (rho**3 - 1) * (np.exp(eps_kb/T) - 1))

# Aproximació altes temperatures
def B2_aprox(T, r0, rho, eps_kb):
    return (2*np.pi/3) * NA * r0**3 * (1 - (rho**3 - 1) * (eps_kb/T))

# Ajust
p0 = [3.5e-8, 1.7, 180]
params, cov = curve_fit(B2_model, T, B2, p0=p0)

r0, rho, eps_kb = params

r0_A = r0 * 1e8

# límit T → ∞
B2_inf = (2*np.pi/3) * NA * r0**3

# Temperatura de Boyle
TB = eps_kb / np.log(rho**3 / (rho**3 - 1))

# Temperatura d’inversió màxima
Tinv = eps_kb / (lambertw(np.exp(1) * rho**3 / (rho**3 - 1)).real - 1)

print(f"r0 = {r0_A:.3f} Å")
print(f"rho = {rho:.3f}")
print(f"epsilon/kB = {eps_kb:.2f} K")
print(f"B2(T→∞) = {B2_inf:.2f} cm^3/mol")
print(f"TB = {TB:.2f} K")
print(f"Tinv = {Tinv:.2f} K")

T_plot = np.linspace(250, 2000, 400)

plt.figure(figsize=(8,5))

plt.scatter(T, B2, color='teal', edgecolor='black', s=50, label='Dades experimentals', zorder=5)

plt.plot(T_plot, B2_model(T_plot, r0, rho, eps_kb), color='darkorange', linewidth=2.5, label='Ajust no lineal')

plt.plot(T_plot, B2_aprox(T_plot, r0, rho, eps_kb), color='turquoise', linewidth=2, linestyle='-', label='Aproximació altes T')

#B2(T → ∞)
plt.axhline(B2_inf, linestyle='--', label='$B_2(T → ∞)$')

# línia vertical de la temperatura de Boyle
plt.axvline(TB, color='red', linestyle='--', linewidth=1.5, label='$T_B$')

# Línia vertical de la temperatura d'inversió màxima
plt.axvline(Tinv, color='purple', linestyle='--', linewidth=1.5, label='$T_{inv}$')

# Temperatura d’inversió màxima 
plt.scatter(Tinv, B2_model(Tinv, r0, rho, eps_kb),  color='teal', edgecolor='black', s=50)


plt.xlabel('Temperatura (K)', fontsize=15)
plt.ylabel('$B_2$ (cm³/mol)', fontsize=15)
plt.grid(alpha=0.4)
plt.legend(fontsize=9)
plt.show()