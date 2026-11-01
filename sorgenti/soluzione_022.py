import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dati = np.loadtxt("datigpa.txt")

n, bordi = np.histogram(dati, bins=80)
x = (bordi[:-1] + bordi[1:]) / 2
err = np.sqrt(np.maximum(n, 1))

xref = 2.5

def modello(x, N, mu, sigma, c0, c1, c2):
    seg = N * np.exp(-0.5*((x-mu)/sigma)**2)
    fondo = c0 + c1*(x-xref) + c2*(x-xref)**2
    return seg + fondo

# N è l'altezza del picco, quindi conviene partire dal massimo dell'istogramma
p0 = [n.max(), 3, 1.5, 20, 0, 5]

popt, pcov = curve_fit(
    modello, x, n,
    p0=p0,
    sigma=err,
    absolute_sigma=True,
    maxfev=50000
)

print(popt)

perr = np.sqrt(np.diag(pcov))

xx = np.linspace(bordi[0], bordi[-1], 1000)

plt.errorbar(x, n, yerr=err, fmt="o", ms=3, label="Dati")
plt.plot(xx, modello(xx, *popt),
         label="Gaussiana + fondo parabolico")

plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")

plt.text(
    .03, .96,
    f"N = {popt[0]:.1f} +/- {perr[0]:.1f}\n"
    f"mu = {popt[1]:.3f} +/- {perr[1]:.3f}\n"
    f"sigma = {popt[2]:.3f} +/- {perr[2]:.3f}\n"
    f"c2 = {popt[5]:.3f} +/- {perr[5]:.3f}",
    transform=plt.gca().transAxes,
    va="top"
)

plt.legend()
plt.tight_layout()
plt.show()
