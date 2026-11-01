"""
Esercizio 2
============
Fit di due gaussiane sovrapposte a un fondo parabolico.

Il programma mantiene l'impostazione originale e aggiunge:
  1. una struttura piu ordinata e commentata;
  2. propagazione dell'errore del fondo mediante la matrice di covarianza;
  3. residui e pull;
  4. confronto dell'effetto del binning sulla distribuzione dei pull;
  5. fit Poissoniano di massima verosimiglianza (likelihood) degli stessi dati.

Nota didattica
--------------
Il curve_fit iniziale realizza un fit ai minimi quadrati pesati usando gli errori
contenuti nel file. La sezione finale usa invece la likelihood di Poisson,
appropriata quando n rappresenta conteggi per bin.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize


# =============================================================================
# 1. LETTURA DEI DATI
# =============================================================================

# Il file cornuto.txt deve essere nella stessa directory dello script.
# Dalla struttura originale:
#   colonna 0 -> x
#   colonna 1 -> contenuti/conteggi n
#   colonna 3 -> errore associato ai dati

dati = np.loadtxt("cornuto.txt")

x = dati[:, 0]
n = dati[:, 1]
err = dati[:, 3]


# =============================================================================
# 2. DEFINIZIONE DEL MODELLO
# =============================================================================

def signal(x, N, mu, sigma):
    """Gaussiana scritta nella stessa parametrizzazione del codice originale."""
    return N * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def background(x, c0, c1, c2):
    """Fondo parabolico."""
    return c0 + c1 * x + c2 * x**2


def modello(x, N, mu, sigma, N2, mu2, sigma2, c0, c1, c2):
    """Modello completo: prima gaussiana + seconda gaussiana + fondo."""
    seg1 = signal(x, N, mu, sigma)
    seg2 = signal(x, N2, mu2, sigma2)
    fondo = background(x, c0, c1, c2)
    return seg1 + seg2 + fondo


# =============================================================================
# 3. FIT AI MINIMI QUADRATI PESATI
# =============================================================================

# Valori iniziali mantenuti dal programma originale.
p0 = [605, 3, 1.3, 200, 7.5, 0.3, -8, 167, -16]

# curve_fit minimizza la somma dei residui quadratici pesati.
# absolute_sigma=True specifica che err contiene errori assoluti, non semplici pesi.
popt, pcov = curve_fit(
    modello,
    x,
    n,
    p0=p0,
    sigma=err,
    absolute_sigma=True,
    maxfev=50000,
)

# Errori standard dei parametri: radice della diagonale della covarianza.
perr = np.sqrt(np.diag(pcov))

print("\n============================================================")
print("FIT AI MINIMI QUADRATI")
print("============================================================")

nomi_parametri = ["N1", "mu1", "sigma1", "N2", "mu2", "sigma2", "c0", "c1", "c2"]
for nome, valore, errore in zip(nomi_parametri, popt, perr):
    print(f"{nome:7s} = {valore:12.6g} +/- {errore:.6g}")


# =============================================================================
# 4. GRAFICO DEL FIT COMPLETO
# =============================================================================

xx = np.linspace(0, 10, 1000)

plt.figure(figsize=(8, 6))
plt.errorbar(x, n, yerr=err, fmt="o", ms=3, label="Dati")
plt.plot(xx, modello(xx, *popt), label="Gaussiane + fondo parabolico")

plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")

plt.text(
    0.23,
    0.46,
    rf"$\mu_1 = {popt[1]:.3f} \pm {perr[1]:.3f}$" "\n"
    rf"$\sigma_1 = {popt[2]:.3f} \pm {perr[2]:.3f}$" "\n"
    rf"$\mu_2 = {popt[4]:.3f} \pm {perr[4]:.3f}$" "\n"
    rf"$\sigma_2 = {popt[5]:.3f} \pm {perr[5]:.3f}$",
    transform=plt.gca().transAxes,
    va="top",
)

plt.legend()
plt.tight_layout()
plt.savefig("soluzione_02cc.pdf")
plt.show()


# =============================================================================
# 5. SOTTRAZIONE DEL FONDO
# =============================================================================

# Valore del fondo determinato dal fit in corrispondenza dei punti x.
y_bg = background(x, popt[-3], popt[-2], popt[-1])

# Dati dopo la sottrazione del valore centrale del fondo.
y_clean = n - y_bg

# ATTENZIONE:
# NON sarebbe corretto assegnare ai dati puliti sqrt(y_clean), come se fossero
# stati acquisiti direttamente in assenza di fondo. La sottrazione elimina la
# stima del valore medio del fondo, non le fluttuazioni gia presenti nei dati.


# =============================================================================
# 6. PROPAGAZIONE DELL'ERRORE DEL FIT DEL FONDO
# =============================================================================

# I parametri del fondo sono gli ultimi tre del fit: c0, c1, c2.
# Estraggo quindi la relativa sottomatrice 3x3 dalla covarianza completa.
cov_bg = pcov[-3:, -3:]

# Per B(x)=c0+c1*x+c2*x^2, il Jacobiano rispetto a (c0,c1,c2) e:
#       J_i = (1, x_i, x_i^2)
# Una riga per ciascun punto.
J = np.column_stack((np.ones_like(x), x, x**2))

# Propagazione completa:
#       Var[B(x_i)] = J_i C_bg J_i^T
# np.einsum valuta questa espressione per tutti i punti simultaneamente.
# Sono inclusi automaticamente anche i termini fuori diagonale della matrice
# di covarianza, cioe le correlazioni c0-c1, c0-c2 e c1-c2.
var_bg = np.einsum("ij,jk,ik->i", J, cov_bg, J)

# Piccoli valori negativi possono comparire esclusivamente per arrotondamenti
# numerici. Li riportiamo a zero prima della radice quadrata.
var_bg = np.clip(var_bg, 0.0, None)
err_bg = np.sqrt(var_bg)

# Approssimazione usata in questa esercitazione:
#       sigma_clean^2 = sigma_dati^2 + sigma_fondo^2
#
# Poiche il fondo e ottenuto dagli stessi dati, rigorosamente esisterebbe anche
# il termine -2 Cov(y_i, B_i). Qui lo trascuriamo esplicitamente per mantenere
# l'esercizio al livello didattico desiderato.
ey_clean = np.sqrt(err**2 + err_bg**2)


# =============================================================================
# 7. DATI DOPO LA SOTTRAZIONE DEL FONDO
# =============================================================================

plt.figure(figsize=(8, 6))
plt.errorbar(x, y_clean, yerr=ey_clean, fmt="*", ms=5, label="Segnali - fondo sottratto")
plt.plot(x, y_bg, label="Fondo")
plt.plot(x, signal(x, popt[0], popt[1], popt[2]), label="Segnale 1")
plt.plot(x, signal(x, popt[3], popt[4], popt[5]), label="Segnale 2")
plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2 - sottrazione del fondo")
plt.legend()
plt.tight_layout()
plt.show()


# Confronto tra le diverse componenti dell'errore.
plt.figure(figsize=(8, 5))
plt.plot(x, err, "o-", label="Errore dati")
plt.plot(x, err_bg, "o-", label="Errore fit del fondo")
plt.plot(x, ey_clean, "o-", label="Errore totale dopo sottrazione")
plt.xlabel("x")
plt.ylabel("Errore")
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 8. RESIDUI DEL FIT SUI DATI ORIGINALI
# =============================================================================

y_fit = modello(x, *popt)
residuals = n - y_fit

plt.figure(figsize=(8, 4))
plt.errorbar(x, residuals, yerr=err, fmt="o", markersize=4, capsize=2)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("x")
plt.ylabel("Residui")
plt.tight_layout()
plt.show()


# =============================================================================
# 9. PULL
# =============================================================================

# Il pull e il residuo espresso in unita dell'errore del dato.
pull = residuals / err

plt.figure(figsize=(8, 4))
plt.axhline(0, color="red", linestyle="--")
plt.plot(x, pull, "o", markersize=4)
plt.xlabel("x")
plt.ylabel("Pull")
plt.tight_layout()
plt.show()

# Eliminiamo eventuali valori non finiti prima di costruire l'istogramma.
pull = pull[np.isfinite(pull)]


# =============================================================================
# 10. DISTRIBUZIONE DEI PULL E FIT GAUSSIANO
# =============================================================================

def gauss_pull(x, A, mu, sigma):
    """Gaussiana utilizzata soltanto per descrivere l'istogramma dei pull."""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


plt.figure(figsize=(7, 5))
n_pull, bins_pull, _ = plt.hist(
    pull,
    bins=12,
    histtype="step",
    linewidth=1.5,
    color="black",
    label="Pull",
)

x_pull = 0.5 * (bins_pull[1:] + bins_pull[:-1])

# Per il fit eliminiamo i bin vuoti, per i quali sqrt(N)=0.
mask = n_pull > 0
err_pull = np.sqrt(n_pull[mask])

p0_pull = [np.max(n_pull), np.mean(pull), np.std(pull)]

popt_pull, pcov_pull = curve_fit(
    gauss_pull,
    x_pull[mask],
    n_pull[mask],
    p0=p0_pull,
    sigma=err_pull,
    absolute_sigma=True,
)

perr_pull = np.sqrt(np.diag(pcov_pull))

xx_pull = np.linspace(bins_pull[0], bins_pull[-1], 500)
plt.plot(xx_pull, gauss_pull(xx_pull, *popt_pull), "r-", label="Fit gaussiano")

plt.text(
    0.04,
    0.95,
    rf"$\mu = {popt_pull[1]:.3f} \pm {perr_pull[1]:.3f}$" "\n"
    rf"$\sigma = {popt_pull[2]:.3f} \pm {perr_pull[2]:.3f}$",
    transform=plt.gca().transAxes,
    va="top",
)

plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Pull")
plt.ylabel("Conteggi")
plt.legend()
plt.tight_layout()
plt.savefig("pull_residui.pdf")
plt.show()


# =============================================================================
# 11. EFFETTO DEL BINNING SUL FIT DELLA DISTRIBUZIONE DEI PULL
# =============================================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for iax, nbins in enumerate([15, 10]):
    n_pull, edges_pull, _ = ax[iax].hist(
        pull,
        bins=nbins,
        histtype="step",
        linewidth=1.8,
        color="black",
        label="Pull",
    )

    x_pull = 0.5 * (edges_pull[:-1] + edges_pull[1:])
    err_pull = np.sqrt(n_pull)
    mask = n_pull > 0

    popt_pull, pcov_pull = curve_fit(
        gauss_pull,
        x_pull[mask],
        n_pull[mask],
        sigma=err_pull[mask],
        absolute_sigma=True,
        p0=[np.max(n_pull), 0.0, 1.0],
    )

    perr_pull = np.sqrt(np.diag(pcov_pull))
    xx_pull = np.linspace(edges_pull[0], edges_pull[-1], 500)

    ax[iax].plot(
        xx_pull,
        gauss_pull(xx_pull, *popt_pull),
        "r-",
        lw=2,
        label="Fit gaussiano",
    )

    ax[iax].text(
        0.05,
        0.95,
        rf"$\mu = {popt_pull[1]:.3f} \pm {perr_pull[1]:.3f}$" "\n"
        rf"$\sigma = {popt_pull[2]:.3f} \pm {perr_pull[2]:.3f}$",
        transform=ax[iax].transAxes,
        va="top",
    )

    ax[iax].set_title(f"Pull distribution - {nbins} bin")
    ax[iax].set_xlabel("Pull")
    ax[iax].set_ylabel("Conteggi")
    ax[iax].legend()

plt.tight_layout()
plt.savefig("pull_binning_comparison.pdf")
plt.show()


# =============================================================================
# 12. FIT DI MASSIMA VEROSIMIGLIANZA: LIKELIHOOD DI POISSON
# =============================================================================
#
# Questa sezione aggiunge un secondo modo di stimare GLI STESSI parametri.
# Non usiamo gli errori err per costruire la likelihood: se n_i sono conteggi,
# assumiamo
#
#                  n_i ~ Poisson(lambda_i)
#
# con
#                  lambda_i = modello(x_i, parametri).
#
# La likelihood totale e il prodotto delle probabilita di Poisson dei singoli
# bin. Numericamente e molto piu conveniente minimizzare -log L.
#
# Trascurando il termine log(n_i!), che non dipende dai parametri:
#
#    -log L = SUM_i [ lambda_i - n_i log(lambda_i) ] + costante.
#
# Il termine costante puo essere omesso nella minimizzazione senza modificare
# la posizione del massimo della likelihood.


def poisson_nll(par, x, n):
    """Negative log-likelihood di Poisson, a meno di una costante."""
    lambda_i = modello(x, *par)

    # Una media poissoniana deve essere strettamente positiva.
    # Se una combinazione di parametri produce lambda <= 0, restituiamo una
    # penalita molto grande per impedire al minimizzatore di accettarla.
    if np.any(lambda_i <= 0) or np.any(~np.isfinite(lambda_i)):
        return 1.0e100

    return np.sum(lambda_i - n * np.log(lambda_i))


# Partiamo dalla soluzione dei minimi quadrati: in questo modo il confronto e
# pulito e il minimizzatore parte gia da una regione ragionevole dei parametri.
p0_like = popt.copy()

# Impediamo soltanto alle larghezze gaussiane di diventare nulle/negative.
# Gli altri parametri restano liberi, mantenendo il piu possibile il modello
# originale. N e N2 vengono qui vincolati positivi per una previsione fisica
# di conteggi; il fondo puo invece avere coefficienti di qualunque segno,
# purche il modello TOTALE resti positivo nei punti osservati.
eps = 1.0e-8
bounds_like = [
    (0.0, None),       # N1
    (None, None),      # mu1
    (eps, None),       # sigma1
    (0.0, None),       # N2
    (None, None),      # mu2
    (eps, None),       # sigma2
    (None, None),      # c0
    (None, None),      # c1
    (None, None),      # c2
]

result_like = minimize(
    poisson_nll,
    p0_like,
    args=(x, n),
    method="L-BFGS-B",
    bounds=bounds_like,
    options={"maxiter": 50000},
)

popt_like = result_like.x

print("\n============================================================")
print("FIT DI MASSIMA VEROSIMIGLIANZA - POISSON")
print("============================================================")
print("Convergenza:", result_like.success)
print("Messaggio   :", result_like.message)
print(f"-log L      : {result_like.fun:.6g}")

for nome, valore in zip(nomi_parametri, popt_like):
    print(f"{nome:7s} = {valore:12.6g}")


# =============================================================================
# 13. CONFRONTO VISIVO: MINIMI QUADRATI E LIKELIHOOD
# =============================================================================

plt.figure(figsize=(8, 6))
plt.errorbar(x, n, yerr=err, fmt="o", ms=3, label="Dati")
plt.plot(xx, modello(xx, *popt), lw=2, label="Minimi quadrati")
plt.plot(xx, modello(xx, *popt_like), "--", lw=2, label="Likelihood di Poisson")
plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Confronto tra fit ai minimi quadrati e likelihood")
plt.legend()
plt.tight_layout()
plt.savefig("confronto_chi2_likelihood.pdf")
plt.show()


# =============================================================================
# 14. POISSON DEVIANCE: ANALOGO DEL CHI^2 PER IL FIT DI LIKELIHOOD
# =============================================================================
#
# Per conteggi poissoniani e utile calcolare anche la deviance:
#
# D = 2 SUM_i [ lambda_i - n_i + n_i log(n_i/lambda_i) ]
#
# con la convenzione n_i log(n_i/lambda_i)=0 quando n_i=0.
# Per grandi conteggi D si comporta, sotto opportune condizioni, in maniera
# simile a un chi-quadro e permette un ponte didattico tra i due approcci.

lambda_like = modello(x, *popt_like)

term_log = np.zeros_like(n, dtype=float)
mask_positive = n > 0
term_log[mask_positive] = n[mask_positive] * np.log(
    n[mask_positive] / lambda_like[mask_positive]
)

poisson_deviance = 2.0 * np.sum(lambda_like - n + term_log)
ndof_like = len(n) - len(popt_like)

print("\nPoisson deviance =", poisson_deviance)
print("gradi di liberta =", ndof_like)
if ndof_like > 0:
    print("deviance / ndof  =", poisson_deviance / ndof_like)
