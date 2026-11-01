import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.special import gammaln
dati=np.loadtxt("cornuto.txt")
x=dati[:,0]
n=dati[:,1]
err=dati[:,3]

def modello(x,N,mu,sigma,N2,mu2,sigma2,c0,c1,c2):
    seg=N*np.exp(-0.5*((x-mu)/sigma)**2)
    seg2=N2*np.exp(-0.5*((x-mu2)/sigma2)**2)
    fondo=c0+c1*x+c2*x**2
    return seg+seg2+fondo
def background(x,c0,c1,c2):
    fondo=c0+c1*x+c2*x**2
    return fondo

def signal(x,N,mu,sigma):
    seg=N*np.exp(-0.5*((x-mu)/sigma)**2)
    return seg

#p0=[605,3,1.23,-8,167,-16]
#p0=[1,1,1,1,1,1]
p0=[605,3,1.3,200,7.5,0.3,-8,167,-16]

popt,pcov=curve_fit(modello,x,n,p0=p0,sigma=err,absolute_sigma=True,maxfev=50000)
print(popt)
perr=np.sqrt(np.diag(pcov))

# ============================================================
# CONFRONTO TRA FIT CHI^2 E FIT DI MASSIMA LIKELIHOOD POISSONIANA
# ============================================================
# Il fit precedente minimizza il chi^2:
#     chi2 = sum_i [(n_i - mu_i) / err_i]^2
# Per conteggi poissoniani il fit di likelihood minimizza invece
# la negative log-likelihood di Poisson. Il termine log(n_i!) non
# cambia la posizione del minimo, ma viene mantenuto per completezza.

def nll_poisson(par, x, n):
    mu_attesi = modello(x, *par)

    # La media di Poisson deve essere strettamente positiva.
    if np.any(mu_attesi <= 0) or not np.all(np.isfinite(mu_attesi)):
        return 1.0e100

    return np.sum(mu_attesi - n*np.log(mu_attesi) + gammaln(n + 1.0))

# Ampiezze e larghezze gaussiane positive. I coefficienti del fondo
# restano liberi, ma il controllo nella NLL impone un modello totale positivo.
bounds_likelihood = [
    (0.0, None),       # N1
    (None, None),      # mu1
    (1.0e-6, None),    # sigma1
    (0.0, None),       # N2
    (None, None),      # mu2
    (1.0e-6, None),    # sigma2
    (None, None),      # c0
    (None, None),      # c1
    (None, None)       # c2
]

ris_likelihood = minimize(
    nll_poisson,
    x0=popt,                 # parto dalla soluzione del fit chi^2
    args=(x, n),
    method="L-BFGS-B",
    bounds=bounds_likelihood,
    options={"maxiter": 100000, "ftol": 1.0e-12, "gtol": 1.0e-8}
)

if not ris_likelihood.success:
    print("ATTENZIONE: il fit di likelihood non e' convergente:")
    print(ris_likelihood.message)

popt_likelihood = ris_likelihood.x

# Stima della covarianza dalla matrice Hessiana inversa restituita
# da L-BFGS-B. E' un'approssimazione locale della covarianza.
try:
    pcov_likelihood = np.asarray(ris_likelihood.hess_inv.todense())
    perr_likelihood = np.sqrt(np.clip(np.diag(pcov_likelihood), 0, None))
except Exception:
    pcov_likelihood = np.full((len(popt_likelihood), len(popt_likelihood)), np.nan)
    perr_likelihood = np.full(len(popt_likelihood), np.nan)

# Indicatori di confronto calcolati sui due modelli
mu_chi2 = modello(x, *popt)
mu_likelihood = modello(x, *popt_likelihood)

chi2_fit = np.sum(((n - mu_chi2)/err)**2)
ndof_chi2 = len(n) - len(popt)
chi2_ridotto = chi2_fit/ndof_chi2

# Deviance poissoniana: 2 sum[n log(n/mu) - (n-mu)].
# Per n=0 si usa il limite n log(n/mu) = 0.
termine_log = np.zeros_like(n, dtype=float)
mask_positivi = n > 0
termine_log[mask_positivi] = n[mask_positivi]*np.log(
    n[mask_positivi]/mu_likelihood[mask_positivi]
)
deviance_poisson = 2.0*np.sum(
    termine_log - (n - mu_likelihood)
)
ndof_likelihood = len(n) - len(popt_likelihood)
deviance_ridotta = deviance_poisson/ndof_likelihood

nomi_parametri = ["N1", "mu1", "sigma1", "N2", "mu2", "sigma2", "c0", "c1", "c2"]

print("\nCONFRONTO TRA FIT CHI^2 E LIKELIHOOD POISSONIANA")
print("-"*86)
print(f"{'Parametro':<10} {'chi^2':>14} {'err chi^2':>14} "
      f"{'likelihood':>14} {'err likelihood':>16}")
for nome, v_chi2, e_chi2, v_lik, e_lik in zip(
        nomi_parametri, popt, perr, popt_likelihood, perr_likelihood):
    print(f"{nome:<10} {v_chi2:14.6g} {e_chi2:14.6g} "
          f"{v_lik:14.6g} {e_lik:16.6g}")
print("-"*86)
print(f"chi^2 / ndof       = {chi2_fit:.3f} / {ndof_chi2} = {chi2_ridotto:.3f}")
print(f"Deviance / ndof    = {deviance_poisson:.3f} / {ndof_likelihood} "
      f"= {deviance_ridotta:.3f}")
print(f"-log L al minimo   = {ris_likelihood.fun:.3f}")

# Salvataggio tabella numerica
confronto = np.column_stack((popt, perr, popt_likelihood, perr_likelihood))
np.savetxt(
    "confronto_chi2_likelihood.txt",
    confronto,
    header="chi2  err_chi2  likelihood  err_likelihood",
    fmt="%.10e"
)

# Figura dedicata al confronto
xx=np.linspace(0,10,1000)
plt.figure(figsize=(8, 5))
plt.errorbar(x, n, yerr=err, fmt="o", ms=3, color="black", label="Dati")
plt.plot(xx, modello(xx, *popt), lw=2, label=rf"Fit $\chi^2$: $\chi^2/\nu={chi2_ridotto:.2f}$")
plt.plot(xx, modello(xx, *popt_likelihood), "--", lw=2,
         label=rf"Likelihood Poisson: $D/\nu={deviance_ridotta:.2f}$")
plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title(r"Confronto tra fit $\chi^2$ e likelihood poissoniana")
plt.legend()
plt.tight_layout()
plt.savefig("confronto_chi2_likelihood.pdf")
plt.show()

xx=np.linspace(0,10,1000)
plt.errorbar(x,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Gaussiane + fondo parabolico")
plt.xlabel("x") 
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")
plt.text(
    .23, .46,
    f"mu = {popt[1]:.3f} +/- {perr[1]:.3f}\n"
    f"sigma = {popt[2]:.3f} +/- {perr[2]:.3f}\n"
    f"mu2 = {popt[4]:.3f} +/- {perr[4]:.3f}\n"
    f"sigma2 = {popt[5]:.3f} +/- {perr[5]:.3f}",
    transform=plt.gca().transAxes,
    va="top"
    )
plt.legend()
plt.tight_layout()
plt.savefig("soluzione_02cc.pdf")
plt.show() 

#iniziamo con la sottrazione del fondo
#creiamo l'array del fondo
y_bg = background(x,popt[-3],popt[-2],popt[-1])
#fondo sottatto
y_clean = n-y_bg
#calcolo dell'errore dei dati con il fondo sottratto
y_bg = background(x, popt[-3], popt[-2], popt[-1])
y_clean = n - y_bg

# Covarianza dei parametri del fondo
cov_bg = pcov[-3:, -3:]

# Jacobiano
J = np.column_stack((
    np.ones_like(x),
    x,
    x**2
))

# Errore dovuto al fit del fondo
var_bg = np.einsum('ij,jk,ik->i', J, cov_bg, J)
var_bg = np.clip(var_bg, 0, None)

err_bg = np.sqrt(var_bg)

# errore totale sul dato con fondo sottratto
ey_clean = np.sqrt(err**2 + err_bg**2)

plt.xlabel("x") 
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")
plt.errorbar(x,y_clean,yerr=ey_clean,fmt="*",ms=5,label="segnali")
plt.plot(x,y_bg,label="fondo")
plt.plot(x,signal(x,popt[0],popt[1],popt[2]),label="segnale 1")
plt.plot(x,signal(x,popt[3],popt[4],popt[5]),label="segnale 2")
plt.legend()
plt.show()

#errore cresce!
plt.figure()
plt.plot(x, err, "o-", label="errore dati")
plt.plot(x, err_bg, "o-", label="errore fit del fondo")
plt.plot(x, ey_clean, "o-", label="errore totale dopo sottrazione")
plt.xlabel("x")
plt.ylabel("Errore")
plt.legend()
plt.tight_layout()
plt.show()

# Valori previsti dal fit globale
y_fit = modello(x, *popt)

# Residui sui dati ORIGINALI
residuals = n - y_fit

plt.figure(figsize=(8,4))

plt.errorbar(
    x, residuals,
    yerr=err,
    fmt='o',
    markersize=4,
    capsize=2
)

plt.axhline(0, color='red', linestyle='--')

plt.xlabel("x")
plt.ylabel("Residui")
plt.tight_layout()
plt.show()

pull = residuals / err

plt.figure(figsize=(8,4))

plt.axhline(0, color='red', linestyle='--')

plt.plot(
    x, pull,
    'o',
    markersize=4
)

plt.xlabel("x")
plt.ylabel("Pull")
plt.tight_layout()
plt.show()



# eliminiamo eventuali valori non finiti
pull = pull[np.isfinite(pull)]

# istogramma dei pull
plt.figure(figsize=(7, 5))

n_pull, bins_pull, _ = plt.hist(
    pull,
    bins=12,
    histtype="step",
    linewidth=1.5,
    color="black",
    label="Pull"
)

# centri dei bin
x_pull = 0.5 * (bins_pull[1:] + bins_pull[:-1])


# funzione gaussiana
def gauss_pull(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2 / (2*sigma**2))


# valori iniziali
p0_pull = [
    np.max(n_pull),
    np.mean(pull),
    np.std(pull)
]

# considero solo i bin popolati
mask = n_pull > 0

# errore poissoniano sui contenuti dell'istogramma
err_pull = np.sqrt(n_pull[mask])

# fit gaussiano
popt_pull, pcov_pull = curve_fit(
    gauss_pull,
    x_pull[mask],
    n_pull[mask],
    p0=p0_pull,
    sigma=err_pull,
    absolute_sigma=True
)

perr_pull = np.sqrt(np.diag(pcov_pull))


# curva da disegnare
xx = np.linspace(bins_pull[0], bins_pull[-1], 500)

plt.plot(
    xx,
    gauss_pull(xx, *popt_pull),
    "r-",
    label="Fit gaussiano"
)


# risultati del fit
plt.text(
    0.04, 0.95,
    rf"$\mu = {popt_pull[1]:.3f} \pm {perr_pull[1]:.3f}$" "\n"
    rf"$\sigma = {popt_pull[2]:.3f} \pm {perr_pull[2]:.3f}$",
    transform=plt.gca().transAxes,
    va="top"
)

plt.axvline(0, color="gray", linestyle="--")

plt.xlabel("Pull")
plt.ylabel("Conteggi")
plt.legend()
plt.tight_layout()

plt.savefig("pull_residui.pdf")
plt.show()

# ============================================================
# Effetto del binning sul fit della distribuzione dei pull
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for iax, nbins in enumerate([15, 10]):

    # Istogramma dei pull
    n_pull, edges_pull, _ = ax[iax].hist(
        pull,
        bins=nbins,
        histtype="step",
        linewidth=1.8,
        color="black",
        label="Pull"
    )

    # Centri dei bin
    x_pull = 0.5 * (edges_pull[:-1] + edges_pull[1:])

    # Errori poissoniani sui conteggi
    err_pull = np.sqrt(n_pull)

    # elimino i bin vuoti dal fit
    mask = n_pull > 0

    # Fit gaussiano
    popt_pull, pcov_pull = curve_fit(
        gauss_pull,
        x_pull[mask],
        n_pull[mask],
        sigma=err_pull[mask],
        absolute_sigma=True,
        p0=[np.max(n_pull), 0.0, 1.0]
    )

    perr_pull = np.sqrt(np.diag(pcov_pull))

    # Curva continua per disegnare il fit
    xx_pull = np.linspace(edges_pull[0], edges_pull[-1], 500)

    ax[iax].plot(
        xx_pull,
        gauss_pull(xx_pull, *popt_pull),
        "r-",
        lw=2,
        label="Fit gaussiano"
    )

    # Risultati del fit
    ax[iax].text(
        0.05, 0.95,
        rf"$\mu = {popt_pull[1]:.3f} \pm {perr_pull[1]:.3f}$" "\n"
        rf"$\sigma = {popt_pull[2]:.3f} \pm {perr_pull[2]:.3f}$",
        transform=ax[iax].transAxes,
        va="top"
    )

    ax[iax].set_title(f"Pull distribution - {nbins} bin")
    ax[iax].set_xlabel("Pull")
    ax[iax].set_ylabel("Conteggi")
    ax[iax].legend()


plt.tight_layout()
plt.savefig("pull_binning_comparison.pdf")
plt.show()


from scipy.optimize import minimize

# ============================================================
# UNBINNED MAXIMUM LIKELIHOOD FIT DEI PULL
# ============================================================

def nll_gauss(par, data):
    mu, sigma = par

    if sigma <= 0:
        return np.inf

    return (
        len(data) * np.log(sigma)
        + np.sum((data - mu)**2) / (2 * sigma**2)
    )


result = minimize(
    nll_gauss,
    x0=[0.0, 1.0],
    args=(pull,)
)

mu_ml, sigma_ml = result.x

print("UNBINNED MAXIMUM LIKELIHOOD")
print("mu    =", mu_ml)
print("sigma =", sigma_ml)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for iax, nbins in enumerate([10, 20]):

    # istogramma SOLO per visualizzazione
    ax[iax].hist(
        pull,
        bins=nbins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="black",
        label=f"Dati: {nbins} bin"
    )

    xx = np.linspace(
        np.min(pull),
        np.max(pull),
        500
    )

    # gaussiana normalizzata ottenuta dalla likelihood unbinned
    yy = (
        1.0 / (np.sqrt(2*np.pi) * sigma_ml)
        * np.exp(-(xx-mu_ml)**2 / (2*sigma_ml**2))
    )

    ax[iax].plot(
        xx,
        yy,
        "r-",
        lw=2,
        label="Unbinned likelihood"
    )

    ax[iax].text(
        0.05, 0.95,
        rf"$\mu = {mu_ml:.3f}$" "\n"
        rf"$\sigma = {sigma_ml:.3f}$",
        transform=ax[iax].transAxes,
        va="top"
    )

    ax[iax].set_xlabel("Pull")
    ax[iax].set_ylabel("Densità")
    ax[iax].set_title(f"Visualizzazione con {nbins} bin")
    ax[iax].legend()

plt.tight_layout()
plt.savefig("pull_unbinned_likelihood.pdf")
plt.show()


