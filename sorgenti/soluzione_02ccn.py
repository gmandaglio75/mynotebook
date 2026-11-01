import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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


