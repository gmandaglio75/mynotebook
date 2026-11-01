import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def chi2_ridotto(y, yfit, sigma, npar):
    mask = sigma > 0
    chi2 = np.sum(((y[mask]-yfit[mask])/sigma[mask])**2)
    ndof = np.count_nonzero(mask)-npar
    return chi2/ndof, chi2, ndof

def gauss_conteggi(x, N, mu, sigma, larghezza_bin):
    return N*larghezza_bin/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*((x-mu)/sigma)**2)

dati=np.loadtxt("dati_01_gaussiana.txt")
n,bordi=np.histogram(dati,bins="fd")
centri=(bordi[:-1]+bordi[1:])/2
bw=np.diff(bordi)
err=np.sqrt(np.maximum(n,1))
modello=lambda x,N,mu,sigma: gauss_conteggi(x,N,mu,sigma,np.mean(bw))
p0=[len(dati),np.mean(dati),np.std(dati,ddof=1)]
popt,pcov=curve_fit(modello,centri,n,p0=p0,sigma=err,absolute_sigma=True,bounds=([0,-np.inf,1e-6],[np.inf,np.inf,np.inf]))
perr=np.sqrt(np.diag(pcov)); yfit=modello(centri,*popt)
chi2r,chi2,ndof=chi2_ridotto(n,yfit,err,3)
xx=np.linspace(bordi[0],bordi[-1],1000)
plt.errorbar(centri,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Fit gaussiano")
plt.xlabel("x"); plt.ylabel("Conteggi per bin"); plt.title("Esercizio 1")
testo=(f"mu = {popt[1]:.3f} +/- {perr[1]:.3f}\n"
       f"sigma = {popt[2]:.3f} +/- {perr[2]:.3f}\n"
       f"chi2/ndof = {chi2:.1f}/{ndof} = {chi2r:.2f}")
plt.text(.03,.96,testo,transform=plt.gca().transAxes,va="top")
plt.legend(); plt.tight_layout(); plt.savefig("soluzione_01.pdf"); plt.show()
