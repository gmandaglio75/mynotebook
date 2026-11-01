import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# LETTURA DEI DATI
# ============================================================

# Legge il contenuto del file di testo.
# Il risultato è un array NumPy bidimensionale:
# ogni riga corrisponde a un punto/bin,
# ogni colonna contiene una diversa grandezza.
dati = np.loadtxt("cornuto.txt")

# Prima colonna: valori della variabile indipendente x
# (ad esempio il centro dei bin dell'istogramma).
x = dati[:, 0]

# Seconda colonna: numero di conteggi osservati in ciascun bin.
n = dati[:, 1]

# Quarta colonna: incertezza associata ai conteggi.
# Questi errori verranno utilizzati sia nel fit sia
# nella rappresentazione grafica dei dati e dei residui.
err = dati[:, 3]


# ============================================================
# MODELLO: DUE GAUSSIANE + FONDO PARABOLICO
# ============================================================

# Definiamo la funzione che descrive il modello utilizzato nel fit.
#
# Il modello contiene:
#
#   1) una prima gaussiana:
#          N * exp[-1/2 ((x-mu)/sigma)^2]
#
#   2) una seconda gaussiana:
#          N2 * exp[-1/2 ((x-mu2)/sigma2)^2]
#
#   3) un fondo descritto da un polinomio di secondo grado:
#          c0 + c1*x + c2*x^2
#
# In totale abbiamo quindi 9 parametri liberi:
#
# N, mu, sigma,
# N2, mu2, sigma2,
# c0, c1, c2

def modello(x, N, mu, sigma, N2, mu2, sigma2, c0, c1, c2):

    # Prima componente gaussiana
    sig1 = N * np.exp(-0.5 * ((x - mu) / sigma)**2)

    # Seconda componente gaussiana
    sig2 = N2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)

    # Fondo parabolico
    fondo = c0 + c1*x + c2*x**2

    # Il modello complessivo è la somma delle tre componenti
    return sig1 + sig2 + fondo


# ============================================================
# GRIGLIA PIÙ FITTA PER DISEGNARE IL MODELLO
# ============================================================

# I dati sperimentali sono disponibili solo nei punti x.
# Per disegnare invece la funzione di fit come una curva continua,
# costruiamo 1000 punti equidistanti tra 0 e 10.
#
# IMPORTANTE:
# xx NON viene utilizzato per eseguire il fit.
# Serve esclusivamente per disegnare in modo regolare la curva.
xx = np.linspace(0, 10, 1000)


# ============================================================
# IPOTESI 1
# Tutti i parametri iniziali uguali a 1
# ============================================================

# curve_fit deve partire da un'ipotesi iniziale sui parametri.
#
# Poiché il nostro modello ha 9 parametri, dobbiamo fornire
# 9 valori iniziali.
#
# In questa prima prova scegliamo deliberatamente una condizione
# molto semplice (e in generale poco informativa):
# tutti i parametri vengono inizializzati a 1.
#
# L'obiettivo didattico è verificare quanto il risultato del fit
# possa dipendere dalla scelta dei valori iniziali.
p0 = [1, 1, 1, 1, 1, 1, 1, 1, 1]


# ============================================================
# ESECUZIONE DEL FIT
# ============================================================

popt, pcov = curve_fit(
    modello,           # funzione da adattare ai dati
    x,                 # coordinate x dei dati
    n,                 # valori osservati
    p0=p0,             # valori iniziali dei parametri
    sigma=err,         # incertezze sui dati
    absolute_sigma=True,
    maxfev=50000       # numero massimo di valutazioni della funzione
)

# Con sigma=err il fit tiene conto delle diverse incertezze
# associate ai punti.
#
# absolute_sigma=True specifica che gli errori forniti in 'err'
# devono essere interpretati come incertezze assolute.
# Questa scelta influenza anche la matrice di covarianza pcov.


# ============================================================
# RISULTATI DEL FIT
# ============================================================

print("\nIpotesi 1")

# popt ("parameters optimized") contiene i valori dei parametri
# trovati dalla procedura di minimizzazione.
print(popt)


# ============================================================
# ERRORI SUI PARAMETRI
# ============================================================

# curve_fit restituisce anche la matrice di covarianza pcov.
#
# Gli elementi diagonali della matrice rappresentano
# le varianze stimate dei parametri:
#
#     pcov[i,i] = sigma_i^2
#
# Di conseguenza, facendo la radice quadrata degli elementi
# diagonali otteniamo gli errori standard sui parametri.
perr = np.sqrt(np.diag(pcov))


# ============================================================
# CALCOLO DEI RESIDUI
# ============================================================

# Il residuo per ciascun punto è definito come:
#
#       residuo = dato osservato - valore previsto dal modello
#
# Se il modello descrive adeguatamente i dati, i residui dovrebbero
# distribuirsi in maniera approssimativamente casuale attorno a zero,
# senza mostrare strutture sistematiche.
res = n - modello(x, *popt)


# ============================================================
# CREAZIONE DELLA FIGURA CON DUE PANNELLI
# ============================================================

# Creiamo una figura contenente due pannelli disposti verticalmente.
#
# plt.subplots(2, 1) significa:
#
#       2 righe
#       1 colonna
#
# quindi otteniamo:
#
#       ax1 -> pannello superiore
#       ax2 -> pannello inferiore
#
# Il pannello superiore conterrà dati + fit.
# Il pannello inferiore conterrà i residui.

fig, (ax1, ax2) = plt.subplots(
    2, 1,

    # figsize indica la dimensione TOTALE della figura,
    # espressa in pollici:
    #
    #       figsize = (larghezza, altezza)
    #
    # quindi la figura sarà larga 8 e alta 7 pollici.
    figsize=(8, 7),

    # I due pannelli devono utilizzare lo stesso asse x.
    #
    # Questo è particolarmente utile nel confronto tra dati
    # e residui, perché un determinato valore di x appare
    # esattamente nella stessa posizione nei due pannelli.
    sharex=True,

    # I due pannelli NON hanno la stessa altezza.
    #
    # height_ratios specifica il rapporto tra le loro altezze:
    #
    #       ax1 : ax2 = 3 : 1
    #
    # L'altezza disponibile viene quindi idealmente divisa
    # in 4 parti:
    #
    #       ax1 -> 3/4
    #       ax2 -> 1/4
    #
    # Il pannello del fit risulta quindi tre volte più alto
    # del pannello dei residui.
    gridspec_kw={"height_ratios": [3, 1]}
)


# ============================================================
# PANNELLO SUPERIORE: DATI E FIT
# ============================================================

# Disegniamo i dati sperimentali con le rispettive barre d'errore.
ax1.errorbar(
    x,
    n,
    yerr=err,          # incertezza verticale di ciascun punto
    fmt="o",           # rappresentazione dei dati con punti
    ms=3,              # dimensione dei marker
    label="Dati"
)


# Disegniamo sopra i dati il modello corrispondente
# ai parametri ottimali trovati dal fit.
#
# Qui utilizziamo xx anziché x per ottenere una curva più liscia.
#
# L'asterisco in *popt "spacchetta" l'array popt:
#
# modello(xx, *popt)
#
# equivale a passare singolarmente tutti i 9 parametri
# trovati dal fit alla funzione modello.
ax1.plot(
    xx,
    modello(xx, *popt),
    label="Due gaussiane + fondo parabolico"
)


# Etichetta dell'asse y del pannello superiore.
ax1.set_ylabel("Conteggi per bin")

# Titolo del grafico.
# Ricordiamo esplicitamente quali valori iniziali sono stati utilizzati.
ax1.set_title("Ipotesi 1: p0 = [1,1,1,1,1,1,1,1,1]")


# ============================================================
# VISUALIZZAZIONE DI ALCUNI PARAMETRI DEL FIT
# ============================================================

# Inseriamo all'interno del grafico i valori di:
#
#   mu1, sigma1, mu2, sigma2
#
# con le rispettive incertezze.
#
# popt contiene i valori dei parametri.
# perr contiene i loro errori.

ax1.text(
    .23, .46,

    # Il simbolo $...$ attiva la notazione matematica
    # di Matplotlib (MathText), permettendo di visualizzare
    # simboli greci come mu e sigma.
    f"$\\mu_1$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}\n"
    f"$\\sigma_1$ = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}\n"
    f"$\\mu_2$ = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}\n"
    f"$\\sigma_2$ = {popt[5]:.3f} $\\pm$ {perr[5]:.3f}",

    # Senza transform=ax1.transAxes le coordinate .23 e .46
    # sarebbero interpretate nelle unità fisiche degli assi.
    #
    # Con transform=ax1.transAxes diventano invece coordinate
    # relative al pannello:
    #
    #       x = 0 -> bordo sinistro
    #       x = 1 -> bordo destro
    #
    #       y = 0 -> bordo inferiore
    #       y = 1 -> bordo superiore
    #
    # Quindi (.23, .46) significa circa:
    # 23% della larghezza e 46% dell'altezza del pannello.
    transform=ax1.transAxes,

    # Allinea verticalmente il testo rispetto al suo bordo superiore.
    va="top"
)


# Visualizza la legenda contenente le etichette definite
# mediante label="Dati" e
# label="Due gaussiane + fondo parabolico".
ax1.legend()


# ============================================================
# PANNELLO INFERIORE: RESIDUI
# ============================================================

# Disegniamo i residui:
#
#       residuo_i = dato_i - modello_i
#
# con le relative barre d'errore.
#
# Qui utilizziamo err perché stiamo rappresentando
# l'incertezza associata ai dati osservati.
ax2.errorbar(
    x,
    res,
    yerr=err,
    fmt="o",
    ms=3
)
#================================
# LINEA DI RIFERIMENTO DEI RESIDUI
# ============================================================

# Tracciamo una linea orizzontale in corrispondenza di y = 0.
#
# Se dato e modello coincidessero perfettamente:
#
#       dato - modello = 0
#
# quindi tutti i residui cadrebbero su questa linea.
#
# Nella realtà ci aspettiamo invece fluttuazioni statistiche
# sopra e sotto lo zero.
ax2.axhline(
    0,
    color="black",
    linestyle="--"
)


# Etichetta dell'asse x.
#
# Poiché abbiamo utilizzato sharex=True, ax1 e ax2 condividono
# lo stesso asse x. È quindi sufficiente mettere l'etichetta
# sul pannello inferiore.
ax2.set_xlabel("x")

# Etichetta dell'asse y del pannello inferiore.
ax2.set_ylabel("Residui")


# ============================================================
# SISTEMAZIONE AUTOMATICA DEGLI SPAZI
# ============================================================

# tight_layout cerca di ottimizzare automaticamente gli spazi
# tra pannelli, titoli, etichette e bordi della figura,
# evitando per quanto possibile sovrapposizioni.
plt.tight_layout()


# ============================================================
# VISUALIZZAZIONE
# ============================================================

plt.show()

# ============================================================
# IPOTESI 2
# Miglioriamo l'ampiezza della prima gaussiana
# ============================================================

p0 = [605, 1, 1, 1, 1, 1, 1, 1, 1]

popt, pcov = curve_fit(
    modello, x, n,
    p0=p0,
    sigma=err,
    absolute_sigma=True,
    maxfev=50000
)

print("\nIpotesi 2")
print(popt)

perr = np.sqrt(np.diag(pcov))

res = n - modello(x, *popt)


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax1.errorbar(x, n, yerr=err, fmt="o", ms=3, label="Dati")
ax1.plot(xx, modello(xx, *popt), label="Due gaussiane + fondo parabolico")

ax1.set_ylabel("Conteggi per bin")
ax1.set_title("Ipotesi 2: p0 = [605,1,1,1,1,1,1,1,1]")

ax1.text(
    .23, .46,
    f"$\\mu_1$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}\n"
    f"$\\sigma_1$ = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}\n"
    f"$\\mu_2$ = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}\n"
    f"$\\sigma_2$ = {popt[5]:.3f} $\\pm$ {perr[5]:.3f}",
    transform=ax1.transAxes,
    va="top"
)

ax1.legend()

ax2.errorbar(x, res, yerr=err, fmt="o", ms=3)
ax2.axhline(0, color="black", linestyle="--")

ax2.set_xlabel("x")
ax2.set_ylabel("Residui")

plt.tight_layout()
plt.show()


# ============================================================
# IPOTESI 3
# Introduciamo una concavita' negativa per il fondo
# ============================================================

p0 = [605, 1, 1, 1, 1, 1, 1, 1, -10]

popt, pcov = curve_fit(
    modello, x, n,
    p0=p0,
    sigma=err,
    absolute_sigma=True,
    maxfev=50000
)

print("\nIpotesi 3")
print(popt)

perr = np.sqrt(np.diag(pcov))

res = n - modello(x, *popt)


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax1.errorbar(x, n, yerr=err, fmt="o", ms=3, label="Dati")
ax1.plot(xx, modello(xx, *popt), label="Due gaussiane + fondo parabolico")

ax1.set_ylabel("Conteggi per bin")
ax1.set_title("Ipotesi 3: p0 = [605,1,1,1,1,1,1,1,-10]")

ax1.text(
    .23, .46,
    f"$\\mu_1$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}\n"
    f"$\\sigma_1$ = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}\n"
    f"$\\mu_2$ = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}\n"
    f"$\\sigma_2$ = {popt[5]:.3f} $\\pm$ {perr[5]:.3f}",
    transform=ax1.transAxes,
    va="top"
)

ax1.legend()

ax2.errorbar(x, res, yerr=err, fmt="o", ms=3)
ax2.axhline(0, color="black", linestyle="--")

ax2.set_xlabel("x")
ax2.set_ylabel("Residui")

plt.tight_layout()
plt.show()


# ============================================================
# IPOTESI 4
# Spostiamo la seconda gaussiana verso x = 7
# ============================================================

p0 = [605, 1, 1, 1, 7, 1, 1, 1, -10]

popt, pcov = curve_fit(
    modello, x, n,
    p0=p0,
    sigma=err,
    absolute_sigma=True,
    maxfev=50000
)

print("\nIpotesi 4")
print(popt)

perr = np.sqrt(np.diag(pcov))

res = n - modello(x, *popt)


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax1.errorbar(x, n, yerr=err, fmt="o", ms=3, label="Dati")
ax1.plot(xx, modello(xx, *popt), label="Due gaussiane + fondo parabolico")

ax1.set_ylabel("Conteggi per bin")
ax1.set_title("Ipotesi 4: p0 = [605,1,1,1,7,1,1,1,-10]")

ax1.text(
    .23, .46,
    f"$\\mu_1$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}\n"
    f"$\\sigma_1$ = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}\n"
    f"$\\mu_2$ = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}\n"
    f"$\\sigma_2$ = {popt[5]:.3f} $\\pm$ {perr[5]:.3f}",
    transform=ax1.transAxes,
    va="top"
)

ax1.legend()

ax2.errorbar(x, res, yerr=err, fmt="o", ms=3)
ax2.axhline(0, color="black", linestyle="--")

ax2.set_xlabel("x")
ax2.set_ylabel("Residui")

plt.tight_layout()
plt.show()


# ============================================================
# IPOTESI 5
# Spostiamo anche la prima gaussiana verso x = 3
# ============================================================

p0 = [605, 3, 1, 1, 7, 1, 1, 1, -10]

popt, pcov = curve_fit(
    modello, x, n,
    p0=p0,
    sigma=err,
    absolute_sigma=True,
    maxfev=50000
)

print("\nIpotesi 5")
print(popt)

perr = np.sqrt(np.diag(pcov))

res = n - modello(x, *popt)


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax1.errorbar(x, n, yerr=err, fmt="o", ms=3, label="Dati")
ax1.plot(xx, modello(xx, *popt), label="Due gaussiane + fondo parabolico")

ax1.set_ylabel("Conteggi per bin")
ax1.set_title("Ipotesi 5: p0 = [605,3,1,1,7,1,1,1,-10]")

ax1.text(
    .23, .46,
    f"$\\mu_1$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}\n"
    f"$\\sigma_1$ = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}\n"
    f"$\\mu_2$ = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}\n"
    f"$\\sigma_2$ = {popt[5]:.3f} $\\pm$ {perr[5]:.3f}",
    transform=ax1.transAxes,
    va="top"
)

ax1.legend()

ax2.errorbar(x, res, yerr=err, fmt="o", ms=3)
ax2.axhline(0, color="black", linestyle="--")

ax2.set_xlabel("x")
ax2.set_ylabel("Residui")

plt.tight_layout()
plt.show()


# ============================================================
# IPOTESI 6
# Miglioriamo anche l'ampiezza della seconda gaussiana
# ============================================================

p0 = [605, 3, 1, 100, 7, 1, 1, 1, -10]

popt, pcov = curve_fit(
    modello, x, n,
    p0=p0,
    sigma=err,
    absolute_sigma=True,
    maxfev=50000
)

print("\nIpotesi 6")
print(popt)

perr = np.sqrt(np.diag(pcov))

res = n - modello(x, *popt)


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax1.errorbar(x, n, yerr=err, fmt="o", ms=3, label="Dati")
ax1.plot(xx, modello(xx, *popt), label="Due gaussiane + fondo parabolico")

ax1.set_ylabel("Conteggi per bin")
ax1.set_title("Ipotesi 6: p0 = [605,3,1,100,7,1,1,1,-10]")

ax1.text(
    .23, .46,
    f"$\\mu_1$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}\n"
    f"$\\sigma_1$ = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}\n"
    f"$\\mu_2$ = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}\n"
    f"$\\sigma_2$ = {popt[5]:.3f} $\\pm$ {perr[5]:.3f}",
    transform=ax1.transAxes,
    va="top"
)

ax1.legend()

ax2.errorbar(x, res, yerr=err, fmt="o", ms=3)
ax2.axhline(0, color="black", linestyle="--")

ax2.set_xlabel("x")
ax2.set_ylabel("Residui")

plt.tight_layout()
plt.show()


# ============================================================
# IPOTESI 7
# Valori iniziali ormai vicini alla struttura osservata
# ============================================================

p0 = [605, 3, 1.2, 300, 7, 0.2, -8, 10, -100]

popt, pcov = curve_fit(
    modello, x, n,
    p0=p0,
    sigma=err,
    absolute_sigma=True,
    maxfev=50000
)

print("\nIpotesi 7")
print(popt)

perr = np.sqrt(np.diag(pcov))

res = n - modello(x, *popt)


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

ax1.errorbar(x, n, yerr=err, fmt="o", ms=3, label="Dati")
ax1.plot(xx, modello(xx, *popt), label="Due gaussiane + fondo parabolico")

ax1.set_ylabel("Conteggi per bin")
ax1.set_title(
    "Ipotesi 7: p0 = [605,3,1.2,300,7,0.2,-8,10,-100]"
)

ax1.text(
    .23, .46,
    f"$\\mu_1$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}\n"
    f"$\\sigma_1$ = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}\n"
    f"$\\mu_2$ = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}\n"
    f"$\\sigma_2$ = {popt[5]:.3f} $\\pm$ {perr[5]:.3f}",
    transform=ax1.transAxes,
    va="top"
)

ax1.legend()


# ----- Residui -----

ax2.errorbar(
    x,
    res,
    yerr=err,
    fmt="o",
    ms=3
)

ax2.axhline(
    0,
    color="black",
    linestyle="--"
)

ax2.set_xlabel("x")
ax2.set_ylabel("Residui")

plt.tight_layout()
plt.show()
