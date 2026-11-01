# Esercizi di fit: dati e soluzioni

Tutti i campioni sono riproducibili (seed: 29102026). Le soluzioni richiedono:

```bash
pip install numpy scipy matplotlib
```

## Esercizio 1 — Distribuzione gaussiana
Il file `dati_01_gaussiana.txt` contiene 10000 osservazioni. Realizzare un istogramma con barre d'errore di Poisson ed eseguire un fit gaussiano sui conteggi. Stimare media e deviazione standard con le rispettive incertezze. Calcolare il chi-quadro ridotto e rappresentare curva e parametri sulla figura.

## Esercizio 2 — Segnale gaussiano con fondo parabolico
Il file `dati_02_gaussiana_fondo_parabolico.txt` contiene lo stesso segnale dell'esercizio 1, contaminato da un fondo parabolico concavo verso il basso definito nello stesso intervallo dei dati gaussiani. Costruire l'istogramma e fittarlo con il modello `Gaussiana + polinomio di secondo grado`. Confrontare media e sigma con l'esercizio 1. Verificare che il modello rimanga non negativo nell'intervallo di fit.

## Esercizio 3 — Andamenti lineare ed esponenziale
1. `dati_03a_lineare.csv`: fittare `y = q + m x`.
2. `dati_03b_esponenziale.csv`: fittare `y = A exp(k x)`.

Per entrambi: mostrare dati, curva, residui e parametri. In questo esercizio non sono fornite incertezze sperimentali: il fit è non pesato.

## Esercizio 4 — Incertezze su x e y
I file `dati_04a_lineare_errori.csv` e `dati_04b_esponenziale_errori.csv` contengono `x, y, sigma_x, sigma_y`, con incertezze dell'ordine del 10%. Eseguire un fit che tenga conto simultaneamente delle incertezze su entrambe le coordinate usando ODR (Orthogonal Distance Regression). Confrontare con un fit che consideri soltanto `sigma_y`.

## Esercizio 5 — Trend sigmoidale
Il file `dati_05_sigmoide.csv` contiene punti tra x=10 e x=30. Fittare

`y = y0 + L / (1 + exp(-(x-x0)/s))`

Stimare livello inferiore `y0`, ampiezza `L`, punto di flesso `x0` e scala `s`. Disegnare anche i residui.

## File delle soluzioni
- `soluzione_01.py`
- `soluzione_02.py`
- `soluzione_03.py`
- `soluzione_04.py`
- `soluzione_05.py`

Ogni programma salva automaticamente una figura PDF nella cartella corrente.
