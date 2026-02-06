### JWST Extinction Pipeline

The goal of this project is to derive the extinction law(s) in the Galactic Center 
in the vicinity of Sgr A$^*$. 

The procedure is as follows: 

- Generate $n$ concentric annuli around Sgr A$^*$ 
    - ensure each annulus has enough stars, otherwise fuses with the annuli below 
- For each annulus, perform the procedure described in Section 3 of [this paper](https://dev-undergrad.dev/mulab.pdf)
- Analyze the results and make elegant figures.  

All outputs are under `output/runs/<run-id>/` (repo root):

```
output/
  runs/
    14/
      annuli/
        Annuli_0/
          annulus.json
          rc_stars.pickle
          tiles/
          fits/
          ratios/
          plots/
        Annuli_1/
        ...
      summary/
        run.json
        slopes.pickle
        ratios_A_lambda_over_A_F115W.pickle
        ratios_A_lambda_over_A_F212N.pickle
        plots/
```

Run the full pipeline from the repo root 

```
python -m moving_universe all --run-id 14 --n-annuli 14 --plot
```


Or run steps individually from the repo root 

```
python -m moving_universe annuli --run-id 14 --n-annuli 14 --plot
python -m moving_universe tiles --run-id 14 --plot
python -m moving_universe fits --run-id 14 --plot
python -m moving_universe ratios --run-id 14
```

Recreate the analysis plots 

```
python -m moving_universe plots --run-id 14
```
