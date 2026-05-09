# Sofia example logs

Three stripped Gaussian opt logs per macrocycle class. Each file contains only:
- the route line (`#` directive)
- charge/multiplicity
- the **last** `Standard orientation:` block
- a `Normal termination` line at the bottom

The full DFT output (SCF iterations, gradients, energies, populations) has been
removed to keep the repo small. Sofia's parser only needs the geometry + a few
header lines, so these stripped logs run end-to-end.

## What's here

| file | macrocycle | metal | spin | atoms | distortion (D_op) |
|---|---|---|---|---|---|
| `porphyrin/Co1_Pro.log` | bare metalloporphine | Co | S=0 | 37 | essentially planar (0.004 Å) |
| `porphyrin/Fe2_Pro.log` | bare metalloporphine | Fe | S=½ | 37 | essentially planar (0.004 Å) |
| `porphyrin/Ni2_Pro.log` | bare metalloporphine | Ni | S=½ | 37 | essentially planar (0.001 Å); shows the canonical Ni(I)/S=½ B1g Jahn–Teller distortion in the IP channel |
| `corrin/CoI_Corrin_S1.log` | bare Co(I) corrin | Co | S=0 | 45 | 0.49 Å (ruf+pro); default reference for the corrin viz bundle |
| `corrin/Co1corrinOH.log` | di-hydroxy corrin | Co | S=0 | 49 | 0.90 Å (propellered + saddled) |
| `corrin/Fe2corrinOH.log` | di-hydroxy corrin | Fe | S=½ | 49 | 0.94 Å |
| `corrin/Ni4corrinOH.log` | di-hydroxy corrin | Ni | S=3/2 | 49 | 1.27 Å (high-distortion Ni(I)/S=3/2 outlier) |
| `corphin/F430_Co1.log` | F430 / corphin (free) | Co | S=0 | 61 | 2.82 Å (most distorted of the F430 set) |
| `corphin/F430_Ni2.log` | F430 / corphin (free) | Ni | S=½ | 61 | 1.58 Å (the F430-native state — least distorted) |
| `corphin/F430_Mn3.log` | F430 / corphin (free) | Mn | S=1 | 61 | 2.25 Å (Mn-typical ruffling) |

## Smoke tests

Single file:
```bash
python3 sofia.py porphyrin examples/porphyrin/Co1_Pro.log
python3 sofia.py corrin    examples/corrin/Co1corrinOH.log
python3 sofia.py corphin   examples/corphin/F430_Co1.log
```

Auto-detect (Sofia identifies the macrocycle from graph topology):
```bash
python3 sofia.py auto examples/corphin/F430_Co1.log
```

Whole directory (Sofia processes every `.log` recursively, one row per file
in the CSV, one report per file):
```bash
python3 sofia.py porphyrin examples/porphyrin --csv out.csv --report reports/
python3 sofia.py corrin    examples/corrin    --csv out.csv --report reports/
python3 sofia.py corphin   examples/corphin   --csv out.csv --report reports/
```

If any of these fail, your install is broken — file an issue on GitHub.
