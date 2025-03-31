# S2 Coursework
### Keying Song (ks2146)

This coursework explores calendar ring hole data from fragments of the Antikythera Mechanism to infer the original number of holes in the complete ring using Bayesian inference with NUTS Hamiltonian Monte Carlo. Also compares two kind of error models using thermodynamic integration to calculate the odds ratio.


## Declaration
No auto-generation tools were used in this coursework.

## Project Structure
The main structure of the package `antikythera` is like:
```
.
├── anytikythera/
│   ├── __init__.py               # expose all classes and functions for importing
│   ├── compare.py                # module for model comparison
│   ├── explore.py                # module for raw data exploration and visualisation
│   ├── mle.py                    # module for maximum likelihood estimation
│   ├── models.py                 # module for models building
│   └── sampler.py                # module for HMC sampling with NUTS
|
├── report/                       # coursework report
|
├── data/                         # raw data from the Antikythera Mechanism calendar ring
├── results/                      # the folder to save all the results
|
├── run.ipynb                     # the main file to directly answer the questions of coursework 
|
├── pyproject.toml                     
└── README.md               
```

## Installation

1. Clone the repository:
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/s2_coursework/ks2146.git
```

2. Install: In the root directory:
```bash
pip install .
```

3. Use:
- After installing, all the classes and functions in `antikythera` can be imported and used anywhere on your own machine.
```python
from antikythera import AntikytheraModel, AntikytheraHMC, ModelComparison
```

- Run the notebook files in the folder `run` to run all the experiments reported on the report.

## Usage

The main workflow runs intuitively in the `run.ipynb`, which directly answers the questions of the coursework sequentially.

## Data Source

Data comes from Thoeni et al. (2019): "Replication Data for: Antikythera Mechanism Shows Evidence of Lunar Calendar" available at https://doi.org/10.7910/DVN/VJGLVS
