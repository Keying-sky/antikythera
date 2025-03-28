import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class AntikytheraHMC:
    """
    Hamiltonian Monte Carlo sampler for the Antikythera model.
    
    This class implements posterior sampling for both isotropic 
    and anisotropic error models using the No-U-Turn Sampler (NUTS).
    """
    
    def __init__(self, data: np.ndarray, sorted_idxs: List[List[int]]):
        """
        Initialise the HMC sampler with data.
        
        Args:
            data: Array of shape (n_holes, 2) with measured hole coordinates
            sorted_idxs: List of lists containing indices of holes in each section
        """
        self.data = data
        self.sorted_idxs = sorted_idxs
        self.n_holes = data.shape[0]
        self.n_sections = len(sorted_idxs)
        self.model = None
        self.trace = None
        
    def _calc_phi(self, i: int, N: float, alpha_j: float) -> float:
        """
        Calculate the angular position of hole i in section j.
        
        Args:
            i: Index of the hole (1-based)
            N: Number of holes in the complete ring
            alpha_j: Rotation parameter for section j
            
        Returns: Angular position phi_ij
        """
        return 2 * np.pi * (i - 1) / N + alpha_j
    
    def _calc_errors(self, i: int, N: float, r: float, 
                   center_j: np.ndarray, alpha_j: float) -> Tuple[float, float]:
        """
        Calculate radial and tangential errors for hole i in section j.
        
        Args:
            i: Index of the hole (1-based)
            N: Number of holes in the complete ring
            r: Radius of the circle
            center_j: Center coordinates (x0_j, y0_j) for section j
            alpha_j: Rotation parameter for section j
            
        Returns:
            Tuple (e_r, e_t) containing radial and tangential errors
        """
        d_i = self.data[i-1]  # here i is 1-based!!!
        
        phi = self._calc_phi(i, N, alpha_j)
        m_ij = np.array([center_j[0] + r * np.cos(phi), center_j[1] + r * np.sin(phi)])
        
        e_ij = d_i - m_ij
        
        r_hat = np.array([np.cos(phi), np.sin(phi)])
        t_hat = np.array([np.sin(phi), -np.cos(phi)])
        
        e_r = np.dot(e_ij, r_hat) 
        e_t = np.dot(e_ij, t_hat)
        
        return e_r, e_t
    
    def build_model(self, is_aniso: bool = True, 
                   N_prior: Tuple[float, float] = (320, 370),
                   r_prior: Tuple[float, float] = (60, 90),
                   sigma_prior: Tuple[float, float] = (0.001, 0.3),
                   x0j_prior: Tuple[float, float] = (70, 90),
                   y0j_prior: Tuple[float, float] = (125, 145),
                   alpha_prior: Tuple[float, float] = (-3.0, -2.0)) -> pm.Model:
        """
        Build a PyMC model for sampling.
        
        Args:
            is_aniso: Whether to use anisotropic error model
            N_prior: Prior range for N (number of holes)
            r_prior: Prior range for r (radius)
            sigma_prior: Prior range for sigma parameters
            center_prior: Prior range for center coordinates
            alpha_prior: Prior range for alpha parameters
            
        Returns: PyMC model
        """
        with pm.Model() as model:
            # prior distributions for parameters
            N = pm.Uniform("N", lower=N_prior[0], upper=N_prior[1])
            r = pm.Uniform("r", lower=r_prior[0], upper=r_prior[1])
            
            if is_aniso:
                sigma_r = pm.Uniform("sigma_r", lower=sigma_prior[0], upper=sigma_prior[1])
                sigma_t = pm.Uniform("sigma_t", lower=sigma_prior[0], upper=sigma_prior[1])
            else:
                sigma = pm.Uniform("sigma", lower=sigma_prior[0], upper=sigma_prior[1])
            
            centers_x = []
            centers_y = []
            alphas = []
            
            for j in range(self.n_sections):
                x0j = pm.Uniform(f"x0_{j}", lower=x0j_prior[0], upper=x0j_prior[1])
                y0j = pm.Uniform(f"y0_{j}", lower=y0j_prior[0], upper=y0j_prior[1])
                alpha_j = pm.Uniform(f"alpha_{j}", lower=alpha_prior[0], upper=alpha_prior[1])
                
                centers_x.append(x0j)
                centers_y.append(y0j)
                alphas.append(alpha_j)
            
            # log-likelihood calculation for each hole
            for j in range(self.n_sections):
                for i in self.sorted_idxs[j]:
                    e_r, e_t = self._calc_errors(
                        i, N, r, 
                        pm.math.stack([centers_x[j], centers_y[j]]), 
                        alphas[j]
                    )

                    e_r_det = pm.Deterministic(f"e_r_{i}_{j}", e_r)
                    e_t_det = pm.Deterministic(f"e_t_{i}_{j}", e_t)

                    if is_aniso:
                        pm.Potential(
                            f"likelihood_{i}_{j}",
                            -0.5 * (e_r_det**2 / sigma_r**2 + e_t_det**2 / sigma_t**2)
                        )
                    else:
                        pm.Potential(
                            f"likelihood_{i}_{j}",
                            -0.5 * (e_r_det**2 + e_t_det**2) / sigma**2
                        )
            
            model.is_aniso = is_aniso
        
        self.model = model
        return model
    
    def sample(self, n_samples: int = 1000, n_tune: int = 1000, 
               cores: int = 4, target_accept: float = 0.8,
               is_aniso: bool = True, seed: int = 1224,
               **priors) -> az.InferenceData:
        """
        Sample from the posterior distribution using NUTS.
        
        Args:
            n_samples: Number of posterior samples to draw
            n_tune: Number of tuning steps
            cores: Number of cores to use (chains)
            target_accept: Target acceptance rate
            is_aniso: Whether to use anisotropic error model
            seed: Random seed for reproducibility
            **priors: Arguments to pass to build_model for priors
            
        Returns:
            ArviZ InferenceData object with posterior samples
        """
        if self.model is None or self.model.is_aniso != is_aniso:
            self.build_model(is_aniso=is_aniso, **priors)

        # sample from the posterior
        with self.model:
            start = {'r':70,
                     'x0_0':80,
                     'x0_1':80,
                     'x0_2':80,
                     'x0_3':80,
                     'x0_4':80,
                     'x0_5':80,}

            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                cores=cores,
                target_accept=target_accept,
                random_seed=seed,
                return_inferencedata=True,
                init='jitter+adapt_diag',
                initvals=start, 
            )
        return self.trace
