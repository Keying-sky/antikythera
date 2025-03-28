import numpy as np
import pymc as pm
import arviz as az
from typing import List, Tuple
from . import AntikytheraHMC 

class ModelComparison:
    """
    Class for comparing Antikythera models using thermodynamic integration.
    """
    
    def __init__(self, sampler: AntikytheraHMC):
        """
        Initialise with an existing AntikytheraHMC sampler.
        
        Args:
            sampler: Configured AntikytheraHMC sampler with data
        """
        self.sampler = sampler
        self.models = {}
        self.temperatures = None
        self.log_evidence = {}
        
    def set_temperature(self, n_temperatures: int = 20):
        """
        Set up temperatures for thermodynamic integration.
        
        Args:
            n_temperatures: Number of temperature points to use
        """
        # create temperatures
        betas = np.linspace(0, 1, n_temperatures)
        betas = betas**5     # use more points near 0 to improve numerical stability when integrating
        self.temperatures = betas
        
    def power_posterior(self, is_aniso: bool = True, beta: float = 1.0, **priors) -> pm.Model:                                  
        """
        Build a model with a powered posterior for thermodynamic integration.
        
        Args:
            is_aniso: Whether to use anisotropic error model
            beta: Temperature parameter (0 = prior only, 1 = full posterior)
            **priors: Priors passed to build_model
            
        Returns: PyMC model with powered posterior
        """
        # default priors
        N_prior = priors.get('N_prior', (320, 370))
        r_prior = priors.get('r_prior', (60, 90))
        sigma_prior = priors.get('sigma_prior', (0.001, 0.3))
        x0j_prior = priors.get('x0j_prior', (70, 90)) 
        y0j_prior = priors.get('y0j_prior', (125, 145))
        alpha_prior = priors.get('alpha_prior', (-3.0, -2.0))
        
        with pm.Model() as power_model:
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
            
            for j in range(self.sampler.n_sections):
                x0j_name = f"x0_{j}"
                y0j_name = f"y0_{j}"
                alpha_j_name = f"alpha_{j}"
                
                x0j = pm.Uniform(x0j_name, lower=x0j_prior[0], upper=x0j_prior[1])
                y0j = pm.Uniform(y0j_name, lower=y0j_prior[0], upper=y0j_prior[1])
                alpha_j = pm.Uniform(alpha_j_name, lower=alpha_prior[0], upper=alpha_prior[1])
                
                centers_x.append(x0j)
                centers_y.append(y0j)
                alphas.append(alpha_j)
            
            # calculate likelihood terms, scale by beta for power posterior
            for j in range(self.sampler.n_sections):
                for i in self.sampler.sorted_idxs[j]:
                    e_r, e_t = self.sampler._calc_errors(
                        i, N, r, 
                        pm.math.stack([centers_x[j], centers_y[j]]), 
                        alphas[j]
                    )

                    e_r_det = pm.Deterministic(f"e_r_{i}_{j}", e_r)
                    e_t_det = pm.Deterministic(f"e_t_{i}_{j}", e_t)

                    if is_aniso:
                        pm.Potential(
                            f"likelihood_{i}_{j}",
                            -0.5 * beta * (e_r_det**2 / sigma_r**2 + e_t_det**2 / sigma_t**2)
                        )
                    else:
                        pm.Potential(
                            f"likelihood_{i}_{j}",
                            -0.5 * beta * (e_r_det**2 + e_t_det**2) / sigma**2
                        )
            
            power_model.is_aniso = is_aniso
            power_model.beta = beta
            
        return power_model
    
    def run_therm_integ(self, name: str, is_aniso: bool,n_samples: int = 1000, 
                        n_tune: int = 1000, cores: int = 2, **priors) -> None:                       
        """
        Run thermodynamic integration for a specific model type.
        
        Args:
            name: Name to give this model in results
            is_aniso: Whether to use anisotropic error model
            n_samples: Number of posterior samples per temperature
            n_tune: Number of tuning steps per temperature
            cores: Number of cores to use
            **priors: Priors
        """
        if self.temperatures is None:
            self.set_temperature()
            
        # store expected log-likelihoods at each temperature
        expected_lls = []
        
        for beta in self.temperatures:
            print(f"Running temperature β = {beta:.4f}")
            
            # build model
            model = self.power_posterior(
                is_aniso=is_aniso, 
                beta=beta,
                **priors
            )
            
            # sample from the power posterior
            with model:
                start = {'r':70,
                     'x0_0':80,
                     'x0_1':80,
                     'x0_2':80,
                     'x0_3':80,
                     'x0_4':80,
                     'x0_5':80,}
                            
                trace = pm.sample(
                    draws=n_samples,
                    tune=n_tune,
                    cores=cores,
                    target_accept=0.8,
                    random_seed=1224,
                    return_inferencedata=True,
                    init='jitter+adapt_diag',
                    initvals=start
                )
            
            # calculate the expected log-likelihood (not scaled by beta)
            e_ll = self._calc_E_ll(trace, is_aniso)
            expected_lls.append(e_ll)
            
        log_evidence = self._calc_lgZ(expected_lls)
        
        self.models[name] = {
            'is_aniso': is_aniso,
            'temperatures': self.temperatures,
            'expected_lls': expected_lls,
            'log_evidence': log_evidence
        }
        self.log_evidence[name] = log_evidence
        
    def _calc_E_ll(self, trace: az.InferenceData, is_aniso: bool) -> float:                  
        """
        Calculate the expected log-likelihood under a power posterior.
        
        Args:
            trace: ArviZ InferenceData with samples
            is_aniso: Whether the model is anisotropic
            
        Returns:
            expected log-likelihood
        """
        samples = trace.posterior

        lls = []
        
        for i in range(len(samples.chain)):
            for j in range(len(samples.draw)):
                N = samples.N[i, j].values
                r = samples.r[i, j].values
                
                if is_aniso:
                    sigma_r = samples.sigma_r[i, j].values
                    sigma_t = samples.sigma_t[i, j].values
                else:
                    sigma = samples.sigma[i, j].values
                
                centers = []
                alphas = []
                
                for k in range(self.sampler.n_sections):
                    x0 = samples[f"x0_{k}"][i, j].values
                    y0 = samples[f"y0_{k}"][i, j].values
                    alpha = samples[f"alpha_{k}"][i, j].values
                    
                    centers.append((x0, y0))
                    alphas.append(alpha)
                
                # calculate log-likelihood (unscaled by beta)
                if is_aniso:
                    ll = self._calc_ll_aniso(N, r, sigma_r, sigma_t, centers, alphas)
                else:
                    ll = self._calc_ll_iso(N, r, sigma, centers, alphas)
                
                lls.append(ll)
        
        # return expected value
        return np.mean(lls)
    
    def _calc_ll_iso(self, N, r, sigma, centers, alphas):
        """Calculate log-likelihood for isotropic model"""
        n = self.sampler.n_holes
        ll = -n * np.log(2 * np.pi * sigma**2)
        
        sum_squared_errors = 0
        for j in range(self.sampler.n_sections):
            for i in self.sampler.sorted_idxs[j]:
                x0, y0 = centers[j]
                alpha = alphas[j]
                
                phi = 2 * np.pi * (i - 1) / N + alpha
                
                m_x = x0 + r * np.cos(phi)
                m_y = y0 + r * np.sin(phi)
                
                d_i = self.sampler.data[i-1]
                
                e_x = d_i[0] - m_x
                e_y = d_i[1] - m_y
                
                sum_squared_errors += e_x**2 + e_y**2
        
        ll -= sum_squared_errors / (2 * sigma**2)
        return ll
    
    def _calc_ll_aniso(self, N, r, sigma_r, sigma_t, centers, alphas):
        """Calculate log-likelihood for anisotropic model"""
        n = self.sampler.n_holes
        ll = -n * np.log(2 * np.pi * sigma_r * sigma_t)
        
        for j in range(self.sampler.n_sections):
            for i in self.sampler.sorted_idxs[j]:
                x0, y0 = centers[j]
                alpha = alphas[j]
                
                phi = 2 * np.pi * (i - 1) / N + alpha
                
                m_x = x0 + r * np.cos(phi)
                m_y = y0 + r * np.sin(phi)
                
                d_i = self.sampler.data[i-1]
                
                e_x = d_i[0] - m_x
                e_y = d_i[1] - m_y
                
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                
                e_r = e_x * cos_phi + e_y * sin_phi
                e_t = e_x * sin_phi - e_y * cos_phi
                
                ll -= 0.5 * (e_r**2 / sigma_r**2 + e_t**2 / sigma_t**2)
        
        return ll
    
    def _calc_lgZ(self, expected_lls: List[float]) -> float:
        """
        Calculate the log evidence.
        
        Args: 
            expected_lls: List of expected log-likelihoods at each temperature
                
        Returns: 
            log_evidence: estimation of log evidence
        """
        betas = np.array(self.temperatures)
        lls = np.array(expected_lls)
        log_evidence = np.trapz(lls, betas)
        
        return log_evidence
    
    def compare_models(self) -> Tuple[str, float]:
        """
        Compare models using log evidence and Bayes factor.
        
        Returns: The better model and the bayes factor.
        """
      
        better_model = max(self.log_evidence, key=self.log_evidence.get)
        
        lgZ1 = self.log_evidence["Isotropic"]
        lgZ2 = self.log_evidence["Anisotropic"]

        bf = np.exp(lgZ1 - lgZ2) # bayes factor
        
        return better_model, bf
