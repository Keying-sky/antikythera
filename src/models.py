import numpy as np
from typing import Dict, List, Tuple, Callable

class AntikytheraModel:
    """
    Model for the Antikythera mechanism calendar ring hole positions.
    Implements both isotropic and anisotropic (radial-tangential) error models.
    """
    
    def __init__(self, data: np.ndarray, sorted_idxs: List[List[int]]):
        """
        Initialise the model with measured hole locations.
        
        Args:
            data: Array of shape (n_holes, 2) with measured hole coordinates (x, y)
            sorted_idxs: List of lists containing indices of holes in each section
        """
        self.data = data                    # measured hole locations (x, y)
        self.n_holes = data.shape[0]        # number of measured holes
        self.sorted_idxs = sorted_idxs      # indxs of holes in each section (1-based, allow non-contiguous)
        self.n_sections = len(sorted_idxs)  # number of sections
    
    def unpack_params(self, params: np.ndarray) -> Dict:
        """
        Unpack parameter vector into a dictionary containing individual components.
        
        Args: Parameter vector [N, r, sigma(s), centers, alphas]
        
        Returns: Dictionary containing unpacked parameters
        """
        N = params[0]
        r = params[1]

         # check whether have two sigmas
        is_aniso = len(params) == 4 + 3 * self.n_sections 
        
        if is_aniso:
            sigma_r = params[2]
            sigma_t = params[3]
            sigma = np.array([sigma_r, sigma_t])
            offset = 4
        else:
            sigma = params[2]
            sigma = np.array([sigma])
            offset = 3
        
        # centers and alphas for each section
        centers = params[offset:offset + 2*self.n_sections].reshape(self.n_sections, 2)
        alphas = params[offset + 2*self.n_sections:]
        
        return {
            'N': N,
            'r': r,
            'sigma': sigma,
            'centers': centers,
            'alphas': alphas,
            'is_aniso': is_aniso
        }

    def calc_position(self, i: int, N: float, r: float, 
                           center_j: np.ndarray, alpha_j: float) -> np.ndarray:
        """
        Calculate the predicted position of hole i in section j.
        
        Args:
            i: Index of the hole (1-based)
            N: Number of holes in the complete ring
            r: Radius of the circle
            center_j: Center coordinates (x0_j, y0_j) for section j
            alpha_j: Rotation parameter for section j
            
        Returns:
            Predicted position (x_ij, y_ij)
        """
        phi = self._calc_phi(i, N, alpha_j)
        x = center_j[0] + r * np.cos(phi)
        y = center_j[1] + r * np.sin(phi)
        return np.array([x, y])
      
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
    
    def ll_aniso(self, params: np.ndarray) -> float:
        """
        Calculate the log-likelihood for the anisotropic model.
        
        Args: Parameter vector

        Returns: Log-likelihood value
        """
        param_dict = self.unpack_params(params)
        N = param_dict['N']
        r = param_dict['r']
        sigma_r = param_dict['sigma_params'][0]
        sigma_t = param_dict['sigma_params'][1]
        centers = param_dict['centers']
        alphas = param_dict['alphas']
        
        log_like = -self.n_holes * np.log(2 * np.pi * sigma_r * sigma_t)
        
        sum_term = 0
        for j in range(self.n_sections):
            for i in self.sorted_idxs[j]:
                e_r, e_t = self._calc_errors(i, N, r, centers[j], alphas[j])
                sum_term += (e_r**2 / sigma_r**2 + e_t**2 / sigma_t**2)
        
        log_like -= 0.5 * sum_term
        
        return log_like
    
    def ll_iso(self, params: np.ndarray) -> float:
        """
        Calculate the log-likelihood for the isotropic model.
        
        Args: Parameter vector   
        
        Returns: Log-likelihood value
            
        """
        param_dict = self.unpack_params(params)
        N = param_dict['N']
        r = param_dict['r']
        sigma = param_dict['sigma_params'][0]
        centers = param_dict['centers']
        alphas = param_dict['alphas']
        
        # Initialize log-likelihood
        log_like = -self.n_holes * np.log(2 * np.pi * sigma**2)
        
        # Sum over all holes
        sum_term = 0
        for j in range(self.n_sections):
            for i in self.sorted_idxs[j]:
                e_r, e_t = self._calc_errors(i, N, r, centers[j], alphas[j])
                sum_term += (e_r**2 + e_t**2)
        
        log_like -= sum_term / (2 * sigma**2)
        
        return log_like
    
    def grad_ll_aniso(self, params: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the log-likelihood for the anisotropic model,
        according to analytically formulas. (See report for derivation process).
        
        Args: Parameter vector
   
        Returns: Gradient vector     
        """
        param_dict = self.unpack_params(params)
        N = param_dict['N']
        r = param_dict['r']
        sigma_r = param_dict['sigma_params'][0]
        sigma_t = param_dict['sigma_params'][1]
        centers = param_dict['centers']
        alphas = param_dict['alphas']
        
        dldN = 0
        dldr = 0
        dldsigma_r = -self.n_holes / sigma_r
        dldsigma_t = -self.n_holes / sigma_t
        dldcenters = np.zeros_like(centers)
        dldalphas = np.zeros_like(alphas)
        
        sum_er_squared = 0
        sum_et_squared = 0
        
        for j in range(self.n_sections):
            for i in self.sorted_idxs[j]:
                e_r, e_t = self._calc_errors(i, N, r, centers[j], alphas[j])
                phi = self._calc_phi(i, N, alphas[j])
                
                sum_er_squared += e_r**2
                sum_et_squared += e_t**2
                
                xi, yi = self.data[i-1]
                x0j, y0j = centers[j]
                
                # r gradient
                dldr += ((xi - x0j) * np.cos(phi) + (yi - y0j) * np.sin(phi) - r) / sigma_r**2
                
                # N gradient
                dN_term1 = (1/sigma_r**2 - 1/sigma_t**2) * (
                    (xi - x0j)**2 * np.sin(phi) * np.cos(phi) - 
                    (xi - x0j) * (yi - y0j) * np.cos(2*phi) - 
                    (yi - y0j)**2 * np.sin(phi) * np.cos(phi)
                )
                dN_term2 = -r/sigma_r**2 * ((xi - x0j) * np.sin(phi) - (yi - y0j) * np.cos(phi))
                dldN += -2 * np.pi * (i - 1) / (N**2) * (dN_term1 + dN_term2)
                
                # x0j gradient
                dldcenters[j, 0] += (1/sigma_r**2) * ((xi - x0j) * np.cos(phi)**2 + 
                                                     (yi - y0j) * np.sin(phi) * np.cos(phi) - 
                                                     r * np.cos(phi))
                dldcenters[j, 0] += (1/sigma_t**2) * ((xi - x0j) * np.sin(phi)**2 - 
                                                     (yi - y0j) * np.sin(phi) * np.cos(phi))
                
                # y0j gradient
                dldcenters[j, 1] += (1/sigma_r**2) * ((xi - x0j) * np.sin(phi) * np.cos(phi) + 
                                                     (yi - y0j) * np.sin(phi)**2 - 
                                                     r * np.sin(phi))
                dldcenters[j, 1] += (1/sigma_t**2) * ((yi - y0j) * np.cos(phi)**2 - 
                                                     (xi - x0j) * np.sin(phi) * np.cos(phi))
                
                # alphas gradient
                alpha_term1 = (1/sigma_r**2 - 1/sigma_t**2) * (
                    (xi - x0j) * (yi - y0j) * np.cos(2*phi) + 
                    (yi - y0j)**2 * np.sin(phi) * np.cos(phi) - 
                    (xi - x0j)**2 * np.sin(phi) * np.cos(phi)
                )
                alpha_term2 = (1/sigma_r**2) * ((xi - x0j) * r * np.sin(phi) - (yi - y0j) * r * np.cos(phi))
                dldalphas[j] += -(alpha_term1 + alpha_term2)
    
        dldsigma_r += sum_er_squared / sigma_r**3
        dldsigma_t += sum_et_squared / sigma_t**3
        
        grad = np.concatenate([
            [dldN], 
            [dldr], 
            [dldsigma_r], 
            [dldsigma_t], 
            dldcenters.flatten(), 
            dldalphas
        ])
        
        return grad
    
    def grad_ll_iso(self, params: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the log-likelihood for the isotropic model,
        according to analytically formulas. (See report for derivation process).
        
        Args: Parameter vector  
            
        Returns: Gradient vector
        """
        param_dict = self.unpack_params(params)
        N = param_dict['N']
        r = param_dict['r']
        sigma = param_dict['sigma_params'][0]
        centers = param_dict['centers']
        alphas = param_dict['alphas']
        
        dldN = 0
        dldr = 0
        dldsigma = -2 * self.n_holes / sigma
        dldcenters = np.zeros_like(centers)
        dldalphas = np.zeros_like(alphas)
        
        sum_squared_errors = 0
        
        for j in range(self.n_sections):
            for i in self.sorted_idxs[j]:
                e_r, e_t = self._calc_errors(i, N, r, centers[j], alphas[j])
                phi = self._calc_phi(i, N, alphas[j])

                sum_squared_errors += e_r**2 + e_t**2
                
                xi, yi = self.data[i-1]
                x0j, y0j = centers[j]
                
                dldr += ((xi - x0j) * np.cos(phi) + (yi - y0j) * np.sin(phi) - r) / sigma**2

                dldN += (r * 2 * np.pi * (i - 1) / (N**2)) * ((xi - x0j) * np.sin(phi) - (yi - y0j) * np.cos(phi)) / sigma**2
                
                dldcenters[j, 0] += (xi - x0j - r * np.cos(phi)) / sigma**2
                dldcenters[j, 1] += (yi - y0j - r * np.sin(phi)) / sigma**2
                
                dldalphas[j] += -((xi - x0j) * r * np.sin(phi) - (yi - y0j) * r * np.cos(phi)) / sigma**2
        
        dldsigma += sum_squared_errors / sigma**3

        grad = np.concatenate([
            [dldN], 
            [dldr], 
            [dldsigma], 
            dldcenters.flatten(), 
            dldalphas
        ])
        
        return grad

    def _num_grad(self, func: Callable, params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Calculate numerical gradient using finite differences.
        
        Args:
            func: Function to differentiate
            params: Parameter vector
            eps: Step size
            
        Returns: Numerical gradient
        """
        n_params = len(params)
        grad = np.zeros(n_params)
        f0 = func(params)
        
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = func(params_plus)
            
            grad[i] = (f_plus - f0) / eps
            
        return grad
    
    def check_grad(self, params: np.ndarray, is_aniso: bool = True) -> dict:
        """
        Test the analytical gradients against numerical gradients.
        
        Args:
            params: Parameter vector
            is_aniso: Whether to test the anisotropic model
            
        Returns: Dictionary with test results
        """
        if is_aniso:
            func = self.ll_aniso
            analy_grad = self.grad_ll_aniso(params)
        else:
            func = self.ll_iso
            analy_grad = self.grad_ll_iso(params)
        
        num_grad = self._num_grad(func, params)
        
        # calculate differences
        abs_diff = np.abs(analy_grad - num_grad)
        rel_diff = abs_diff / (np.abs(analy_grad) + np.abs(num_grad) + 1e-10)
        
        # prepare parameter names
        if is_aniso:
            param_names = ['N', 'r', 'sigma_r', 'sigma_t']
        else:
            param_names = ['N', 'r', 'sigma']
            
        for j in range(self.n_sections):
            param_names.extend([f'x0_{j}', f'y0_{j}'])
        
        #for j in range(self.n_sections):
            param_names.append(f'alpha_{j}')

        return {
            'param_names': param_names,
            'analy_grad': analy_grad,
            'num_grad': num_grad,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
            'max_rel_diff': np.max(rel_diff),
            'max_abs_diff': np.max(abs_diff)
        }
