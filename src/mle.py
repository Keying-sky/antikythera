import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
from . import AntikytheraModel

def ml_estimate(model, is_aniso=True, init=None):
    """
    Get the maximum likelihood parameters for the model after optimisation.
    
    Args:
        model: The AntikytheraModel instance
        is_aniso: Whether to use the anisotropic model
        init: Initial parameters for optimisation
    
    Returns: Optimised parameters
    """
    def func(params):
        if is_aniso:
            return -model.ll_aniso(params)
        else:
            return -model.ll_iso(params)
    
    def grad(params):
        if is_aniso:
            return -model.grad_ll_aniso(params)
        else:
            return -model.grad_ll_iso(params)
    
    from scipy.optimize import check_grad
    error = check_grad(func, grad, init)
    print(f"Gradient check error: {error}")


    # perform optimisation using L-BFGS-B
    bounds = []
    
    bounds.append((300, 400)) # N
    bounds.append((50, 100)) # r
  
    if is_aniso:
        bounds.append((1e-4, 1))  # sigma_r
        bounds.append((1e-4, 1))  # sigma_t
    else:
        bounds.append((1e-4, 1))  # sigma
    
    for _ in range(model.n_sections):
        bounds.append((70, 90))  # x0j
        bounds.append((100, 150))  # y0j
    for _ in range(model.n_sections):
        bounds.append((np.radians(-180), np.radians(-90)))  # alphas
    
    result = minimize(
        func,
        init,
        method='L-BFGS-B',
        jac=grad,
        bounds=bounds,
        options={'disp': True, 'maxiter': 100000, 'gtol': 1e-6}
    )
    
    print(f"Optimisation successful: {result.success}")
    print(f"Final negative log-likelihood: {result.fun}")
    
    return result.x

def plot_mle(data, sorted_idxs, mle_iso, mle_aniso):
    """
    Plot the measured hole locations with maximum likelihood predictions
    
    Args:
        data: Array of measured hole locations
        sorted_idxs: List of lists with indices for each section
        mle_iso: Maximum likelihood parameters for the isotropic model
        mle_aniso: Maximum likelihood parameters for the anisotropic model
    """
    plt.figure(figsize=(14, 10))
    
    # Plot the original measured hole locations
    sections = range(len(sorted_idxs))
    colors = plt.cm.Set1(np.linspace(0, 1, len(sections)))
    
    for i, section_idxs in enumerate(sorted_idxs):
        section_data = data[[x-1 for x in section_idxs]]
        plt.scatter(
            section_data[:, 0], 
            section_data[:, 1],
            s=50,  
            color=colors[i],
            edgecolors='black',
            alpha=0.25,
            label=f'Section {i} (Measured)'
        )
    
    model = AntikytheraModel(data, sorted_idxs)
    
    iso_params_dict = model.unpack_params(mle_iso)
    aniso_params_dict = model.unpack_params(mle_aniso)
    
    iso_preds = []
    aniso_preds = []
    
    for j in range(model.n_sections):
        for i in sorted_idxs[j]:
            iso_pred = model.calc_position(
                i, 
                iso_params_dict['N'], 
                iso_params_dict['r'],
                iso_params_dict['centers'][j], 
                iso_params_dict['alphas'][j]
            )
            iso_preds.append((i, j, iso_pred))

            aniso_pred = model.calc_position(
                i, 
                aniso_params_dict['N'], 
                aniso_params_dict['r'],
                aniso_params_dict['centers'][j], 
                aniso_params_dict['alphas'][j]
            )
            aniso_preds.append((i, j, aniso_pred))
    

    for i, j, pred in iso_preds:
        plt.scatter(
            pred[0], 
            pred[1],
            s=50,  
            color='none',
            edgecolors=colors[j],
            marker='s', 
            alpha=1,
        )
    
    for i, j, pred in aniso_preds:
        plt.scatter(
            pred[0], 
            pred[1],
            s=50,  
            color='none',
            edgecolors=colors[j],
            marker='^',  
            alpha=1,
        )
    
    plt.scatter([], [], s=50, color='none', edgecolors='black', marker='s', label='Isotropic Model Prediction')
    plt.scatter([], [], s=50, color='none', edgecolors='black', marker='^', label='Anisotropic Model Prediction')
    
    plt.title('Measured vs Maximum Likelihood Predictions', fontsize=16)
    plt.xlabel('X (mm)', fontsize=14)
    plt.ylabel('Y (mm)', fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')

    plt.xlim(10, 120)
    plt.ylim(55, 95)
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.legend(bbox_to_anchor=(0.75, 1), loc='upper left')
    
    parent_path = Path().resolve().parent
    result_path = parent_path / "results"
    result_path.mkdir(exist_ok=True)
    plt.savefig(result_path / "hole_positions_with_predictions.png", dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

