import numpy as np

def position_samples(trace, is_anisotropic, hole_idx, section_idx, nsamples=100):                           
    """
    Generate posterior predictive samples for a specific hole.
    
    Args:
        trace: Posterior trace from HMC sampler
        is_anisotropic: Whether the model is anisotropic or isotropic
        hole_idx: Index of the hole to generate predictions for
        section_idx: Section index containing the selected hole
        nsamples: Number of predictive samples to generate
        
    Returns:
        Array of predicted hole positions (x, y)
    """
    # Get all parameter samples from the trace
    N_samples = trace.posterior["N"].values.flatten()
    r_samples = trace.posterior["r"].values.flatten()
    x0_samples = trace.posterior[f"x0_{section_idx}"].values.flatten()
    y0_samples = trace.posterior[f"y0_{section_idx}"].values.flatten()
    alpha_samples = trace.posterior[f"alpha_{section_idx}"].values.flatten()
    
    if is_anisotropic:
        sigma_r_samples = trace.posterior["sigma_r"].values.flatten()
        sigma_t_samples = trace.posterior["sigma_t"].values.flatten()
    else:
        sigma_samples = trace.posterior["sigma"].values.flatten()
    
    # Randomly select parameter combinations for predictive samples
    idx = np.random.choice(len(N_samples), nsamples, replace=True)
    
    predictive_samples = []
    for i in idx:
        # Calculate the angular position
        phi = 2 * np.pi * (hole_idx - 1) / N_samples[i] + alpha_samples[i]
        
        # Predicted position without error
        x_pred = x0_samples[i] + r_samples[i] * np.cos(phi)
        y_pred = y0_samples[i] + r_samples[i] * np.sin(phi)
        
        # Generate errors from the error model
        if is_anisotropic:
            # Radial error
            e_r = np.random.normal(0, sigma_r_samples[i])
            # Tangential error
            e_t = np.random.normal(0, sigma_t_samples[i])
        else:
            # Isotropic error
            e_r = np.random.normal(0, sigma_samples[i])
            e_t = np.random.normal(0, sigma_samples[i])
        
        # Convert radial and tangential errors to Cartesian coordinates
        e_x = e_r * np.cos(phi) - e_t * np.sin(phi)
        e_y = e_r * np.sin(phi) + e_t * np.cos(phi)
        
        # Add errors to predicted position
        x_pred_with_error = x_pred + e_x
        y_pred_with_error = y_pred + e_y
        
        predictive_samples.append([x_pred_with_error, y_pred_with_error])
    
    return np.array(predictive_samples)
