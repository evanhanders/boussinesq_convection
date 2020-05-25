import logging
import numpy as np

logger = logging.getLogger(__name__)

def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.

    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.scales
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)


def global_noise(domain, seed=42, n_modes=None, **kwargs):
    """
    Create a field filled with random noise of order 1.  

    Arguments:
    ----------
    seed : int, optional
        The seed for the random number generator; change it to get a different noise field.
    n_modes : int, optional
        The number of chebyshev modes to fill in the noise field.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand  = np.random.RandomState(seed=seed)
    noise_field = domain.new_field()

    if n_modes is None:
        noise = rand.standard_normal(gshape)[slices]

        # filter in k-space
        noise_field.set_scales(domain.dealias, keep_data=False)
        noise_field['g'] = noise
        filter_field(noise_field, **kwargs)
    else:
        n_modes = int(n_modes)
        scale   = n_modes/gshape[-1]
        gshape_small = domain.dist.grid_layout.global_shape(scales=scale)
        slices_small = domain.dist.grid_layout.slices(scales=scale)
        noise = rand.standard_normal(gshape_small)[slices_small]

        noise_field.set_scales(scale, keep_data=False)
        noise_field['g'] = noise
        noise_field.set_scales(domain.dealias, keep_data=True)
        
    return noise_field




