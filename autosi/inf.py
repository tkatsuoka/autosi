import numpy as np

from sicore import SelectiveInferenceNorm

from .utils import siarray


def inference(
    eta: siarray,
    prob_vec: siarray,
    var: float | np.ndarray,
    algorithm: callable,
    model: siarray = None,
    **kwargs,
):
    eta.add_prob_vec_nodes(prob_vec)
    eta.add_algorithm(algorithm)

    if model is None:
        eta.add_observed_model(eta)
    else:
        eta.add_observed_model(model)

    prob_vec = prob_vec.values().ravel()
    eta_vec = eta.values().ravel()

    si_calculator = SelectiveInferenceNorm(prob_vec, var, eta_vec)

    result = si_calculator.inference(
        algorithm=eta.calc_intervals,  # (a, b, z) -> eta, l, u
        model_selector=eta.model_selector,
        **kwargs,
    )

    if np.isnan(si_calculator.stat):
        raise ValueError("No Hypothesis is obtained.")

    return result
