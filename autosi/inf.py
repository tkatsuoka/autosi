import numpy as np
from sicore import SelectiveInferenceNorm

from .utils import siarray


def inference(
    eta: siarray, prob_vec: siarray, var: float | np.ndarray, model: siarray, **kwargs
):
    # prob_val_vec = prob_vec.values().ravel()
    eta.add_prob_vec_nodes(prob_vec)
    eta.add_observed_model(model)
    eta_vec = eta.values().ravel()

    # 次元を確認
    # print("prob_val:", prob_val_vec.shape)
    si_calculator = SelectiveInferenceNorm(prob_vec.values().ravel(), var, eta_vec)


    result = si_calculator.inference(
        algorithm=eta.algorithm,  # (a, b, z) -> eta, l, u
        model_selector=eta.model_selector,
        **kwargs,
    )

    if np.isnan(si_calculator.stat):
        raise ValueError("No Hypothesis is obtained.")

    return result
