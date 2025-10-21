"""
Notebook Utilities.
"""

from typing import Literal

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .activations import Activations
from .data import RepeHonestDishonestFacts
from .detectors import (
    DirectionDetector,
    LATDetector,
    LogisticRegressionDetector,
    MMSDetector,
)
from .log import logger
from .tokenized_data import get_tokenized_dataset_pair
from .utils import REPO_ROOT


def get_repe_detector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    method: Literal["mms", "lat", "lr"] = "mms",
    from_file: bool = True,
) -> DirectionDetector:
    """Returns a lie detector based on the RepE paper.

    A positive score indicates honesty."""

    out_folder = REPO_ROOT / "deception_detection/notebooks/out/detectors/"
    out_folder.mkdir(exist_ok=True, parents=True)
    out_file = out_folder / f"{method}_repe_detector.pt"

    if method == "lr":
        detector_class = LogisticRegressionDetector
    elif method == "mms":
        detector_class = MMSDetector
    elif method == "lat":
        detector_class = LATDetector

    if from_file and out_file.exists():
        logger.info(f"Loading {method} detector from {out_file}")
        return detector_class.load(out_file)

    fit_data = RepeHonestDishonestFacts()
    honest_fit_toks, deceptive_fit_toks = get_tokenized_dataset_pair(fit_data, tokenizer)

    honest_fit_acts = Activations(model, honest_fit_toks)
    deceptive_fit_acts = Activations(model, deceptive_fit_toks)

    layers = list(range(8, 24))
    detector = detector_class(layers=layers)
    detector.fit(honest_fit_acts, deceptive_fit_acts)
    logger.info(f"Saving {method} detector to {out_file}")
    detector.save(out_file)

    return detector
