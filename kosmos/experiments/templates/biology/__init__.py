"""Biology domain experiment templates."""

from kosmos.experiments.templates.biology.gwas_multimodal import GWASMultiModalTemplate
from kosmos.experiments.templates.biology.metabolomics_comparison import (
    MetabolomicsComparisonTemplate,
)


__all__ = [
    "MetabolomicsComparisonTemplate",
    "GWASMultiModalTemplate",
]
