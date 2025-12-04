"""Materials Science domain module - materials optimization and parameter analysis"""

from kosmos.domains.materials.apis import (
    AflowClient,
    AflowMaterial,
    CitrinationClient,
    CitrinationData,
    MaterialProperties,
    MaterialsProjectClient,
    NOMADClient,
    NomadEntry,
    PerovskiteDBClient,
    PerovskiteExperiment,
)
from kosmos.domains.materials.ontology import (
    MaterialsConcept,
    MaterialsOntology,
    MaterialsRelation,
    MaterialsRelationType,
)
from kosmos.domains.materials.optimization import (
    CorrelationResult,
    DOEResult,
    MaterialsOptimizer,
    OptimizationResult,
    SHAPResult,
)


__all__ = [
    # API Clients
    "MaterialsProjectClient",
    "NOMADClient",
    "AflowClient",
    "CitrinationClient",
    "PerovskiteDBClient",
    # API Data Models
    "MaterialProperties",
    "NomadEntry",
    "AflowMaterial",
    "CitrinationData",
    "PerovskiteExperiment",
    # Optimization
    "MaterialsOptimizer",
    "CorrelationResult",
    "SHAPResult",
    "OptimizationResult",
    "DOEResult",
    # Ontology
    "MaterialsOntology",
    "MaterialsConcept",
    "MaterialsRelation",
    "MaterialsRelationType",
]
