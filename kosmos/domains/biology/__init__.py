"""Biology domain module - metabolomics, genomics, and multi-modal integration"""

from kosmos.domains.biology.apis import (
    ENCODEClient,
    EnsemblClient,
    GTExClient,
    GWASCatalogClient,
    HMDBClient,
    KEGGClient,
    MetaboLightsClient,
    PDBClient,
    UniProtClient,
    dbSNPClient,
)
from kosmos.domains.biology.genomics import (
    CompositeScore,
    EffectDirection,
    EvidenceLevel,
    GenomicsAnalyzer,
    GenomicsResult,
    MechanismRanking,
)
from kosmos.domains.biology.metabolomics import (
    MetaboliteCategory,
    MetaboliteType,
    MetabolomicsAnalyzer,
    MetabolomicsResult,
    PathwayComparison,
    PathwayPattern,
)
from kosmos.domains.biology.ontology import (
    BiologicalConcept,
    BiologicalRelation,
    BiologicalRelationType,
    BiologyOntology,
)


__all__ = [
    # API Clients
    "KEGGClient",
    "GWASCatalogClient",
    "GTExClient",
    "ENCODEClient",
    "dbSNPClient",
    "EnsemblClient",
    "HMDBClient",
    "MetaboLightsClient",
    "UniProtClient",
    "PDBClient",
    # Metabolomics
    "MetabolomicsAnalyzer",
    "MetabolomicsResult",
    "PathwayPattern",
    "PathwayComparison",
    "MetaboliteCategory",
    "MetaboliteType",
    # Genomics
    "GenomicsAnalyzer",
    "GenomicsResult",
    "CompositeScore",
    "MechanismRanking",
    "EvidenceLevel",
    "EffectDirection",
    # Ontology
    "BiologyOntology",
    "BiologicalConcept",
    "BiologicalRelation",
    "BiologicalRelationType",
]
