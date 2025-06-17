from enum import Enum


class ReferenceMethod(str, Enum):
    """Enumeration of reference methods."""

    NONE = "none"
    """No reference method."""
    POINT = "point"
    """Reference point."""
    MEDIAN = "median"
    """Full-scene median per date."""
    BORDER = "border"
    """Median of border pixels."""
    HIGH_COHERENCE = "high_coherence"
    """Mean/median of high-quality mask."""

    def __str__(self) -> str:
        return self.value


class DisplacementDataset(str, Enum):
    """Enumeration of displacement datasets."""

    DISPLACEMENT = "displacement"
    SHORT_WAVELENGTH = "short_wavelength_displacement"

    def __str__(self) -> str:
        return self.value


class CorrectionDataset(str, Enum):
    """Enumeration of correction datasets."""

    SOLID_EARTH_TIDE = "/corrections/solid_earth_tide"
    IONOSPHERIC_DELAY = "/corrections/ionospheric_delay"

    def __str__(self) -> str:
        return self.value


class SamePerMinistackDataset(str, Enum):
    """Enumeration of datasets that are same per ministack."""

    TEMPORAL_COHERENCE = "temporal_coherence"
    PHASE_SIMILARITY = "phase_similarity"
    PERSISTENT_SCATTERER_MASK = "persistent_scatterer_mask"
    SHP_COUNTS = "shp_counts"

    def __str__(self) -> str:
        return self.value


class QualityDataset(str, Enum):
    """Enumeration of quality datasets."""

    TIMESERIES_INVERSION_RESIDUALS = "timeseries_inversion_residuals"
    CONNECTED_COMPONENT_LABELS = "connected_component_labels"
    RECOMMENDED_MASK = "recommended_mask"
    ESTIMATED_PHASE_QUALITY = "estimated_phase_quality"
    SHP_COUNTS = "shp_counts"
    WATER_MASK = "water_mask"

    def __str__(self) -> str:
        return self.value


SAME_PER_MINISTACK_DATASETS = [
    SamePerMinistackDataset.TEMPORAL_COHERENCE,
    SamePerMinistackDataset.PHASE_SIMILARITY,
    SamePerMinistackDataset.PERSISTENT_SCATTERER_MASK,
    SamePerMinistackDataset.SHP_COUNTS,
]
UNIQUE_PER_DATE_DATASETS = [
    QualityDataset.TIMESERIES_INVERSION_RESIDUALS,
    QualityDataset.CONNECTED_COMPONENT_LABELS,
    QualityDataset.RECOMMENDED_MASK,
]
