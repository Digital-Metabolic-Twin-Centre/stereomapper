"""Feature extraction for confidence scoring."""

from .models import ConfidenceResult
from .confidence import ConfidenceScorer
from stereomapper.models.stereo_elements import StereoCounts

class FeatureBuilder:
    """Build features for confidence scoring."""

    def build_features_for_confidence(
        self,
        assigned_class: str,
        *,
        rmsd: float,
        charge1: int,
        charge2: int,
        counts: StereoCounts,
        tanimoto2d=None,
        ik_first_eq=None,
        ik_stereo_layer_eq=None,
        ik_protonation_layer_eq=None,
    ) -> ConfidenceResult:
        """Build confidence features from analysis results"""

        scorer = ConfidenceScorer()

        return scorer.score(
            assigned_class=assigned_class,
            rmsd=rmsd,
            charge_match=int(charge1 == charge2),
            tanimoto2d=tanimoto2d,
            ik_first_eq=ik_first_eq,
            ik_stereo_layer_eq=ik_stereo_layer_eq,
            ik_protonation_layer_eq=ik_protonation_layer_eq,
            counts=counts
        )
    