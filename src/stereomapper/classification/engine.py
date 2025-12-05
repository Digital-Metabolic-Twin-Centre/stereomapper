"""Engine module for stereochemical relationship analysis."""

from rdkit import Chem
from stereomapper.models.stereo_classification import StereoClassification
from stereomapper.models.stereo_elements import StereoCounts
from stereomapper.domain.models import SimilarityResult
from stereomapper.domain.chemistry import ChemistryOperations, StereoAnalyser, ChemistryUtils
from stereomapper.classification.inchi import InChIFallbackAnalyser
from stereomapper.scoring import ConfidenceScorer
from stereomapper.scoring.features import FeatureBuilder
from stereomapper.utils.logging import setup_logging

logger, summary_logger = setup_logging(
    console=True,
    level="INFO",           # Detailed logging to files
    quiet_console=True,     # Minimal console output during progress bar
    console_level="ERROR"   # Only errors to console
)

# used to satisfy testing requirements
def __getattr__(name):
    if name == "StereochemicalClassifier":
        from stereomapper.classification import StereochemicalClassifier as SC
        return SC
    raise AttributeError(name)


class RelationshipAnalyser:
    """
    Analyzes stereochemical relationships between molecular structures.

    This class provides functionality to compare two molecules and determine
    their stereochemical relationship (e.g., enantiomers, diastereomers, etc.).
    """

    def __init__(self):
        """Initialize the RelationshipAnalyser with classifier and scorer instances."""
        from stereomapper.classification import StereochemicalClassifier
        self.classifier = StereochemicalClassifier()
        self.scorer = ConfidenceScorer()

    def calc_relationship(
            self,
            mol_object1,
            mol_object2,
            charge1,
            charge2,
            cid1,
            cid2,
            isRadio1,
            isRadio2,
            has_sru1,
            has_sru2,
            is_undef_sru1,
            is_undef_sru2,
            sru_repeat_count1,
            sru_repeat_count2,
    ) -> SimilarityResult:
        """
        Calculate the stereochemical relationship between two molecular structures.

        Args:
            mol_object1: First RDKit Mol object.
            mol_object2: Second RDKit Mol object.
            charge1: Charge of the first molecule.
            charge2: Charge of the second molecule.
            cid1: Compound ID of the first molecule.
            cid2: Compound ID of the second molecule.
            isRadio1: Whether the first molecule is radioactive.
            isRadio2: Whether the second molecule is radioactive.
            has_sru1: Whether the first molecule has structural repeat units.
            has_sru2: Whether the second molecule has structural repeat units.
            is_undef_sru1: Whether the first molecule has undefined SRUs.
            is_undef_sru2: Whether the second molecule has undefined SRUs.
            sru_repeat_count1: SRU repeat count for the first molecule.
            sru_repeat_count2: SRU repeat count for the second molecule.

        Returns:
            SimilarityResult: The calculated stereochemical relationship and associated metrics.

        Raises:
            ValueError: If inputs are not valid RDKit Mol objects or
            if stereogenic elements cannot be identified.
        """
        from stereomapper.classification import StereochemicalClassifier
        TAG = "[calc-stereo-similarity]"

        def _to_similarity_result(stereo_class: StereoClassification) -> SimilarityResult:
            """
            Convert StereoClassification to SimilarityResult.

            Args:
                stereo_class: The StereoClassification instance to convert.

            Returns:
                SimilarityResult: The converted result.
            """
            return SimilarityResult.from_stereo_classification(stereo_class)

        # ensure the inputs are valid RDKit Mol objects
        if not isinstance(mol_object1, Chem.Mol) or not isinstance(mol_object2, Chem.Mol):
            logger.error("%s Both inputs must be RDKit Mol objects.", TAG)
            raise ValueError("Both inputs must be RDKit Mol objects.")

        if isRadio1 != isRadio2:
            logger.info("%s One of the molecules is radioactive, cannot compare", TAG)
            res = _to_similarity_result(StereoClassification.no_classification())
            # attach reason via dict for downstream extraction
            return SimilarityResult(
                classification=res.classification,
                rmsd=res.rmsd,
                confidence_score=res.confidence_score,
                confidence_bin=res.confidence_bin,
                confidence=res.confidence,
                details={**(res.details or {}), 'reason': 'Radioactivity mismatch; cannot compare'}
            )

        if has_sru1 != has_sru2:
            logger.info("%s One molecule has SRUs and the other does not; cannot compare", TAG)
            res = _to_similarity_result(StereoClassification.no_classification())
            return SimilarityResult(
                classification=res.classification,
                rmsd=res.rmsd,
                confidence_score=res.confidence_score,
                confidence_bin=res.confidence_bin,
                confidence=res.confidence,
                details={**(res.details or {}), 'reason': 'SRU presence mismatch; cannot compare'}
            )
        if has_sru1 and has_sru2:
            # 2) undefined vs defined
            if is_undef_sru1 != is_undef_sru2:
                logger.info("%s Undefined SRUs mismatch; cannot compare", TAG)
                res = _to_similarity_result(StereoClassification.no_classification())
                return SimilarityResult(
                    classification=res.classification,
                    rmsd=res.rmsd,
                    confidence_score=res.confidence_score,
                    confidence_bin=res.confidence_bin,
                    confidence=res.confidence,
                    details={
                        **(res.details or {}),
                        'reason': 'Undefined SRU state mismatch; cannot compare'
                        }
                )
            # 3) optional: enforce same repeat count when defined
            if ((not is_undef_sru1)
                and (sru_repeat_count1 is not None)
                and (sru_repeat_count2 is not None)
                and (sru_repeat_count1 != sru_repeat_count2)):
                logger.info("%s SRU repeat counts differ; cannot compare", TAG)
                res = _to_similarity_result(StereoClassification.no_classification())
                return SimilarityResult(
                    classification=res.classification,
                    rmsd=res.rmsd,
                    confidence_score=res.confidence_score,
                    confidence_bin=res.confidence_bin,
                    confidence=res.confidence,
                    details={
                        **(res.details or {}),
                        'reason': 'SRU repeat counts differ; cannot compare'
                        }
                )

        # Defensive: if any SRU signal exists but we didn't block, log once to hunt anomalies
        if (has_sru1 or has_sru2) and not (
            (has_sru1 != has_sru2) or
            (has_sru1 and (is_undef_sru1 != is_undef_sru2)) or
            (has_sru1 and (not is_undef_sru1) and
             (sru_repeat_count1 is not None) and
             (sru_repeat_count2 is not None)
            and (sru_repeat_count1 != sru_repeat_count2))
        ):
            logger.debug(
                "%s SRU present but pair passed guard: "
                "has=(%s,%s) undef=(%s,%s) rep=(%s,%s)",
                TAG, has_sru1, has_sru2, is_undef_sru1, is_undef_sru2,
                sru_repeat_count1, sru_repeat_count2
            )

        # align the two molecules
        rmsd_raw = ChemistryOperations.align_molecules(mol_object1, mol_object2, cid1, cid2)
        rmsd, rmsd_err = ChemistryUtils.normalise_rmsd(rmsd_raw)

        if rmsd_err:
            return SimilarityResult(
                classification="RMSD_ERROR",
                rmsd=None,
                details={"error": rmsd_err}
            )

        stereo_elements = StereoAnalyser.compare_stereo_elements(mol_object1, mol_object2)

        # check if the stereogenic elements were found
        if stereo_elements is None:
            logger.error(
                "%s Could not identify stereogenic elements in one or both molecules.",
                TAG
                )
            raise ValueError("Could not identify stereogenic elements in one or both molecules.")

        # extract the relevant information from the stereo_elements dictionary
        counts = StereoCounts.from_stereo_elements(stereo_elements)

        # compute tanimoto2d similarity
        tanimoto2d = ChemistryOperations.fingerprint_tanimoto(mol_object1, mol_object2)
        if not tanimoto2d:
            tanimoto2d = None
            logger.warning(
                "%s Tanimoto similarity could not be computed for %s vs %s",
                TAG,
                cid1,
                cid2
                )
        # extract inchikey layers for scoring with better error handling
        ik_first_eq = None
        ik_stereo_layer_eq = None
        ik_protonation_layer_eq = None

        # Initialize the layer dictionaries
        inchikey_layer1 = None
        inchikey_layer2 = None

        fallback_analyser = InChIFallbackAnalyser()

        try:
            inchikey_layer1 = fallback_analyser._get_inchikey_layers(mol_object1)
        except Exception as e:
            logger.warning("%s InChIKey layer extraction failed for mol1 %s: %s", TAG, cid1, str(e))
            inchikey_layer1 = None  # Set to None instead of setting wrong variables

        try:
            inchikey_layer2 = fallback_analyser._get_inchikey_layers(mol_object2)
        except Exception as e:
            logger.warning("%s InChIKey layer extraction failed for mol2 %s: %s", TAG, cid2, str(e))
            inchikey_layer2 = None  # Set to None instead of setting wrong variables

        # Compare layers individually (only set if both molecules have valid layers)
        if inchikey_layer1 is not None and inchikey_layer2 is not None:
            # extract first layer from dictionaries
            ik_first_eq1 = inchikey_layer1.get('first', None)
            ik_first_eq2 = inchikey_layer2.get('first', None)
            if ik_first_eq1 is not None and ik_first_eq2 is not None:
                ik_first_eq = (ik_first_eq1 == ik_first_eq2)

            ik_stereo_layer_eq1 = inchikey_layer1.get('second', None)
            ik_stereo_layer_eq2 = inchikey_layer2.get('second', None)
            if ik_stereo_layer_eq1 is not None and ik_stereo_layer_eq2 is not None:
                ik_stereo_layer_eq = (ik_stereo_layer_eq1 == ik_stereo_layer_eq2)

            ik_protonation_layer_eq1 = inchikey_layer1.get('third', None)
            ik_protonation_layer_eq2 = inchikey_layer2.get('third', None)
            if ik_protonation_layer_eq1 is not None and ik_protonation_layer_eq2 is not None:
                ik_protonation_layer_eq = (ik_protonation_layer_eq1 == ik_protonation_layer_eq2)

        builder = FeatureBuilder()

        # First explicitly handle molecules without stereogenic elements
        if counts.num_stereogenic_elements == 0:
            if rmsd is None: # shouldn't happen, but just in case
                res = _to_similarity_result(StereoClassification.no_classification())
                return SimilarityResult(
                    classification=res.classification,
                    rmsd=res.rmsd,
                    confidence_score=res.confidence_score,
                    confidence_bin=res.confidence_bin,
                    confidence=res.confidence,
                    details={
                        **(res.details or {}),
                        'reason': 'No stereochemical properties and RMSD error; cannot classify'
                        }
                )
            # RMSD threshold logic to determine identity
            if rmsd is not None:
                if charge1 != charge2:
                    conf = builder.build_features_for_confidence(
                        "PROTOMERS",
                        rmsd=rmsd,
                        charge1=charge1, charge2=charge2,
                        counts=counts,
                        tanimoto2d=tanimoto2d,
                        ik_first_eq=ik_first_eq,
                        ik_stereo_layer_eq=ik_stereo_layer_eq,
                        ik_protonation_layer_eq=ik_protonation_layer_eq,
                    )
                    return _to_similarity_result(StereoClassification.protomers(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=rmsd,
                        **counts.as_classification_kwargs()
                    ))
                elif charge1 is None or charge2 is None:
                    conf = builder.build_features_for_confidence(
                        "IDENTICAL_MISSING_CHARGE",
                        rmsd=rmsd,
                        charge1=charge1, charge2=charge2,
                        counts=counts,
                        tanimoto2d=tanimoto2d,
                        ik_first_eq=ik_first_eq,
                        ik_stereo_layer_eq=ik_stereo_layer_eq,
                        ik_protonation_layer_eq=ik_protonation_layer_eq,
                    )
                    return _to_similarity_result(StereoClassification.identical_missing_charge(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=rmsd,
                        **counts.as_classification_kwargs()
                    ))
                else:
                    conf = builder.build_features_for_confidence(
                        "IDENTICAL",
                        rmsd=rmsd,
                        charge1=charge1, charge2=charge2,
                        counts=counts,
                        tanimoto2d=tanimoto2d,
                        ik_first_eq=ik_first_eq,
                        ik_stereo_layer_eq=ik_stereo_layer_eq,
                        ik_protonation_layer_eq=ik_protonation_layer_eq,
                    )
                    return _to_similarity_result(StereoClassification.unresolved(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=rmsd,
                        **counts.as_classification_kwargs(),
                        details={
                            "reason": """Possible pipeline error -
                            should be no identical relationships in this phase"""
                            }
                    ))
            else:
                return _to_similarity_result(StereoClassification.indistinguishable_structures(rmsd=rmsd))

        num_matches = counts.num_tetra_matches + counts.num_db_matches
        num_flips = counts.num_tetra_flips + counts.num_db_flips

        # identical must include unspecified too, should be consistent
        is_identical = (
            (num_flips == 0) and
            (counts.num_missing == 0) and
            ((num_matches + counts.num_unspecified) == counts.num_stereogenic_elements) # should work
        )

        if is_identical:
            if charge1 != charge2:
                conf = builder.build_features_for_confidence(
                        "PROTOMERS",
                        rmsd=rmsd,
                        charge1=charge1, charge2=charge2,
                        counts=counts,
                        tanimoto2d=tanimoto2d,
                        ik_first_eq=ik_first_eq,
                        ik_stereo_layer_eq=ik_stereo_layer_eq,
                        ik_protonation_layer_eq=ik_protonation_layer_eq,
                    )
                return _to_similarity_result(StereoClassification.protomers(
                    stereo_score=conf.score,
                    confidence=conf.as_dict(),
                    rmsd=rmsd,
                    **counts.as_classification_kwargs()
                ))
            else:
                conf = builder.build_features_for_confidence(
                    "IDENTICAL",
                    rmsd=rmsd,
                    charge1=charge1, charge2=charge2,
                    counts=counts,
                    tanimoto2d=tanimoto2d,
                    ik_first_eq=ik_first_eq,
                    ik_stereo_layer_eq=ik_stereo_layer_eq,
                    ik_protonation_layer_eq=ik_protonation_layer_eq,
                )
                return _to_similarity_result(StereoClassification.unresolved(
                    stereo_score=conf.score,
                    confidence=conf.as_dict(),
                    rmsd=rmsd,
                    penalties=None,
                    **counts.as_classification_kwargs(),
                    details={
                        "reason": """
                        Possible pipeline error -
                        should be no identical relationships in this phase
                        """
                        }
                ))

        else:
            if StereochemicalClassifier.is_enantiomer(stereo_elements):
                if charge1 != charge2:
                    res =  _to_similarity_result(StereoClassification.no_classification())
                    return SimilarityResult(
                        classification=res.classification,
                        rmsd=res.rmsd,
                        confidence_score=res.confidence_score,
                        confidence_bin=res.confidence_bin,
                        confidence=res.confidence,
                        details={
                            **(res.details or {}),
                            'reason': 'Enantiomers must share protonation/charge; no valid class'}
                    )
                else:
                    conf = builder.build_features_for_confidence(
                        "ENANTIOMERS",
                        rmsd=rmsd,
                        charge1=charge1, charge2=charge2,
                        counts=counts,
                        tanimoto2d=tanimoto2d,
                        ik_first_eq=ik_first_eq,
                        ik_stereo_layer_eq=ik_stereo_layer_eq,
                        ik_protonation_layer_eq=ik_protonation_layer_eq,
                    )

                    return _to_similarity_result(StereoClassification.enantiomers(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=rmsd,
                        **counts.as_classification_kwargs()
                    ))

            elif StereochemicalClassifier.is_diastereomer(stereo_elements):
                if charge1 != charge2:
                    # Provide explicit explanation for UI/DB
                    res = _to_similarity_result(StereoClassification.no_classification())
                    return SimilarityResult(
                        classification=res.classification,
                        rmsd=res.rmsd,
                        confidence_score=res.confidence_score,
                        confidence_bin=res.confidence_bin,
                        confidence=res.confidence,
                        details={
                            **(res.details or {}),
                            'reason': 'Diastereomers must share protonation/charge; no valid class'}
                    )
                else:
                    conf = builder.build_features_for_confidence(
                        "DIASTEREOMERS",
                        rmsd=rmsd,
                        charge1=charge1, charge2=charge2,
                        counts=counts,
                        tanimoto2d=tanimoto2d,
                        ik_first_eq=ik_first_eq,
                        ik_stereo_layer_eq=ik_stereo_layer_eq,
                        ik_protonation_layer_eq=ik_protonation_layer_eq,
                    )
                    return _to_similarity_result(StereoClassification.diastereomers(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=rmsd,
                        **counts.as_classification_kwargs()
                    ))
            elif StereochemicalClassifier.is_parent_child(stereo_elements):
                if charge1 != charge2:
                    res = _to_similarity_result(StereoClassification.no_classification())
                    return SimilarityResult(
                        classification=res.classification,
                        rmsd=res.rmsd,
                        confidence_score=res.confidence_score,
                        confidence_bin=res.confidence_bin,
                        confidence=res.confidence,
                        details={
                            **(res.details or {}),
                            'reason': """Parent-child stereochemical relationships must share protonation/charge;
                            no valid class"""}
                    )
                else:
                    conf = builder.build_features_for_confidence(
                        "PLANAR_VS_STEREO",
                        rmsd=rmsd,
                        charge1=charge1, charge2=charge2,
                        counts=counts,
                        tanimoto2d=tanimoto2d,
                        ik_first_eq=ik_first_eq,
                        ik_stereo_layer_eq=ik_stereo_layer_eq,
                        ik_protonation_layer_eq=ik_protonation_layer_eq,
                    )
                    return _to_similarity_result(StereoClassification.planar_vs_stereo(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=rmsd,
                        **counts.as_classification_kwargs()
                    ))
            else:
                return _to_similarity_result(StereoClassification.indistinguishable_structures(rmsd=rmsd))