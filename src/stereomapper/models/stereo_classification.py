""" Specifies the Class for Stereo Classification """

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Literal
import numpy as np
from enum import Enum 

class StereoClass(Enum):
    IDENTICAL = "Identical structures"
    IDENTICAL_MISSING_CHARGE = "Identical structures with undetermined charge"
    PROTOMERS = "Protomers"
    INDISTINGUISHABLE = "Indistinguishable structures"
    ENANTIOMERS = "Enantiomers"
    DIASTEREOMERS = "Diastereomers"
    PLANAR_VS_STEREO = "Stereo-resolution pairs"
    NO_CLASSIFICATION = "Unclassified"
    UNRESOLVED = "Unresolved"

RelBin = Literal["high", "medium", "low", "very_low"]

@dataclass
class StereoClassification:
    """Specify the structure of the stereo classification results."""
    classification: str                       # e.g. StereoClass.ENANTIOMERS.value
    rmsd: Optional[float]
    details: Dict[str, Any]

    # New: confidence (all optional for backward-compat)
    confidence_score: Optional[int] = None    # 0..100
    confidence_bin: Optional[RelBin] = None
    confidence: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    # -------- internal helper to unify inputs --------
    @staticmethod
    def _extract_conf_fields(
        stereo_score: Optional[float] = None,
        confidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if confidence:
            # prefer full dict if provided
            out["confidence"] = confidence
            # be robust if keys are missing
            out["confidence_score"] = int(confidence.get("score")) if confidence.get("score") is not None else None
            out["confidence_bin"] = confidence.get("bin")
        elif stereo_score is not None:
            # headline only (legacy path)
            out["confidence_score"] = float(stereo_score)
            out["confidence_bin"] = None
            out["confidence"] = None
        return out
    
    # -------- base factory with confidence plumbing --------
    @classmethod
    def _base(
        cls,
        *,
        classification: "StereoClass",
        rmsd: Optional[float],
        details: Optional[Dict[str, Any]] = None,
        stereo_score: Optional[float] = None,             # headline score (optional)
        confidence: Optional[Dict[str, Any]] = None,      # full breakdown (optional)
        **counts
    ) -> "StereoClassification":
        defaults = dict(
            num_stereogenic_elements=0,
            num_tetra_matches=0,
            num_tetra_flips=0,
            num_db_matches=0,
            num_db_flips=0,
            num_missing=0,
            num_unspecified=0,
        )
        merged_details = {**defaults, **(details or {}), **counts}
        conf_fields = cls._extract_conf_fields(stereo_score=stereo_score, confidence=confidence)

        return cls(
            classification=classification.value,
            rmsd=rmsd,
            details=merged_details,
            **conf_fields,
        )

    ## --------------- Class Methods for each classification --------------- ##
    ############################################################################

    ## --------------- Class Methods for Identical Structures --------------- ##
    @classmethod
    def identical(cls, **kwargs) -> "StereoClassification":
        return cls._base(classification=StereoClass.IDENTICAL, **kwargs)
    
    @classmethod
    def unresolved(cls, **kwargs) -> "StereoClassification":
        return cls._base(classification=StereoClass.UNRESOLVED, **kwargs)
    
    @classmethod
    def identical_missing_charge(cls, **kwargs) -> "StereoClassification":
        return cls._base(classification=StereoClass.IDENTICAL_MISSING_CHARGE, **kwargs)
        
    
    @classmethod
    def protomers(cls, **kwargs) -> "StereoClassification":
        return cls._base(classification=StereoClass.PROTOMERS, **kwargs)
    
    ## --------------- Class Methods for Stereoisomeric Structures --------------- ##
    #################################################################################
    
    @classmethod
    def enantiomers(cls, **kwargs) -> "StereoClassification":
        """Accepts stereo_score=int|float and/or confidence=dict."""
        return cls._base(classification=StereoClass.ENANTIOMERS, **kwargs)
    
    @classmethod
    def diastereomers(cls, **kwargs) -> "StereoClassification":
        return cls._base(classification=StereoClass.DIASTEREOMERS, **kwargs)
    
    @classmethod
    def putative_structures(cls, **kwargs) -> "StereoClassification":
        return cls._base(classification=StereoClass.PUTATIVE, **kwargs)
    
        ## --------------- Class Methods for Ambiguous/Undefined Structures --------------- ##
        #####################################################################################
    
    @classmethod
    def ambiguous_structures(cls, **kwargs) -> "StereoClassification":
        return cls._base(classification=StereoClass.AMBIGUOUS, **kwargs)
    
        ## --------------- Class Methods for Planar vs Stereo Structures --------------- ##
        #####################################################################################
    
    @classmethod
    def planar_vs_stereo(cls, **kwargs) -> "StereoClassification":
        return cls._base(classification=StereoClass.PLANAR_VS_STEREO, **kwargs)

        ## --------------- Class Methods for Indistinguishable Structures --------------- ##
        #####################################################################################
    @classmethod
    def indistinguishable_structures(cls, **kwargs) -> "StereoClassification":
        return cls._base(classification=StereoClass.INDISTINGUISHABLE, **kwargs)

        ## --------------- Class Methods for No Classification --------------- ##
        #####################################################################################
    
    @classmethod
    def no_classification(cls):
        """ Return a StereoClassification instance for no classification """
        return cls._base(
            classification=StereoClass.NO_CLASSIFICATION,
            rmsd=np.nan,
            penalties=[]
        )