"""Core domain models that match existing stereomapper patterns."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

@dataclass(frozen=True)
class CacheEntry:
    """Represents a cached molecule entry (matches your cache_repo.get_cached_entry)."""
    molecule_id: int
    smiles: Optional[str]
    inchikey_first: Optional[str]
    error: Optional[str]
    is_undef_sru: int
    is_def_sru: int
    sru_repeat_count: Optional[int]
    namespace: str = "default"
    
    @property
    def has_error(self) -> bool:
        """Check if this entry has an error."""
        return self.error is not None
    
    @property
    def is_valid(self) -> bool:
        """Check if this entry represents a valid molecule."""
        return not self.has_error and self.smiles is not None

@dataclass(frozen=True)
class ProcessingResult:
    """Result from process_and_cache_molecule function."""
    molecule_id: Optional[int]
    smiles: Optional[str] 
    error: Optional[str]
    file_path: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Whether processing was successful."""
        return self.molecule_id is not None and self.error is None

"""Similarity and relationship models."""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class SimilarityResult:
    """Represents the result from calc_stereo_similarity (enhanced to match StereoClassification)."""
    classification: str
    rmsd: Optional[float]
    confidence_score: Optional[int] = None  # 0-100
    confidence_bin: Optional[str] = None    # "high", "medium", "low", "very_low"
    confidence: Optional[Dict[str, Any]] = None  # Full confidence breakdown
    details: Optional[Dict[str, Any]] = None     # Stereo analysis details
    
    @classmethod
    def from_stereo_classification(cls, stereo_class: 'StereoClassification') -> 'SimilarityResult':
        """Create SimilarityResult from StereoClassification object."""
        return cls(
            classification=stereo_class.classification,
            rmsd=stereo_class.rmsd,
            confidence_score=stereo_class.confidence_score,
            confidence_bin=stereo_class.confidence_bin,
            confidence=stereo_class.confidence,
            details=stereo_class.details
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "classification": self.classification,
            "rmsd": self.rmsd,
            "confidence_score": self.confidence_score,
            "confidence_bin": self.confidence_bin,
            "confidence": self.confidence,
            "details": self.details
        }
    
    @property
    def score(self) -> Optional[float]:
        """Extract numeric score for compatibility."""
        if self.confidence_score is not None:
            return float(self.confidence_score) 
        return None

    # Legacy compatibility methods for tuple-based calling code
    def to_legacy_tuple(self) -> tuple:
        """Convert to legacy tuple format (score, interpretation, classification)."""
        score = self.score or 0.0
        interpretation = self.classification  # or could be more detailed
        classification = self.classification
        return (score, interpretation, classification)

@dataclass(frozen=True)
class ClusterData:
    """Represents cluster information (matches your cluster processing)."""
    cluster_id: int
    inchikey_first: str
    identity_key_strict: str
    is_undef_sru: bool
    is_def_sru: bool
    sru_repeat_count: Optional[int]
    member_count: int
    members_json: str
    
    @property
    def has_sru(self) -> bool:
        """Whether this cluster has SRU."""
        return self.is_def_sru or self.is_undef_sru

@dataclass
class PipelineStats:
    """Statistics from pipeline execution (matches your current tracking)."""
    total_files: int = 0
    processed_files: int = 0
    successful_molecules: int = 0
    failed_molecules: int = 0
    unique_inchikeys: int = 0
    clusters_created: int = 0
    relationships_calculated: int = 0
    processing_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate processing success rate."""
        if self.processed_files == 0:
            return 0.0
        return self.successful_molecules / self.processed_files