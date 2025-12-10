import pytest
from dataclasses import FrozenInstanceError
from typing import Dict, Any

from stereomapper.domain.models import (
    CacheEntry,
    ProcessingResult,
    SimilarityResult,
    ClusterData,
    PipelineStats
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test basic CacheEntry creation."""
        entry = CacheEntry(
            molecule_id=123,
            smiles="CCO",
            inchikey_first="LFQSCWFLJHTTHZ",
            error=None,
            is_undef_sru=0,
            is_def_sru=1,
            sru_repeat_count=3,
            namespace="test"
        )
        
        assert entry.molecule_id == 123
        assert entry.smiles == "CCO"
        assert entry.inchikey_first == "LFQSCWFLJHTTHZ"
        assert entry.error is None
        assert entry.is_undef_sru == 0
        assert entry.is_def_sru == 1
        assert entry.sru_repeat_count == 3
        assert entry.namespace == "test"
    
    def test_cache_entry_default_namespace(self):
        """Test CacheEntry with default namespace."""
        entry = CacheEntry(
            molecule_id=123,
            smiles="CCO",
            inchikey_first="LFQSCWFLJHTTHZ",
            error=None,
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None
        )
        
        assert entry.namespace == "default"
    
    def test_cache_entry_frozen(self):
        """Test that CacheEntry is frozen (immutable)."""
        entry = CacheEntry(
            molecule_id=123,
            smiles="CCO",
            inchikey_first="LFQSCWFLJHTTHZ",
            error=None,
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None
        )
        
        with pytest.raises(FrozenInstanceError):
            entry.molecule_id = 456
    
    def test_cache_entry_has_error_property(self):
        """Test has_error property."""
        # Entry without error
        entry_no_error = CacheEntry(
            molecule_id=123,
            smiles="CCO",
            inchikey_first="LFQSCWFLJHTTHZ",
            error=None,
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None
        )
        assert entry_no_error.has_error is False
        
        # Entry with error
        entry_with_error = CacheEntry(
            molecule_id=123,
            smiles=None,
            inchikey_first=None,
            error="Invalid SMILES",
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None
        )
        assert entry_with_error.has_error is True
    
    def test_cache_entry_is_valid_property(self):
        """Test is_valid property."""
        # Valid entry
        valid_entry = CacheEntry(
            molecule_id=123,
            smiles="CCO",
            inchikey_first="LFQSCWFLJHTTHZ",
            error=None,
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None
        )
        assert valid_entry.is_valid is True
        
        # Invalid entry - has error
        invalid_entry_error = CacheEntry(
            molecule_id=123,
            smiles="CCO",
            inchikey_first="LFQSCWFLJHTTHZ",
            error="Some error",
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None
        )
        assert invalid_entry_error.is_valid is False
        
        # Invalid entry - no SMILES
        invalid_entry_no_smiles = CacheEntry(
            molecule_id=123,
            smiles=None,
            inchikey_first="LFQSCWFLJHTTHZ",
            error=None,
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None
        )
        assert invalid_entry_no_smiles.is_valid is False
    
    def test_cache_entry_optional_fields(self):
        """Test CacheEntry with optional fields as None."""
        entry = CacheEntry(
            molecule_id=123,
            smiles=None,
            inchikey_first=None,
            error=None,
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None
        )
        
        assert entry.smiles is None
        assert entry.inchikey_first is None
        assert entry.error is None
        assert entry.sru_repeat_count is None


class TestProcessingResult:
    """Test ProcessingResult dataclass."""
    
    def test_processing_result_creation(self):
        """Test basic ProcessingResult creation."""
        result = ProcessingResult(
            molecule_id=123,
            smiles="CCO",
            error=None,
            file_path="/path/to/file.sdf"
        )
        
        assert result.molecule_id == 123
        assert result.smiles == "CCO"
        assert result.error is None
        assert result.file_path == "/path/to/file.sdf"
    
    def test_processing_result_default_file_path(self):
        """Test ProcessingResult with default file_path."""
        result = ProcessingResult(
            molecule_id=123,
            smiles="CCO",
            error=None
        )
        
        assert result.file_path is None
    
    def test_processing_result_frozen(self):
        """Test that ProcessingResult is frozen (immutable)."""
        result = ProcessingResult(
            molecule_id=123,
            smiles="CCO",
            error=None
        )
        
        with pytest.raises(FrozenInstanceError):
            result.molecule_id = 456
    
    def test_processing_result_success_property(self):
        """Test success property."""
        # Successful result
        success_result = ProcessingResult(
            molecule_id=123,
            smiles="CCO",
            error=None
        )
        assert success_result.success is True
        
        # Failed result - no molecule_id
        failed_result_no_id = ProcessingResult(
            molecule_id=None,
            smiles="CCO",
            error="Processing failed"
        )
        assert failed_result_no_id.success is False
        
        # Failed result - has error
        failed_result_with_error = ProcessingResult(
            molecule_id=123,
            smiles="CCO",
            error="Some error occurred"
        )
        assert failed_result_with_error.success is False
        
        # Failed result - both conditions
        failed_result_both = ProcessingResult(
            molecule_id=None,
            smiles=None,
            error="Failed to process"
        )
        assert failed_result_both.success is False


class TestSimilarityResult:
    """Test SimilarityResult dataclass."""
    
    def test_similarity_result_creation(self):
        """Test basic SimilarityResult creation."""
        result = SimilarityResult(
            classification="similar",
            rmsd=0.5,
            confidence_score=85,
            confidence_bin="high",
            confidence={"level": "high", "score": 0.85},
            details={"method": "fingerprint"}
        )
        
        assert result.classification == "similar"
        assert result.rmsd == 0.5
        assert result.confidence_score == 85
        assert result.confidence_bin == "high"
        assert result.confidence == {"level": "high", "score": 0.85}
        assert result.details == {"method": "fingerprint"}
    
    def test_similarity_result_minimal_creation(self):
        """Test SimilarityResult with only required fields."""
        result = SimilarityResult(
            classification="dissimilar",
            rmsd=None
        )
        
        assert result.classification == "dissimilar"
        assert result.rmsd is None
        assert result.confidence_score is None
        assert result.confidence_bin is None
        assert result.confidence is None
        assert result.details is None
    
    def test_similarity_result_frozen(self):
        """Test that SimilarityResult is frozen (immutable)."""
        result = SimilarityResult(
            classification="similar",
            rmsd=0.5
        )
        
        with pytest.raises(FrozenInstanceError):
            result.classification = "dissimilar"
    
    def test_similarity_result_score_property(self):
        """Test score property."""
        # With confidence_score
        result_with_score = SimilarityResult(
            classification="similar",
            rmsd=0.5,
            confidence_score=85
        )
        assert result_with_score.score == 85.0
        
        # Without confidence_score
        result_no_score = SimilarityResult(
            classification="similar",
            rmsd=0.5
        )
        assert result_no_score.score is None
    
    def test_similarity_result_to_dict(self):
        """Test to_dict method."""
        result = SimilarityResult(
            classification="similar",
            rmsd=0.5,
            confidence_score=85,
            confidence_bin="high",
            confidence={"level": "high"},
            details={"method": "test"}
        )
        
        expected_dict = {
            "classification": "similar",
            "rmsd": 0.5,
            "confidence_score": 85,
            "confidence_bin": "high",
            "confidence": {"level": "high"},
            "details": {"method": "test"}
        }
        
        assert result.to_dict() == expected_dict
    
    def test_similarity_result_to_dict_with_nones(self):
        """Test to_dict method with None values."""
        result = SimilarityResult(
            classification="dissimilar",
            rmsd=None
        )
        
        expected_dict = {
            "classification": "dissimilar",
            "rmsd": None,
            "confidence_score": None,
            "confidence_bin": None,
            "confidence": None,
            "details": None
        }
        
        assert result.to_dict() == expected_dict
    
    def test_similarity_result_to_legacy_tuple(self):
        """Test to_legacy_tuple method."""
        # With confidence_score
        result_with_score = SimilarityResult(
            classification="similar",
            rmsd=0.5,
            confidence_score=85
        )
        legacy_tuple = result_with_score.to_legacy_tuple()
        assert legacy_tuple == (85.0, "similar", "similar")
        
        # Without confidence_score
        result_no_score = SimilarityResult(
            classification="dissimilar",
            rmsd=1.5
        )
        legacy_tuple = result_no_score.to_legacy_tuple()
        assert legacy_tuple == (0.0, "dissimilar", "dissimilar")
    
    def test_similarity_result_from_stereo_classification(self):
        """Test from_stereo_classification class method."""
        # Mock StereoClassification object
        class MockStereoClassification:
            def __init__(self):
                self.classification = "similar"
                self.rmsd = 0.3
                self.confidence_score = 90
                self.confidence_bin = "high"
                self.confidence = {"level": "high", "score": 0.9}
                self.details = {"stereo_analysis": "complete"}
        
        mock_stereo = MockStereoClassification()
        result = SimilarityResult.from_stereo_classification(mock_stereo)
        
        assert result.classification == "similar"
        assert result.rmsd == 0.3
        assert result.confidence_score == 90
        assert result.confidence_bin == "high"
        assert result.confidence == {"level": "high", "score": 0.9}
        assert result.details == {"stereo_analysis": "complete"}


class TestClusterData:
    """Test ClusterData dataclass."""
    
    def test_cluster_data_creation(self):
        """Test basic ClusterData creation."""
        cluster = ClusterData(
            cluster_id=1,
            inchikey_first="LFQSCWFLJHTTHZ",
            identity_key_strict="ethanol_strict",
            is_undef_sru=False,
            is_def_sru=True,
            sru_repeat_count=5,
            member_count=3,
            members_json='["mol1", "mol2", "mol3"]'
        )
        
        assert cluster.cluster_id == 1
        assert cluster.inchikey_first == "LFQSCWFLJHTTHZ"
        assert cluster.identity_key_strict == "ethanol_strict"
        assert cluster.is_undef_sru is False
        assert cluster.is_def_sru is True
        assert cluster.sru_repeat_count == 5
        assert cluster.member_count == 3
        assert cluster.members_json == '["mol1", "mol2", "mol3"]'
    
    def test_cluster_data_frozen(self):
        """Test that ClusterData is frozen (immutable)."""
        cluster = ClusterData(
            cluster_id=1,
            inchikey_first="LFQSCWFLJHTTHZ",
            identity_key_strict="ethanol_strict",
            is_undef_sru=False,
            is_def_sru=False,
            sru_repeat_count=None,
            member_count=1,
            members_json='["mol1"]'
        )
        
        with pytest.raises(FrozenInstanceError):
            cluster.cluster_id = 2
    
    def test_cluster_data_has_sru_property(self):
        """Test has_sru property."""
        # No SRU
        no_sru_cluster = ClusterData(
            cluster_id=1,
            inchikey_first="LFQSCWFLJHTTHZ",
            identity_key_strict="ethanol_strict",
            is_undef_sru=False,
            is_def_sru=False,
            sru_repeat_count=None,
            member_count=1,
            members_json='["mol1"]'
        )
        assert no_sru_cluster.has_sru is False
        
        # Undefined SRU
        undef_sru_cluster = ClusterData(
            cluster_id=2,
            inchikey_first="HGBOYTHUEUWSSQ",
            identity_key_strict="ethylamine_strict",
            is_undef_sru=True,
            is_def_sru=False,
            sru_repeat_count=None,
            member_count=1,
            members_json='["mol2"]'
        )
        assert undef_sru_cluster.has_sru is True
        
        # Defined SRU
        def_sru_cluster = ClusterData(
            cluster_id=3,
            inchikey_first="ATUOYWHBWRKTHZ",
            identity_key_strict="propane_strict",
            is_undef_sru=False,
            is_def_sru=True,
            sru_repeat_count=3,
            member_count=1,
            members_json='["mol3"]'
        )
        assert def_sru_cluster.has_sru is True
        
        # Both SRU flags (edge case)
        both_sru_cluster = ClusterData(
            cluster_id=4,
            inchikey_first="SOMEOTHER",
            identity_key_strict="test_strict",
            is_undef_sru=True,
            is_def_sru=True,
            sru_repeat_count=2,
            member_count=1,
            members_json='["mol4"]'
        )
        assert both_sru_cluster.has_sru is True
    
    def test_cluster_data_optional_sru_repeat_count(self):
        """Test with None sru_repeat_count."""
        cluster = ClusterData(
            cluster_id=1,
            inchikey_first="LFQSCWFLJHTTHZ",
            identity_key_strict="ethanol_strict",
            is_undef_sru=True,
            is_def_sru=False,
            sru_repeat_count=None,
            member_count=1,
            members_json='["mol1"]'
        )
        
        assert cluster.sru_repeat_count is None
        assert cluster.has_sru is True


class TestPipelineStats:
    """Test PipelineStats dataclass."""
    
    def test_pipeline_stats_creation(self):
        """Test basic PipelineStats creation."""
        stats = PipelineStats(
            total_files=100,
            processed_files=95,
            successful_molecules=90,
            failed_molecules=5,
            unique_inchikeys=85,
            clusters_created=80,
            relationships_calculated=500,
            processing_time=120.5
        )
        
        assert stats.total_files == 100
        assert stats.processed_files == 95
        assert stats.successful_molecules == 90
        assert stats.failed_molecules == 5
        assert stats.unique_inchikeys == 85
        assert stats.clusters_created == 80
        assert stats.relationships_calculated == 500
        assert stats.processing_time == 120.5
    
    def test_pipeline_stats_default_values(self):
        """Test PipelineStats with default values."""
        stats = PipelineStats()
        
        assert stats.total_files == 0
        assert stats.processed_files == 0
        assert stats.successful_molecules == 0
        assert stats.failed_molecules == 0
        assert stats.unique_inchikeys == 0
        assert stats.clusters_created == 0
        assert stats.relationships_calculated == 0
        assert stats.processing_time == 0.0
    
    def test_pipeline_stats_mutable(self):
        """Test that PipelineStats is mutable (not frozen)."""
        stats = PipelineStats()
        
        # Should be able to modify values
        stats.total_files = 10
        stats.processed_files = 8
        stats.successful_molecules = 7
        
        assert stats.total_files == 10
        assert stats.processed_files == 8
        assert stats.successful_molecules == 7
    
    def test_pipeline_stats_success_rate_property(self):
        """Test success_rate property."""
        # Normal case
        stats = PipelineStats(
            processed_files=100,
            successful_molecules=85
        )
        assert stats.success_rate == 0.85
        
        # Zero processed files (edge case)
        empty_stats = PipelineStats()
        assert empty_stats.success_rate == 0.0
        
        # Perfect success rate
        perfect_stats = PipelineStats(
            processed_files=50,
            successful_molecules=50
        )
        assert perfect_stats.success_rate == 1.0
        
        # Zero success rate
        failed_stats = PipelineStats(
            processed_files=10,
            successful_molecules=0
        )
        assert failed_stats.success_rate == 0.0
    
    def test_pipeline_stats_incremental_updates(self):
        """Test incremental updates to stats."""
        stats = PipelineStats()
        
        # Simulate processing pipeline
        stats.total_files = 5
        
        # Process first file
        stats.processed_files += 1
        stats.successful_molecules += 1
        
        # Process second file (failed)
        stats.processed_files += 1
        stats.failed_molecules += 1
        
        # Process third file
        stats.processed_files += 1
        stats.successful_molecules += 1
        stats.unique_inchikeys += 1
        stats.clusters_created += 1
        
        assert stats.total_files == 5
        assert stats.processed_files == 3
        assert stats.successful_molecules == 2
        assert stats.failed_molecules == 1
        assert stats.unique_inchikeys == 1
        assert stats.clusters_created == 1
        assert stats.success_rate == 2/3


class TestModelIntegration:
    """Test integration between different models."""
    
    def test_cache_entry_to_processing_result_workflow(self):
        """Test workflow from CacheEntry to ProcessingResult."""
        # Valid cache entry
        cache_entry = CacheEntry(
            molecule_id=123,
            smiles="CCO",
            inchikey_first="LFQSCWFLJHTTHZ",
            error=None,
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None
        )
        
        # Convert to processing result (simulate successful processing)
        if cache_entry.is_valid:
            processing_result = ProcessingResult(
                molecule_id=cache_entry.molecule_id,
                smiles=cache_entry.smiles,
                error=None,
                file_path="/processed/123.sdf"
            )
        else:
            processing_result = ProcessingResult(
                molecule_id=None,
                smiles=None,
                error="Invalid cache entry"
            )
        
        assert processing_result.success is True
        assert processing_result.molecule_id == 123
        assert processing_result.smiles == "CCO"
    
    def test_similarity_result_data_consistency(self):
        """Test data consistency in SimilarityResult methods."""
        result = SimilarityResult(
            classification="similar",
            rmsd=0.3,
            confidence_score=88,
            confidence_bin="high",
            confidence={"detailed": "analysis"},
            details={"method": "stereo_comparison"}
        )
        
        # Check consistency between different representations
        dict_repr = result.to_dict()
        legacy_tuple = result.to_legacy_tuple()
        
        assert dict_repr["classification"] == result.classification
        assert dict_repr["confidence_score"] == result.confidence_score
        assert legacy_tuple[0] == result.score
        assert legacy_tuple[2] == result.classification
    
    def test_cluster_data_sru_logic_consistency(self):
        """Test SRU logic consistency in ClusterData."""
        test_cases = [
            # (is_undef_sru, is_def_sru, expected_has_sru)
            (False, False, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),  # Edge case - both flags set
        ]
        
        for is_undef, is_def, expected_has_sru in test_cases:
            cluster = ClusterData(
                cluster_id=1,
                inchikey_first="TEST",
                identity_key_strict="test",
                is_undef_sru=is_undef,
                is_def_sru=is_def,
                sru_repeat_count=None,
                member_count=1,
                members_json='["test"]'
            )
            
            assert cluster.has_sru == expected_has_sru
    
    def test_pipeline_stats_calculation_accuracy(self):
        """Test calculation accuracy in PipelineStats."""
        stats = PipelineStats(
            total_files=1000,
            processed_files=950,
            successful_molecules=900,
            failed_molecules=50,
            unique_inchikeys=850,
            clusters_created=800,
            relationships_calculated=5000
        )
        
        # Verify internal consistency
        assert stats.successful_molecules + stats.failed_molecules == stats.processed_files
        assert stats.success_rate == 900 / 950
        assert abs(stats.success_rate - 0.9473684210526315) < 1e-10  # Check precision
    
    def test_frozen_vs_mutable_behavior(self):
        """Test the different mutability behaviors of models."""
        # Frozen models should raise FrozenInstanceError
        cache_entry = CacheEntry(
            molecule_id=1, smiles="C", inchikey_first="TEST",
            error=None, is_undef_sru=0, is_def_sru=0, sru_repeat_count=None
        )
        
        processing_result = ProcessingResult(
            molecule_id=1, smiles="C", error=None
        )
        
        similarity_result = SimilarityResult(
            classification="similar", rmsd=0.1
        )
        
        cluster_data = ClusterData(
            cluster_id=1, inchikey_first="TEST", identity_key_strict="test",
            is_undef_sru=False, is_def_sru=False, sru_repeat_count=None,
            member_count=1, members_json='["test"]'
        )
        
        # These should all raise FrozenInstanceError
        frozen_models = [cache_entry, processing_result, similarity_result, cluster_data]
        
        for model in frozen_models:
            with pytest.raises(FrozenInstanceError):
                # Try to modify the first attribute (they all have different first attributes)
                if hasattr(model, 'molecule_id'):
                    model.molecule_id = 999
                elif hasattr(model, 'classification'):
                    model.classification = "different"
                elif hasattr(model, 'cluster_id'):
                    model.cluster_id = 999
        
        # PipelineStats should be mutable
        stats = PipelineStats()
        stats.total_files = 100  # Should not raise an error
        assert stats.total_files == 100