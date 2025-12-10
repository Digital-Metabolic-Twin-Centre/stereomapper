import pytest
import json
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Set, Tuple, Optional, Any

from stereomapper.comparison.compare import (
    _preload_cluster_members,
    _load_cluster_data,
    _is_valid_primary_result,
    _analyze_pair,
    _process_result,
    compare_cluster_relationships
)


class TestPreloadClusterMembers:
    """Test the _preload_cluster_members function."""
    
    def test_empty_cluster_ids(self):
        """Test with empty cluster IDs list."""
        result = _preload_cluster_members("dummy_path", [])
        assert result == {}
    
    @patch('sqlite3.connect')
    def test_preload_cluster_members_success(self, mock_connect):
        """Test successful preloading of cluster members."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Mock query results
        mock_conn.execute.return_value = [
            ("cluster1", '["mol1", "mol2"]', 2),
            ("cluster2", '["mol3", "mol4", "mol5"]', 3)
        ]
        
        cluster_ids = ["cluster1", "cluster2"]
        result = _preload_cluster_members("test.db", cluster_ids)
        
        expected = {
            "cluster1": {"members_json": '["mol1", "mol2"]', "member_count": 2},
            "cluster2": {"members_json": '["mol3", "mol4", "mol5"]', "member_count": 3}
        }
        
        assert result == expected
        mock_connect.assert_called_once_with("test.db")


class TestLoadClusterData:
    """Test the _load_cluster_data function."""
    
    @patch('stereomapper.comparison.compare.results_repo.preload_processed_pairs')
    @patch('stereomapper.comparison.compare.assemblers.build_mols_for_reps')
    @patch('stereomapper.comparison.compare.results_repo.fetch_cluster_reps_for_inchikey')
    def test_insufficient_reps(self, mock_fetch_reps, mock_build_mols, mock_preload_pairs):
        """Test when there are insufficient cluster representatives."""
        mock_fetch_reps.return_value = ["single_rep"]  # Only one rep
        
        result = _load_cluster_data("test.db", "ABCD", "v1.0")
        
        assert result is None
        mock_fetch_reps.assert_called_once_with("test.db", "ABCD")
        mock_build_mols.assert_not_called()
    
    @patch('stereomapper.comparison.compare.results_repo.preload_processed_pairs')
    @patch('stereomapper.comparison.compare.assemblers.build_mols_for_reps')
    @patch('stereomapper.comparison.compare.results_repo.fetch_cluster_reps_for_inchikey')
    def test_insufficient_clusters_after_build(self, mock_fetch_reps, mock_build_mols, mock_preload_pairs):
        """Test when build_mols_for_reps returns insufficient clusters."""
        mock_fetch_reps.return_value = ["rep1", "rep2"]
        mock_build_mols.return_value = (
            {"cluster1": Mock()},  # Only one cluster
            {"cluster1": "SMILES1"},
            {"cluster1": (0, False)},
            {}  # No fallback candidates
        )
        
        with patch('stereomapper.comparison.compare.logger') as mock_logger:
            result = _load_cluster_data("test.db", "ABCD", "v1.0")
        
        assert result is None
        mock_logger.warning.assert_called_once()
    
    @patch('stereomapper.comparison.compare.results_repo.preload_processed_pairs')
    @patch('stereomapper.comparison.compare.assemblers.build_mols_for_reps')
    @patch('stereomapper.comparison.compare.results_repo.fetch_cluster_reps_for_inchikey')
    def test_successful_load(self, mock_fetch_reps, mock_build_mols, mock_preload_pairs):
        """Test successful cluster data loading."""
        mock_fetch_reps.return_value = ["rep1", "rep2", "rep3"]
        mock_build_mols.return_value = (
            {"cluster1": Mock(), "cluster2": Mock()},
            {"cluster1": "SMILES1", "cluster2": "SMILES2"},
            {"cluster1": (0, False), "cluster2": (1, True)},
            {"cluster3": Mock()}  # Fallback candidate
        )
        mock_preload_pairs.return_value = [("cluster1", "cluster2")]
        
        with patch('stereomapper.comparison.compare.logger') as mock_logger:
            result = _load_cluster_data("test.db", "ABCD", "v1.0")
        
        assert result is not None
        data, cluster_ids, processed_pairs = result
        assert len(cluster_ids) == 3  # cluster1, cluster2, cluster3
        assert processed_pairs == {("cluster1", "cluster2")}


class TestIsValidPrimaryResult:
    """Test the _is_valid_primary_result function."""
    
    def test_none_result(self):
        """Test with None result."""
        assert not _is_valid_primary_result(None)
    
    def test_no_classification_attribute(self):
        """Test with object lacking classification attribute."""
        result = Mock(spec=[])  # Mock without classification attribute
        assert not _is_valid_primary_result(result)
    
    def test_failure_cases(self):
        """Test various failure classification cases."""
        failure_cases = [
            None, "RMSD_ERROR", "RMSD ERROR", "FAILED", "ERROR", "ALIGNMENT_FAILED"
        ]
        
        for failure_case in failure_cases:
            result = Mock()
            result.classification = failure_case
            assert not _is_valid_primary_result(result)
    
    def test_no_classification_case(self):
        """Test the 'No classification' case."""
        result = Mock()
        result.classification = "No classification"
        assert not _is_valid_primary_result(result)
    
    def test_valid_classifications(self):
        """Test valid classification cases."""
        valid_cases = [
            "Enantiomers", "Diastereomers", "Identical structures", 
            "Constitutional isomers", "Different structures"
        ]
        
        for valid_case in valid_cases:
            result = Mock()
            result.classification = valid_case
            assert _is_valid_primary_result(result)
    
    def test_empty_string_classification(self):
        """Test with empty string classification."""
        result = Mock()
        result.classification = ""
        assert not _is_valid_primary_result(result)


class TestAnalyzePair:
    """Test the _analyze_pair function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_analyser = Mock()
        self.mock_fallback_analyser = Mock()
        self.mol_by_cid = {"cluster1": Mock(), "cluster2": Mock()}
        self.props_by_cid = {"cluster1": (0, False), "cluster2": (1, True)}
        self.fallback_candidates = {}
        self.sru_by_cid = {
            "cluster1": {"has_sru": False, "is_undef": False, "rep": None},
            "cluster2": {"has_sru": True, "is_undef": False, "rep": 2}
        }
    
    def test_missing_molecules(self):
        """Test when one or both molecules are missing."""
        mol_by_cid = {"cluster1": None, "cluster2": Mock()}
        
        with patch('stereomapper.comparison.compare.summary_logger'):
            result = _analyze_pair(
                self.mock_analyser, self.mock_fallback_analyser,
                mol_by_cid, self.props_by_cid, self.fallback_candidates,
                self.sru_by_cid, "cluster1", "cluster2"
            )
        
        # Should skip to fallback but fail since mol_a is None
        assert result is None
    
    def test_successful_primary_analysis(self):
        """Test successful primary analysis."""
        # Mock a successful primary result
        mock_primary_result = Mock()
        mock_primary_result.classification = "Enantiomers"
        self.mock_analyser.calc_relationship.return_value = mock_primary_result
        
        with patch('stereomapper.comparison.compare.summary_logger'):
            with patch('stereomapper.comparison.compare._is_valid_primary_result', return_value=True):
                result = _analyze_pair(
                    self.mock_analyser, self.mock_fallback_analyser,
                    self.mol_by_cid, self.props_by_cid, self.fallback_candidates,
                    self.sru_by_cid, "cluster1", "cluster2"
                )
        
        assert result == mock_primary_result
        self.mock_analyser.calc_relationship.assert_called_once()
    
    def test_primary_fails_fallback_succeeds(self):
        """Test when primary analysis fails but fallback succeeds."""
        # Mock failed primary result
        mock_primary_result = Mock()
        mock_primary_result.classification = "No classification"
        self.mock_analyser.calc_relationship.return_value = mock_primary_result
        
        # Mock successful fallback result
        mock_fallback_result = Mock()
        mock_fallback_result.classification = "Constitutional isomers"
        mock_fallback_result.penalties = {}
        self.mock_fallback_analyser.analyze_relationship_fallback.return_value = mock_fallback_result
        
        with patch('stereomapper.comparison.compare.summary_logger'):
            with patch('stereomapper.comparison.compare._is_valid_primary_result', return_value=False):
                result = _analyze_pair(
                    self.mock_analyser, self.mock_fallback_analyser,
                    self.mol_by_cid, self.props_by_cid, self.fallback_candidates,
                    self.sru_by_cid, "cluster1", "cluster2"
                )
        
        assert result == mock_fallback_result
        assert result.penalties['used_fallback_method'] is True
        self.mock_fallback_analyser.analyze_relationship_fallback.assert_called_once()


class TestProcessResult:
    """Test the _process_result function."""
    
    def test_none_result(self):
        """Test processing None result."""
        with patch('stereomapper.comparison.compare.logger'):
            result = _process_result(
                None, "cluster1", "cluster2", "v1.0",
                '["mol1"]', '["mol2"]', 1, 1
            )
        
        assert result is None
    
    def test_missing_classification(self):
        """Test result with missing classification."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"score": 0.8}
        
        with patch('stereomapper.comparison.compare.assemblers._normalise_classification', return_value=None):
            with patch('stereomapper.comparison.compare.logger'):
                result = _process_result(
                    mock_result, "cluster1", "cluster2", "v1.0",
                    '["mol1"]', '["mol2"]', 1, 1
                )
        
        assert result is None
    
    def test_successful_processing(self):
        """Test successful result processing."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"classification": "Enantiomers", "score": 0.9}
        
        normalized = {"classification": "Enantiomers", "score": 0.9}
        details = {"method": "primary", "confidence": "high"}
        
        with patch('stereomapper.comparison.compare.assemblers._normalise_classification', return_value=normalized):
            with patch('stereomapper.comparison.compare.assemblers._coerce_scalar', return_value=0.9):
                with patch('stereomapper.comparison.compare.assemblers._details_from_res', return_value=details):
                    result = _process_result(
                        mock_result, "cluster1", "cluster2", "v1.0",
                        '["mol1"]', '["mol2"]', 1, 1
                    )
        
        assert result is not None
        assert result[0] == "cluster1"  # cid_a
        assert result[1] == "cluster2"  # cid_b
        assert result[6] == "Enantiomers"  # classification
        assert result[7] == 0.9  # score


class TestCompareClusterRelationships:
    """Test the main compare_cluster_relationships function."""
    
    @patch('stereomapper.comparison.compare._load_cluster_data')
    def test_no_data_loaded(self, mock_load_data):
        """Test when no cluster data is loaded."""
        mock_load_data.return_value = None
        mock_logger = Mock()
        
        # Should return early without error
        compare_cluster_relationships(
            results_db_path="test.db",
            inchikey_first="ABCD",
            version_tag="v1.0",
            logger=mock_logger
        )
        
        mock_load_data.assert_called_once()
    
    @patch('stereomapper.comparison.compare.results_repo.batch_insert_cluster_pairs')
    @patch('stereomapper.comparison.compare._preload_cluster_members')
    @patch('stereomapper.comparison.compare.results_repo.preload_cluster_sru')
    @patch('stereomapper.comparison.compare._analyze_pair')
    @patch('stereomapper.comparison.compare._process_result')
    @patch('stereomapper.comparison.compare._load_cluster_data')
    def test_successful_comparison(self, mock_load_data, mock_process_result, 
                                 mock_analyze_pair, mock_preload_sru, 
                                 mock_preload_members, mock_batch_insert):
        """Test successful cluster relationship comparison."""
        # Mock loaded data
        mock_data = (
            {"cluster1": Mock(), "cluster2": Mock()},  # mol_by_cid
            {"cluster1": (0, False), "cluster2": (1, True)},  # props_by_cid
            {},  # fallback_candidates
            ["cluster1", "cluster2"]  # cluster_ids
        )
        mock_load_data.return_value = (mock_data, ["cluster1", "cluster2"], set())
        
        # Mock other dependencies
        mock_preload_sru.return_value = {
            "cluster1": {"has_sru": False, "is_undef": False, "rep": None},
            "cluster2": {"has_sru": False, "is_undef": False, "rep": None}
        }
        mock_preload_members.return_value = {
            "cluster1": {"members_json": '["mol1"]', "member_count": 1},
            "cluster2": {"members_json": '["mol2"]', "member_count": 1}
        }
        
        # Mock analysis and processing
        mock_analyze_pair.return_value = Mock()
        mock_process_result.return_value = ("cluster1", "cluster2", '["mol1"]', '["mol2"]', 
                                          1, 1, "Enantiomers", 0.9, '{}', "v1.0")
        
        mock_logger = Mock()
        
        compare_cluster_relationships(
            results_db_path="test.db",
            inchikey_first="ABCD",
            version_tag="v1.0",
            logger=mock_logger
        )
        
        # Verify the analysis was performed for the pair
        mock_analyze_pair.assert_called_once()
        mock_process_result.assert_called_once()
        mock_batch_insert.assert_called_once()


@pytest.fixture
def sample_cluster_data():
    """Fixture providing sample cluster data for tests."""
    return {
        "mol_by_cid": {"cluster1": Mock(), "cluster2": Mock()},
        "props_by_cid": {"cluster1": (0, False), "cluster2": (1, True)},
        "fallback_candidates": {},
        "sru_by_cid": {
            "cluster1": {"has_sru": False, "is_undef": False, "rep": None},
            "cluster2": {"has_sru": False, "is_undef": False, "rep": None}
        }
    }


@pytest.fixture
def mock_analyzers():
    """Fixture providing mock analyzers."""
    return {
        "primary": Mock(),
        "fallback": Mock()
    }