import pytest
from typing import Dict, Any

from stereomapper.classification.classifier import StereochemicalClassifier


class TestStereochemicalClassifier:
    """Test cases for StereochemicalClassifier methods."""
    
    def test_is_enantiomer_valid_case(self):
        """Test is_enantiomer with valid enantiomer data."""
        stereo_elements = {
            'total_tetra': 2,
            'tetra_matches': 0,
            'tetra_flips': 2,
            'total_db': 1,
            'db_matches': 1,
            'db_flips': 0,
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_enantiomer(stereo_elements) == True
    
    def test_is_enantiomer_no_tetra_centers(self):
        """Test is_enantiomer returns False when no tetrahedral centers."""
        stereo_elements = {
            'total_tetra': 0,
            'tetra_matches': 0,
            'tetra_flips': 0,
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_enantiomer(stereo_elements) == False
    
    def test_is_enantiomer_with_missing_centres(self):
        """Test is_enantiomer returns False when missing centres exist."""
        stereo_elements = {
            'total_tetra': 2,
            'tetra_matches': 0,
            'tetra_flips': 2,
            'missing_centres': 1,
            'unspecified': 0
        }
        assert StereochemicalClassifier.is_enantiomer(stereo_elements) == False
    
    def test_is_enantiomer_with_unspecified(self):
        """Test is_enantiomer returns False when unspecified centers exist."""
        stereo_elements = {
            'total_tetra': 2,
            'tetra_matches': 0,
            'tetra_flips': 2,
            'unspecified': 1,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_enantiomer(stereo_elements) == False
    
    def test_is_enantiomer_with_db_flips(self):
        """Test is_enantiomer returns False when DB flips exist."""
        stereo_elements = {
            'total_tetra': 2,
            'tetra_matches': 0,
            'tetra_flips': 2,
            'total_db': 1,
            'db_flips': 1,
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_enantiomer(stereo_elements) == False
    
    def test_is_enantiomer_with_tetra_matches(self):
        """Test is_enantiomer returns False when tetrahedral matches exist."""
        stereo_elements = {
            'total_tetra': 2,
            'tetra_matches': 1,
            'tetra_flips': 1,
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_enantiomer(stereo_elements) == False
    
    def test_is_enantiomer_incomplete_flips(self):
        """Test is_enantiomer returns False when not all tetra centers are flipped."""
        stereo_elements = {
            'total_tetra': 3,
            'tetra_matches': 0,
            'tetra_flips': 2,  # Not all flipped
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_enantiomer(stereo_elements) == False
    
    def test_is_diastereomer_valid_tetra_case(self):
        """Test is_diastereomer with valid diastereomer data."""
        stereo_elements = {
            'tetra_matches': 1,
            'tetra_flips': 1,
            'db_matches': 0,
            'db_flips': 0,
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_diastereomer(stereo_elements) == True
    
    def test_is_diastereomer_valid_db_case(self):
        """Test is_diastereomer with valid diastereomer data involving double bonds."""
        stereo_elements = {
            'tetra_matches': 0,
            'tetra_flips': 0,
            'db_matches': 0,
            'db_flips': 1,
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_diastereomer(stereo_elements) == True

    def test_is_diastereomer_valid_mixed_stereo_case(self):
        """Test is_diastereomer with valid diastereomer data involving both tetrahedral and double bond changes."""
        stereo_elements = {
            'tetra_matches': 1,
            'tetra_flips': 1,
            'db_matches': 1,
            'db_flips': 1,
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_diastereomer(stereo_elements) == True
    
    def test_is_diastereomer_with_unspecified(self):
        """Test is_diastereomer returns False when unspecified centers exist."""
        stereo_elements = {
            'tetra_matches': 1,
            'tetra_flips': 1,
            'unspecified': 1,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_diastereomer(stereo_elements) == False
    
    def test_is_diastereomer_with_missing(self):
        """Test is_diastereomer returns False when missing centers exist."""
        stereo_elements = {
            'tetra_matches': 1,
            'tetra_flips': 1,
            'unspecified': 0,
            'missing_centres': 1
        }
        assert StereochemicalClassifier.is_diastereomer(stereo_elements) == False
    
    def test_is_diastereomer_only_matches(self):
        """Test is_diastereomer with only matches (no flips)."""
        stereo_elements = {
            'tetra_matches': 2,
            'tetra_flips': 0,
            'db_matches': 0,
            'db_flips': 0,
            'unspecified': 0,
            'missing_centres': 0
        }
        # should be False, if no inversions ==> False, not a diastereomer, needs at least one flip
        assert StereochemicalClassifier.is_diastereomer(stereo_elements) == False
        
    def test_default_values_handling(self):
        """Test that methods handle missing keys with default values."""
        empty_stereo_elements = {}
        
        assert StereochemicalClassifier.is_enantiomer(empty_stereo_elements) == False
        assert StereochemicalClassifier.is_diastereomer(empty_stereo_elements) == False
        assert StereochemicalClassifier.is_parent_child(empty_stereo_elements) == False
    
    def test_enantiomer_with_invariants(self):
        """Test is_enantiomer with optional invariant checks."""
        stereo_elements = {
            'total_tetra': 2,
            'tetra_matches': 0,
            'tetra_flips': 2,
            'tetra_missing': 0,  # Optional invariant field
            'total_db': 1,
            'db_matches': 1,
            'db_flips': 0,
            'db_missing': 0,  # Optional invariant field
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_enantiomer(stereo_elements) == True

    def test_parent_child_no_stereo(self):
        """Test is_parent_child returns False when no stereochemical elements exist."""
        stereo_elements = {
            'tetra_matches': 0,
            'tetra_flips': 0,
            'db_matches': 0,
            'db_flips': 0,
            'unspecified': 0,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_parent_child(stereo_elements) == False

    def test_parent_child_with_stereo(self):
        """Test is_parent_child returns True when stereochemical elements exist."""
        stereo_elements = {
            'total_stereo': 3,
            'missing_centres': 2
        }
        assert StereochemicalClassifier.is_parent_child(stereo_elements) == True

    def test_parent_child_all_matched(self):
        """Test is_parent_child returns False when all stereochemical elements match."""
        stereo_elements = {
            'total_stereo': 2,
            'missing_centres': 0
        }
        assert StereochemicalClassifier.is_parent_child(stereo_elements) == False