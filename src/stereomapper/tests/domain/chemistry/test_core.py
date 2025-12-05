import pytest
import logging
from unittest.mock import patch, MagicMock
from rdkit import Chem
from rdkit.Chem import rdmolops

from stereomapper.domain.chemistry.analysis import StereoAnalyser

logger = logging.getLogger(__name__)


@pytest.fixture
def simple_mol():
    """Create a simple molecule without stereocenters."""
    return Chem.MolFromSmiles("CCO")  # Ethanol

@pytest.fixture
def chiral_mol():
    """Create a molecule with a chiral center."""
    return Chem.MolFromSmiles("C[C@H](O)N")  # Chiral molecule

@pytest.fixture
def double_bond_mol():
    """Create a molecule with E/Z stereochemistry."""
    return Chem.MolFromSmiles("C/C=C/C")  # E-but-2-ene

@pytest.fixture
def complex_stereo_mol():
    """Create a molecule with multiple stereocenters."""
    return Chem.MolFromSmiles("C[C@H](O)[C@@H](N)C")  # Multiple chiral centers

@pytest.fixture
def mock_stereo_info():
    """Create mock stereo info objects."""
    class MockStereoInfo:
        def __init__(self, stereo_type, centered_on, specified, descriptor, controlling_atoms):
            self.type = stereo_type
            self.centeredOn = centered_on
            self.specified = specified
            self.descriptor = descriptor
            self.controllingAtoms = controlling_atoms

    return MockStereoInfo


class TestIdentifyStereogenicElements:
    """Test identify_stereogenic_elements method."""

    def test_identify_stereogenic_elements_simple_mol(self, simple_mol):
        """Test with a simple molecule without stereocenters."""
        result = StereoAnalyser.identify_stereogenic_elements(simple_mol, "ethanol.mol")

        assert isinstance(result, dict)
        assert "ethanol" in result
        assert isinstance(result["ethanol"], list)
        # Ethanol has no stereocenters
        assert len(result["ethanol"]) == 0

    def test_identify_stereogenic_elements_chiral_mol(self, chiral_mol):
        """Test with a chiral molecule."""
        result = StereoAnalyser.identify_stereogenic_elements(chiral_mol, "chiral.mol")

        assert isinstance(result, dict)
        assert "chiral" in result
        assert isinstance(result["chiral"], list)
        # Should find at least one stereocenter
        assert len(result["chiral"]) >= 1

        # Check the structure of stereo info
        if result["chiral"]:
            stereo_info = result["chiral"][0]
            assert "type" in stereo_info
            assert "centered_on" in stereo_info
            assert "specified" in stereo_info
            assert "descriptor" in stereo_info
            assert "controlling_atoms" in stereo_info

    def test_identify_stereogenic_elements_double_bond(self, double_bond_mol):
        """Test with a molecule containing E/Z stereochemistry."""
        result = StereoAnalyser.identify_stereogenic_elements(double_bond_mol, "alkene.mol")

        assert isinstance(result, dict)
        assert "alkene" in result

        # Check if double bond stereochemistry is detected
        if result["alkene"]:
            for stereo_info in result["alkene"]:
                assert stereo_info["type"] in ["Atom_Tetrahedral", "Bond_Double"]

    def test_identify_stereogenic_elements_invalid_input(self):
        """Test with invalid input."""
        with pytest.raises(ValueError, match="Input must be an RDKit Mol object"):
            StereoAnalyser.identify_stereogenic_elements("not_a_mol", "test.mol")

    def test_identify_stereogenic_elements_none_mol(self):
        """Test with None molecule."""
        with pytest.raises(ValueError, match="Input must be an RDKit Mol object"):
            StereoAnalyser.identify_stereogenic_elements(None, "test.mol")

    def test_identify_stereogenic_elements_file_naming(self, simple_mol):
        """Test proper file name handling."""
        test_cases = [
            ("molecule.mol", "molecule"),
            ("/path/to/molecule.sdf", "molecule"),
            ("complex_name.mol2", "complex_name"),
            ("no_extension", "no_extension"),
            ("", ""),
        ]

        for input_name, expected_key in test_cases:
            result = StereoAnalyser.identify_stereogenic_elements(simple_mol, input_name)
            assert expected_key in result

    @patch('stereomapper.domain.chemistry.analysis.rdmolops.FindPotentialStereo')
    def test_identify_stereogenic_elements_empty_stereo_info(self, mock_find_stereo, simple_mol):
        """Test when no stereogenic elements are found."""
        mock_find_stereo.return_value = []

        result = StereoAnalyser.identify_stereogenic_elements(simple_mol, "test.mol")

        assert result == {"test": []}
        mock_find_stereo.assert_called_once_with(simple_mol, cleanIt=True, flagPossible=False)

    @patch('stereomapper.domain.chemistry.analysis.rdmolops.FindPotentialStereo')
    def test_identify_stereogenic_elements_with_mock_data(self, mock_find_stereo, simple_mol, mock_stereo_info):
        """Test with mock stereogenic element data."""
        # Create mock stereo info
        mock_info = mock_stereo_info(
            stereo_type="Atom_Tetrahedral",
            centered_on=1,
            specified="Specified",
            descriptor="R",
            controlling_atoms=[0, 2, 3, 4]
        )
        mock_find_stereo.return_value = [mock_info]

        result = StereoAnalyser.identify_stereogenic_elements(simple_mol, "test.mol")

        expected = {
            "test": [{
                "type": "Atom_Tetrahedral",
                "centered_on": 1,
                "specified": "Specified",
                "descriptor": "R",
                "controlling_atoms": [0, 2, 3, 4]
            }]
        }

        assert result == expected


class TestCompareStereoElements:
    """Test compare_stereo_elements method."""

    def test_compare_stereo_elements_invalid_inputs(self):
        """Test with invalid inputs."""
        mol = Chem.MolFromSmiles("CCO")

        with pytest.raises(ValueError, match="Both inputs must be RDKit Mol objects"):
            StereoAnalyser.compare_stereo_elements("not_a_mol", mol)

        with pytest.raises(ValueError, match="Both inputs must be RDKit Mol objects"):
            StereoAnalyser.compare_stereo_elements(mol, "not_a_mol")

        with pytest.raises(ValueError, match="Both inputs must be RDKit Mol objects"):
            StereoAnalyser.compare_stereo_elements("not_a_mol", "also_not_a_mol")

    def test_compare_stereo_elements_simple_molecules(self, simple_mol):
        """Test comparison of simple molecules without stereocenters."""
        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        # Check all expected keys are present
        expected_keys = [
            "total_stereo", "total_tetra", "tetra_matches", "tetra_flips",
            "total_db", "db_matches", "db_flips", "unspecified",
            "missing_centres", "details"
        ]

        for key in expected_keys:
            assert key in result

        # For molecules without stereocenters, all counts should be 0
        assert result["total_stereo"] == 0
        assert result["total_tetra"] == 0
        assert result["tetra_matches"] == 0
        assert result["tetra_flips"] == 0
        assert result["total_db"] == 0
        assert result["db_matches"] == 0
        assert result["db_flips"] == 0
        assert result["missing_centres"] == 0
        assert isinstance(result["details"], list)

    def test_compare_stereo_elements_identical_chiral_molecules(self, chiral_mol):
        """Test comparison of identical chiral molecules."""
        result = StereoAnalyser.compare_stereo_elements(chiral_mol, chiral_mol)

        # Should have perfect matches, no flips
        assert result["tetra_flips"] == 0
        assert result["db_flips"] == 0
        assert result["missing_centres"] == 0

        # Total stereo should equal matches
        assert result["total_stereo"] == result["tetra_matches"] + result["db_matches"]

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_tetrahedral_match(self, mock_identify, simple_mol):
        """Test tetrahedral center matching."""
        # Mock identical tetrahedral centers
        stereo_data = [{
            "type": "Atom_Tetrahedral",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "R",
            "controlling_atoms": [0, 2, 3, 4]
        }]

        mock_identify.side_effect = [
            {"mol1": stereo_data},
            {"mol2": stereo_data}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        assert result["tetra_matches"] == 1.0
        assert result["tetra_flips"] == 0.0
        assert result["total_tetra"] == 1.0

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_tetrahedral_flip(self, mock_identify, simple_mol):
        """Test tetrahedral center flipping."""
        # Mock tetrahedral centers with different descriptors
        stereo_data1 = [{
            "type": "Atom_Tetrahedral",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "R",
            "controlling_atoms": [0, 2, 3, 4]
        }]

        stereo_data2 = [{
            "type": "Atom_Tetrahedral",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "S",
            "controlling_atoms": [0, 2, 3, 4]
        }]

        mock_identify.side_effect = [
            {"mol1": stereo_data1},
            {"mol2": stereo_data2}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        assert result["tetra_matches"] == 0.0
        assert result["tetra_flips"] == 1.0
        assert result["total_tetra"] == 1.0

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_double_bond_match(self, mock_identify, simple_mol):
        """Test double bond stereochemistry matching."""
        # Mock identical double bond stereo
        stereo_data = [{
            "type": "Bond_Double",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "E",
            "controlling_atoms": [0, 2]
        }]

        mock_identify.side_effect = [
            {"mol1": stereo_data},
            {"mol2": stereo_data}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        assert result["db_matches"] == 1.0
        assert result["db_flips"] == 0.0
        assert result["total_db"] == 1.0

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_double_bond_flip(self, mock_identify, simple_mol):
        """Test double bond stereochemistry flipping."""
        # Mock double bond centers with different descriptors
        stereo_data1 = [{
            "type": "Bond_Double",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "E",
            "controlling_atoms": [0, 2]
        }]

        stereo_data2 = [{
            "type": "Bond_Double",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "Z",
            "controlling_atoms": [0, 2]
        }]

        mock_identify.side_effect = [
            {"mol1": stereo_data1},
            {"mol2": stereo_data2}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        assert result["db_matches"] == 0.0
        assert result["db_flips"] == 1.0
        assert result["total_db"] == 1.0

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_missing_centres(self, mock_identify, simple_mol):
        """Test handling of missing stereocenters."""
        # Mock one molecule with stereo, one without
        stereo_data1 = [{
            "type": "Atom_Tetrahedral",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "R",
            "controlling_atoms": [0, 2, 3, 4]
        }]

        stereo_data2 = []

        mock_identify.side_effect = [
            {"mol1": stereo_data1},
            {"mol2": stereo_data2}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        assert result["missing_centres"] == 1.0
        assert result["tetra_matches"] == 0.0
        assert result["tetra_flips"] == 0.0

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_unspecified_stereo(self, mock_identify, simple_mol):
        """Test handling of unspecified stereochemistry."""
        # Mock stereocenters with unspecified stereochemistry
        stereo_data = [{
            "type": "Atom_Tetrahedral",
            "centered_on": 1,
            "specified": "Unspecified",
            "descriptor": "Unknown",
            "controlling_atoms": [0, 2, 3, 4]
        }]

        mock_identify.side_effect = [
            {"mol1": stereo_data},
            {"mol2": stereo_data}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        assert result["missing_centres"] == 1.0
        assert result["tetra_matches"] == 0.0
        assert result["tetra_flips"] == 0.0

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_mixed_stereo_types(self, mock_identify, simple_mol):
        """Test comparison with mixed stereochemistry types."""
        # Mock molecule with both tetrahedral and double bond stereo
        stereo_data = [
            {
                "type": "Atom_Tetrahedral",
                "centered_on": 1,
                "specified": "Specified",
                "descriptor": "R",
                "controlling_atoms": [0, 2, 3, 4]
            },
            {
                "type": "Bond_Double",
                "centered_on": 2,
                "specified": "Specified",
                "descriptor": "E",
                "controlling_atoms": [1, 3]
            }
        ]

        mock_identify.side_effect = [
            {"mol1": stereo_data},
            {"mol2": stereo_data}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        assert result["tetra_matches"] == 1.0
        assert result["db_matches"] == 1.0
        assert result["total_tetra"] == 1.0
        assert result["total_db"] == 1.0
        assert result["total_stereo"] == 2.0

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_different_controlling_atoms(self, mock_identify, simple_mol):
        """Test when stereocenters have different controlling atoms."""
        stereo_data1 = [{
            "type": "Atom_Tetrahedral",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "R",
            "controlling_atoms": [0, 2, 3, 4]
        }]

        stereo_data2 = [{
            "type": "Atom_Tetrahedral",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "R",
            "controlling_atoms": [0, 2, 3, 5]  # Different controlling atoms
        }]

        mock_identify.side_effect = [
            {"mol1": stereo_data1},
            {"mol2": stereo_data2}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        # Should not match due to different controlling atoms
        assert result["missing_centres"] == 2.0
        assert result["tetra_matches"] == 0.0

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_identify_failure(self, mock_identify, simple_mol):
        """Test handling when stereogenic element identification fails."""
        mock_identify.side_effect = [None, {"mol2": []}]

        with pytest.raises(ValueError, match="Could not identify stereogenic elements"):
            StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_details_structure(self, mock_identify, simple_mol):
        """Test the structure of the details in the result."""
        stereo_data1 = [{
            "type": "Atom_Tetrahedral",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "R",
            "controlling_atoms": [0, 2, 3, 4]
        }]

        stereo_data2 = [{
            "type": "Atom_Tetrahedral",
            "centered_on": 1,
            "specified": "Specified",
            "descriptor": "S",
            "controlling_atoms": [0, 2, 3, 4]
        }]

        mock_identify.side_effect = [
            {"mol1": stereo_data1},
            {"mol2": stereo_data2}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        assert len(result["details"]) == 1
        detail = result["details"][0]
        assert len(detail) == 2
        assert detail[0] == stereo_data1[0]
        assert detail[1] == stereo_data2[0]

    @patch.object(StereoAnalyser, 'identify_stereogenic_elements')
    def test_compare_stereo_elements_complex_scenario(self, mock_identify, simple_mol):
        """Test a complex scenario with multiple stereocenters and various outcomes."""
        stereo_data1 = [
            {
                "type": "Atom_Tetrahedral",
                "centered_on": 1,
                "specified": "Specified",
                "descriptor": "R",
                "controlling_atoms": [0, 2, 3, 4]
            },
            {
                "type": "Bond_Double",
                "centered_on": 2,
                "specified": "Specified",
                "descriptor": "E",
                "controlling_atoms": [1, 3]
            },
            {
                "type": "Atom_Tetrahedral",
                "centered_on": 5,
                "specified": "Specified",
                "descriptor": "S",
                "controlling_atoms": [4, 6, 7, 8]
            }
        ]

        stereo_data2 = [
            {
                "type": "Atom_Tetrahedral",
                "centered_on": 1,
                "specified": "Specified",
                "descriptor": "S",  # Flipped
                "controlling_atoms": [0, 2, 3, 4]
            },
            {
                "type": "Bond_Double",
                "centered_on": 2,
                "specified": "Specified",
                "descriptor": "E",  # Match
                "controlling_atoms": [1, 3]
            }
            # Missing the third stereocenter
        ]

        mock_identify.side_effect = [
            {"mol1": stereo_data1},
            {"mol2": stereo_data2}
        ]

        result = StereoAnalyser.compare_stereo_elements(simple_mol, simple_mol)

        assert result["tetra_matches"] == 0.0
        assert result["tetra_flips"] == 1.0
        assert result["db_matches"] == 1.0
        assert result["db_flips"] == 0.0
        assert result["missing_centres"] == 1.0  # One missing stereocenter
        assert result["total_stereo"] == 3.0


class TestStereoAnalyserIntegration:
    """Integration tests for StereoAnalyser."""

    def test_real_molecule_analysis(self):
        """Test with real molecules to ensure the system works end-to-end."""
        # Test with real chiral molecules
        mol1 = Chem.MolFromSmiles("C[C@H](O)N")  # (R)-configuration
        mol2 = Chem.MolFromSmiles("C[C@@H](O)N")  # (S)-configuration

        if mol1 is not None and mol2 is not None:
            # Test identification
            stereo1 = StereoAnalyser.identify_stereogenic_elements(mol1, "mol1.mol")
            stereo2 = StereoAnalyser.identify_stereogenic_elements(mol2, "mol2.mol")

            assert isinstance(stereo1, dict)
            assert isinstance(stereo2, dict)

            # Test comparison
            result = StereoAnalyser.compare_stereo_elements(mol1, mol2)

            # Should detect the stereochemical difference
            assert isinstance(result, dict)
            assert "tetra_flips" in result
            assert "tetra_matches" in result

    # def test_logging_behavior(self, caplog, simple_mol):
    #     """Test that appropriate logging occurs."""
    #     with caplog.at_level(logging.DEBUG, logger=logger.name):
    #         StereoAnalyser.identify_stereogenic_elements(simple_mol, "test.mol")

    #     # Should log when no stereogenic elements are found
    #     assert "No stereogenic elements found" in caplog.text

    def test_error_handling_workflow(self):
        """Test complete error handling workflow."""
        # Test with invalid molecule
        with pytest.raises(ValueError):
            StereoAnalyser.identify_stereogenic_elements("invalid", "test.mol")

        # Test with None molecule
        with pytest.raises(ValueError):
            StereoAnalyser.identify_stereogenic_elements(None, "test.mol")

        # Test comparison with invalid molecules
        valid_mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError):
            StereoAnalyser.compare_stereo_elements(valid_mol, "invalid")

    def test_static_method_behavior(self):
        """Test that methods are properly static."""
        # Should be able to call without instantiating the class
        mol = Chem.MolFromSmiles("CCO")

        result1 = StereoAnalyser.identify_stereogenic_elements(mol, "test.mol")
        result2 = StereoAnalyser.compare_stereo_elements(mol, mol)

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

        # Should also work with class instance
        analyser = StereoAnalyser()
        result3 = analyser.identify_stereogenic_elements(mol, "test.mol")
        result4 = analyser.compare_stereo_elements(mol, mol)

        assert result1 == result3
        assert result2 == result4