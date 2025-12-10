import pytest
from unittest.mock import Mock, patch, MagicMock
from rdkit import Chem

from stereomapper.classification.engine import RelationshipAnalyser
from stereomapper.models.stereo_classification import StereoClassification
from stereomapper.domain.models import SimilarityResult


class TestRelationshipAnalyser:
    """Test cases for RelationshipAnalyser class."""

    @pytest.fixture
    def analyser(self):
        """Create a RelationshipAnalyser instance for testing."""
        return RelationshipAnalyser()

    @pytest.fixture
    def mock_mol1(self):
        """Create a mock RDKit Mol object."""
        mol = Mock(spec=Chem.Mol)
        return mol

    @pytest.fixture
    def mock_mol2(self):
        """Create a mock RDKit Mol object."""
        mol = Mock(spec=Chem.Mol)
        return mol

    @pytest.fixture
    def basic_params(self):
        """Basic parameters for calc_relationship method."""
        return {
            'charge1': 0,
            'charge2': 0,
            'cid1': 'CID1',
            'cid2': 'CID2',
            'isRadio1': False,
            'isRadio2': False,
            'has_sru1': False,
            'has_sru2': False,
            'is_undef_sru1': False,
            'is_undef_sru2': False,
            'sru_repeat_count1': None,
            'sru_repeat_count2': None,
        }

    def test_init(self, analyser):
        """Test RelationshipAnalyser initialization."""
        assert analyser.classifier is not None
        assert analyser.scorer is not None

    def test_invalid_mol_objects(self, analyser, basic_params):
        """Test calc_relationship with invalid mol objects."""
        with pytest.raises(ValueError, match="Both inputs must be RDKit Mol objects"):
            analyser.calc_relationship(
                "not_a_mol", "also_not_a_mol", **basic_params
            )

    def test_radioactive_mismatch(self, analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship when one molecule is radioactive."""
        basic_params['isRadio1'] = True
        basic_params['isRadio2'] = False
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "Unclassified"

    def test_sru_mismatch(self, analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship when SRU presence differs."""
        basic_params['has_sru1'] = True
        basic_params['has_sru2'] = False
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "Unclassified"

    def test_undefined_sru_mismatch(self, analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship when undefined SRU status differs."""
        basic_params['has_sru1'] = True
        basic_params['has_sru2'] = True
        basic_params['is_undef_sru1'] = True
        basic_params['is_undef_sru2'] = False
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "Unclassified"

    def test_sru_repeat_count_mismatch(self, analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship when SRU repeat counts differ."""
        basic_params['has_sru1'] = True
        basic_params['has_sru2'] = True
        basic_params['is_undef_sru1'] = False
        basic_params['is_undef_sru2'] = False
        basic_params['sru_repeat_count1'] = 3
        basic_params['sru_repeat_count2'] = 5
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "Unclassified"

    @patch('stereomapper.classification.engine.ChemistryOperations.align_molecules')
    @patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd')
    def test_rmsd_error(self, mock_normalise, mock_align, analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship when RMSD calculation fails."""
        mock_align.return_value = 1.5
        mock_normalise.return_value = (None, "RMSD calculation failed")
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "RMSD_ERROR"
        assert result.rmsd is None
        assert "error" in result.details

    @patch('stereomapper.classification.engine.StereoAnalyser.compare_stereo_elements')
    @patch('stereomapper.classification.engine.ChemistryOperations.align_molecules')
    @patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd')
    def test_stereo_elements_none(self, mock_normalise, mock_align, mock_stereo, 
                                 analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship when stereo elements cannot be identified."""
        mock_align.return_value = 1.5
        mock_normalise.return_value = (0.1, None)
        mock_stereo.return_value = None
        
        with pytest.raises(ValueError, match="Could not identify stereogenic elements"):
            analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)

    @patch('stereomapper.classification.engine.StereoAnalyser.compare_stereo_elements')
    @patch('stereomapper.classification.engine.ChemistryOperations.align_molecules')
    @patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd')
    @patch('stereomapper.classification.engine.ChemistryOperations.fingerprint_tanimoto')
    @patch('stereomapper.classification.engine.InChIFallbackAnalyser')
    @patch('stereomapper.classification.engine.FeatureBuilder')
    def test_no_stereogenic_elements_identical(self, mock_builder_class, mock_inchi_class, 
                                             mock_tanimoto, mock_normalise, mock_align, 
                                             mock_stereo, analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship with no stereogenic elements (identical)."""
        # Setup mocks
        mock_align.return_value = 1.5
        mock_normalise.return_value = (0.1, None)
        mock_stereo.return_value = {
            'tetra_matches': 0, 'tetra_flips': 0,
            'db_matches': 0, 'db_flips': 0,
            'missing_centres': 0, 'unspecified': 0,
            'total_stereo': 0
        }
        mock_tanimoto.return_value = 0.85
        
        # Mock InChI analyser
        mock_inchi = Mock()
        mock_inchi._get_inchikey_layers.return_value = {
            'first': 'LAYER1', 'second': 'LAYER2', 'third': 'LAYER3'
        }
        mock_inchi_class.return_value = mock_inchi
        
        # Mock FeatureBuilder
        mock_builder = Mock()
        mock_confidence = Mock()
        mock_confidence.score = 0.95
        mock_confidence.as_dict.return_value = {'confidence': 0.95}
        mock_builder.build_features_for_confidence.return_value = mock_confidence
        mock_builder_class.return_value = mock_builder
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "Unresolved"
        assert result.rmsd == 0.1

    @patch('stereomapper.classification.engine.StereoAnalyser.compare_stereo_elements')
    @patch('stereomapper.classification.engine.ChemistryOperations.align_molecules')
    @patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd')
    @patch('stereomapper.classification.engine.ChemistryOperations.fingerprint_tanimoto')
    @patch('stereomapper.classification.engine.InChIFallbackAnalyser')
    @patch('stereomapper.classification.engine.FeatureBuilder')
    def test_no_stereogenic_elements_protomers(self, mock_builder_class, mock_inchi_class,
                                             mock_tanimoto, mock_normalise, mock_align,
                                             mock_stereo, analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship with no stereogenic elements (protomers)."""
        # Setup different charges
        basic_params['charge1'] = 0
        basic_params['charge2'] = 1
        
        # Setup mocks
        mock_align.return_value = 1.5
        mock_normalise.return_value = (0.1, None)
        mock_stereo.return_value = {
            'tetra_matches': 0, 'tetra_flips': 0,
            'db_matches': 0, 'db_flips': 0,
            'missing_centres': 0, 'unspecified': 0,
            'total_stereo': 0
        }
        mock_tanimoto.return_value = 0.85
        
        # Mock InChI analyser
        mock_inchi = Mock()
        mock_inchi._get_inchikey_layers.return_value = {
            'first': 'LAYER1', 'second': 'LAYER2', 'third': 'LAYER3'
        }
        mock_inchi_class.return_value = mock_inchi
        
        # Mock FeatureBuilder
        mock_builder = Mock()
        mock_confidence = Mock()
        mock_confidence.score = 0.90
        mock_confidence.as_dict.return_value = {'confidence': 0.90}
        mock_builder.build_features_for_confidence.return_value = mock_confidence
        mock_builder_class.return_value = mock_builder
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "Protomers"
        assert result.rmsd == 0.1

    @patch('stereomapper.classification.engine.StereoAnalyser.compare_stereo_elements')
    @patch('stereomapper.classification.engine.ChemistryOperations.align_molecules')
    @patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd')
    @patch('stereomapper.classification.engine.ChemistryOperations.fingerprint_tanimoto')
    @patch('stereomapper.classification.engine.InChIFallbackAnalyser')
    @patch('stereomapper.classification.engine.FeatureBuilder')
    @patch('stereomapper.classification.engine.StereochemicalClassifier.is_enantiomer')
    def test_enantiomers(self, mock_is_enantiomer, mock_builder_class, mock_inchi_class,
                        mock_tanimoto, mock_normalise, mock_align, mock_stereo,
                        analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship for enantiomers."""
        # Setup mocks
        mock_align.return_value = 1.5
        mock_normalise.return_value = (0.1, None)
        mock_stereo.return_value = {
            'tetra_matches': 0, 'tetra_flips': 2,
            'db_matches': 1, 'db_flips': 0,
            'missing_centres': 0, 'unspecified': 0,
            'total_stereo': 3
        }
        mock_tanimoto.return_value = 0.85
        mock_is_enantiomer.return_value = True
        
        # Mock InChI analyser
        mock_inchi = Mock()
        mock_inchi._get_inchikey_layers.return_value = {
            'first': 'LAYER1', 'second': 'LAYER2', 'third': 'LAYER3'
        }
        mock_inchi_class.return_value = mock_inchi
        
        # Mock FeatureBuilder
        mock_builder = Mock()
        mock_confidence = Mock()
        mock_confidence.score = 0.88
        mock_confidence.as_dict.return_value = {'confidence': 0.88}
        mock_builder.build_features_for_confidence.return_value = mock_confidence
        mock_builder_class.return_value = mock_builder
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "Enantiomers"
        assert result.rmsd == 0.1

    @patch('stereomapper.classification.engine.StereoAnalyser.compare_stereo_elements')
    @patch('stereomapper.classification.engine.ChemistryOperations.align_molecules')
    @patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd')
    @patch('stereomapper.classification.engine.ChemistryOperations.fingerprint_tanimoto')
    @patch('stereomapper.classification.engine.InChIFallbackAnalyser')
    @patch('stereomapper.classification.engine.FeatureBuilder')
    @patch('stereomapper.classification.engine.StereochemicalClassifier.is_enantiomer')
    @patch('stereomapper.classification.engine.StereochemicalClassifier.is_diastereomer')
    def test_diastereomers(self, mock_is_diastereomer, mock_is_enantiomer, 
                          mock_builder_class, mock_inchi_class, mock_tanimoto,
                          mock_normalise, mock_align, mock_stereo,
                          analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship for diastereomers."""
        # Setup mocks
        mock_align.return_value = 1.5
        mock_normalise.return_value = (0.2, None)
        mock_stereo.return_value = {
            'tetra_matches': 1, 'tetra_flips': 1,
            'db_matches': 0, 'db_flips': 1,
            'missing_centres': 0, 'unspecified': 0,
            'total_stereo': 3
        }
        mock_tanimoto.return_value = 0.75
        mock_is_enantiomer.return_value = False
        mock_is_diastereomer.return_value = True
        
        # Mock InChI analyser
        mock_inchi = Mock()
        mock_inchi._get_inchikey_layers.return_value = {
            'first': 'LAYER1', 'second': 'LAYER2', 'third': 'LAYER3'
        }
        mock_inchi_class.return_value = mock_inchi
        
        # Mock FeatureBuilder
        mock_builder = Mock()
        mock_confidence = Mock()
        mock_confidence.score = 0.82
        mock_confidence.as_dict.return_value = {'confidence': 0.82}
        mock_builder.build_features_for_confidence.return_value = mock_confidence
        mock_builder_class.return_value = mock_builder
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "Diastereomers"
        assert result.rmsd == 0.2

    @patch('stereomapper.classification.engine.StereoAnalyser.compare_stereo_elements')
    @patch('stereomapper.classification.engine.ChemistryOperations.align_molecules')
    @patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd')
    @patch('stereomapper.classification.engine.ChemistryOperations.fingerprint_tanimoto')
    @patch('stereomapper.classification.engine.InChIFallbackAnalyser')
    @patch('stereomapper.classification.engine.FeatureBuilder')
    @patch('stereomapper.classification.engine.StereochemicalClassifier')
    def test_parent_child_structures(self, mock_classifier, mock_builder_class, mock_inchi_class,
                                mock_tanimoto, mock_normalise, mock_align, mock_stereo,
                                analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship for putative structures."""
        # Setup mocks
        mock_align.return_value = 1.5
        mock_normalise.return_value = (0.3, None)
        mock_stereo.return_value = {
            'tetra_matches': 1, 'tetra_flips': 1,
            'db_matches': 0, 'db_flips': 0,
            'missing_centres': 1, 'unspecified': 0,
            'total_stereo': 3
        }
        mock_tanimoto.return_value = 0.70
        
        # Setup classifier mocks
        mock_classifier.is_enantiomer.return_value = False
        mock_classifier.is_diastereomer.return_value = False
        mock_classifier.is_parent_child.return_value = True
        
        # Mock InChI analyser
        mock_inchi = Mock()
        mock_inchi._get_inchikey_layers.return_value = {
            'first': 'LAYER1', 'second': 'LAYER2', 'third': 'LAYER3'
        }
        mock_inchi_class.return_value = mock_inchi
        
        # Mock FeatureBuilder
        mock_builder = Mock()
        mock_confidence = Mock()
        mock_confidence.score = 0.75
        mock_confidence.as_dict.return_value = {'confidence': 0.75}
        mock_builder.build_features_for_confidence.return_value = mock_confidence
        mock_builder_class.return_value = mock_builder
        
        result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
        
        assert result.classification == "Parent-child"
        assert result.rmsd == 0.3

    @patch('stereomapper.classification.engine.StereoAnalyser.compare_stereo_elements')
    @patch('stereomapper.classification.engine.ChemistryOperations.align_molecules')
    @patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd')
    @patch('stereomapper.classification.engine.ChemistryOperations.fingerprint_tanimoto')
    def test_tanimoto_failure(self, mock_tanimoto, mock_normalise, mock_align, mock_stereo,
                             analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship when Tanimoto similarity calculation fails."""
        mock_align.return_value = 1.5
        mock_normalise.return_value = (0.1, None)
        mock_stereo.return_value = {
            'tetra_matches': 0, 'tetra_flips': 0,
            'db_matches': 0, 'db_flips': 0,
            'missing_centres': 0, 'unspecified': 0,
            'total_stereo': 0
        }
        mock_tanimoto.return_value = None  # Tanimoto calculation fails
        
        # Should still work, just with None tanimoto value
        with patch('stereomapper.classification.engine.InChIFallbackAnalyser'), \
             patch('stereomapper.classification.engine.FeatureBuilder') as mock_builder_class:
            
            mock_builder = Mock()
            mock_confidence = Mock()
            mock_confidence.score = 0.95
            mock_confidence.as_dict.return_value = {'confidence': 0.95}
            mock_builder.build_features_for_confidence.return_value = mock_confidence
            mock_builder_class.return_value = mock_builder
            
            result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
            
            assert result.classification == "Unresolved"

    @patch('stereomapper.classification.engine.StereoAnalyser.compare_stereo_elements')
    @patch('stereomapper.classification.engine.ChemistryOperations.align_molecules')
    @patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd')
    @patch('stereomapper.classification.engine.ChemistryOperations.fingerprint_tanimoto')
    @patch('stereomapper.classification.engine.InChIFallbackAnalyser')
    def test_inchi_layer_extraction_failure(self, mock_inchi_class, mock_tanimoto,
                                           mock_normalise, mock_align, mock_stereo,
                                           analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship when InChI layer extraction fails."""
        mock_align.return_value = 1.5
        mock_normalise.return_value = (0.1, None)
        mock_stereo.return_value = {
            'tetra_matches': 0, 'tetra_flips': 0,
            'db_matches': 0, 'db_flips': 0,
            'missing_centres': 0, 'unspecified': 0,
            'total_stereo': 0
        }
        mock_tanimoto.return_value = 0.85
        
        # Mock InChI analyser to raise exception
        mock_inchi = Mock()
        mock_inchi._get_inchikey_layers.side_effect = Exception("InChI extraction failed")
        mock_inchi_class.return_value = mock_inchi
        
        with patch('stereomapper.classification.engine.FeatureBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_confidence = Mock()
            mock_confidence.score = 0.95
            mock_confidence.as_dict.return_value = {'confidence': 0.95}
            mock_builder.build_features_for_confidence.return_value = mock_confidence
            mock_builder_class.return_value = mock_builder
            
            result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
            
            assert result.classification == "Unresolved"

    def test_missing_charge_identical(self, analyser, mock_mol1, mock_mol2, basic_params):
        """Test calc_relationship when one charge is None."""
        basic_params['charge1'] = None
        basic_params['charge2'] = 0
        
        with patch('stereomapper.classification.engine.StereoAnalyser.compare_stereo_elements') as mock_stereo, \
             patch('stereomapper.classification.engine.ChemistryOperations.align_molecules') as mock_align, \
             patch('stereomapper.classification.engine.ChemistryUtils.normalise_rmsd') as mock_normalise, \
             patch('stereomapper.classification.engine.ChemistryOperations.fingerprint_tanimoto') as mock_tanimoto, \
             patch('stereomapper.classification.engine.InChIFallbackAnalyser'), \
             patch('stereomapper.classification.engine.FeatureBuilder') as mock_builder_class:
            
            mock_align.return_value = 1.5
            mock_normalise.return_value = (0.1, None)
            mock_stereo.return_value = {
                'tetra_matches': 0, 'tetra_flips': 0,
                'db_matches': 0, 'db_flips': 0,
                'missing_centres': 0, 'unspecified': 0,
                'total_stereo': 0
            }
            mock_tanimoto.return_value = 0.85
            
            mock_builder = Mock()
            mock_confidence = Mock()
            mock_confidence.score = 0.90
            mock_confidence.as_dict.return_value = {'confidence': 0.90}
            mock_builder.build_features_for_confidence.return_value = mock_confidence
            mock_builder_class.return_value = mock_builder
            
            result = analyser.calc_relationship(mock_mol1, mock_mol2, **basic_params)
            
            assert result.classification == "Protomers"