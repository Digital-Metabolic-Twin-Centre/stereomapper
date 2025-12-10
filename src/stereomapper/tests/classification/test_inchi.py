import pytest
from unittest.mock import Mock, patch, MagicMock
from rdkit import Chem

from stereomapper.classification.inchi import (
    InChIFallbackAnalyser, 
    parse_tms_with_unknowns, 
    undefined_fraction_from_inchi,
    _extract_double_bond_stereo,
    _has_unknown,
    _has_unknown_stereo,
    _defined_subset,
    _all_signs_inverted_defined,
    _any_defined_signs_different,
    _to_similarity_result
)
from stereomapper.models.stereo_classification import StereoClassification
from stereomapper.domain.models import SimilarityResult


class TestInChIUtilityFunctions:
    """Test cases for utility functions in inchi module."""
    
    def test_parse_tms_with_unknowns_simple(self):
        """Test parse_tms_with_unknowns with simple InChI."""
        inchi = "InChI=1S/C8H16O2/c1-3-5-7-8(9)6-4-2/h3-7H2,1-2H3/t3+,4-,5?/m1/s1"
        t_dict, m_val, s_val = parse_tms_with_unknowns(inchi)
        
        assert t_dict == {'3': '+', '4': '-', '5': '?'}
        assert m_val == "1"
        assert s_val == "1"
    
    def test_parse_tms_with_unknowns_no_layers(self):
        """Test parse_tms_with_unknowns with no TMS layers."""
        inchi = "InChI=1S/C2H6/c1-2/h1-2H3"
        t_dict, m_val, s_val = parse_tms_with_unknowns(inchi)
        
        assert t_dict == {}
        assert m_val == ""
        assert s_val == ""
    
    def test_parse_tms_with_unknowns_complex(self):
        """Test parse_tms_with_unknowns with complex stereochemistry."""
        inchi = "InChI=1S/C10H20/c1-3-5-7-9-10-8-6-4-2/h3-10H2,1-2H3/t3+,4-,5+,6-,7?,8+/m0/s1"
        t_dict, m_val, s_val = parse_tms_with_unknowns(inchi)
        
        assert t_dict == {'3': '+', '4': '-', '5': '+', '6': '-', '7': '?', '8': '+'}
        assert m_val == "0"
        assert s_val == "1"
    
    def test_parse_tms_with_unknowns_malformed(self):
        """Test parse_tms_with_unknowns with malformed InChI."""
        inchi = "not_an_inchi"
        t_dict, m_val, s_val = parse_tms_with_unknowns(inchi)
        
        assert t_dict == {}
        assert m_val == ""
        assert s_val == ""
    
    def test_extract_double_bond_stereo(self):
        """Test _extract_double_bond_stereo function."""
        inchi = "InChI=1S/C13H10O2/c1-2-3-4-5-6-7-8-9-10-11-12-13(14)15/h1,5,7-11H,12H2,(H,14,15)/b11-10+"
        result = _extract_double_bond_stereo(inchi)
        
        assert result == {'10': '+'}
    
    def test_extract_double_bond_stereo_multiple(self):
        """Test _extract_double_bond_stereo with multiple double bonds."""
        inchi = "InChI=1S/C6H10/c1-3-5-6-4-2/h3-6H,1-2H3/b5-3+,6-4-"
        result = _extract_double_bond_stereo(inchi)
        
        assert result == {'3': '+', '4': '-'}
    
    def test_extract_double_bond_stereo_no_b_layer(self):
        """Test _extract_double_bond_stereo with no b layer."""
        inchi = "InChI=1S/C2H6/c1-2/h1-2H3"
        result = _extract_double_bond_stereo(inchi)
        
        assert result == {}
    
    def test_undefined_fraction_from_inchi_all_defined_db(self):
        """Test undefined_fraction_from_inchi with all defined db."""
        inchi = "InChI=1S/C13H10O2/c1-2-3-4-5-6-7-8-9-10-11-12-13(14)15/h1,5,7-11H,12H2,(H,14,15)/b9-8-,11-10+"
        result = undefined_fraction_from_inchi(inchi)
        
        assert result == 0.0
    
    def test_undefined_fraction_from_inchi_all_defined_tetra(self):
        """Test undefined_fraction_from_inchi with all defined tetra centers."""
        inchi = "InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6-/m1/s1"
        result = undefined_fraction_from_inchi(inchi)

        assert result == 0.0
    
    def test_undefined_fraction_from_inchi_partial_undefined(self):
        """Test undefined_fraction_from_inchi with some undefined centers."""
        inchi = "InChI=1S/C6H12O/c1-3-5-6(7)4-2/h5-6H,3-4H2,1-2H3/t5?,6+/m1/s1"
        result = undefined_fraction_from_inchi(inchi)
        
        assert result == 0.5 # 1 undefined out of 2 total
    
    def test_undefined_fraction_from_inchi_s_not_1(self):
        """Test undefined_fraction_from_inchi with s layer != 1."""
        inchi = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m1/s2"
        result = undefined_fraction_from_inchi(inchi)
        
        assert result == 1.0  # Conservative fallback
    
    def test_undefined_fraction_from_inchi_no_stereo(self):
        """Test undefined_fraction_from_inchi with no stereochemistry."""
        inchi = "InChI=1S/C2H6/c1-2/h1-2H3"
        result = undefined_fraction_from_inchi(inchi)
        
        assert result == 0.0
    
    def test_has_unknown_true(self):
        """Test _has_unknown returns True when '?' present."""
        t_dict = {'1': '+', '2': '?', '3': '-'}
        assert _has_unknown(t_dict) == True
    
    def test_has_unknown_false(self):
        """Test _has_unknown returns False when no '?' present."""
        t_dict = {'1': '+', '2': '-', '3': '+'}
        assert _has_unknown(t_dict) == False
    
    def test_has_unknown_none(self):
        """Test _has_unknown with None input."""
        assert _has_unknown(None) == False
    
    def test_has_unknown_stereo(self):
        """Test _has_unknown_stereo with both tetra and double bond."""
        t_dict = {'1': '+', '2': '-'}
        db_dict = {'3': '?', '4': '+'}
        assert _has_unknown_stereo(t_dict, db_dict) == True
    
    def test_defined_subset(self):
        """Test _defined_subset filters only defined centers."""
        t_dict = {'1': '+', '2': '?', '3': '-', '4': '?'}
        result = _defined_subset(t_dict)
        
        assert result == {'1': '+', '3': '-'}
    
    def test_defined_subset_none(self):
        """Test _defined_subset with None input."""
        result = _defined_subset(None)
        assert result == {}
    
    def test_all_signs_inverted_defined_true(self):
        """Test _all_signs_inverted_defined returns True for enantiomers."""
        t1 = {'1': '+', '2': '-'}
        t2 = {'1': '-', '2': '+'}
        db1 = {'3': '+'}
        db2 = {'3': '+'}  # Double bonds should match, not invert
        
        assert _all_signs_inverted_defined(t1, db1, t2, db2) == True
    
    def test_all_signs_inverted_defined_false_tetra(self):
        """Test _all_signs_inverted_defined returns False when tetra not inverted."""
        t1 = {'1': '+', '2': '-'}
        t2 = {'1': '+', '2': '-'}  # Same, not inverted
        db1 = {'3': '+'}
        db2 = {'3': '+'}
        
        assert _all_signs_inverted_defined(t1, db1, t2, db2) == False
    
    def test_all_signs_inverted_defined_false_db(self):
        """Test _all_signs_inverted_defined returns False when DB differ."""
        t1 = {'1': '+', '2': '-'}
        t2 = {'1': '-', '2': '+'}
        db1 = {'3': '+'}
        db2 = {'3': '-'}  # Different double bond
        
        assert _all_signs_inverted_defined(t1, db1, t2, db2) == False
    
    def test_any_defined_signs_different_true(self):
        """Test _any_defined_signs_different returns True when differences exist."""
        t1 = {'1': '+', '2': '-', '3': '?'}
        t2 = {'1': '-', '2': '-', '3': '?'}  # Position 1 differs
        
        assert _any_defined_signs_different(t1, t2) == True
    
    def test_any_defined_signs_different_false(self):
        """Test _any_defined_signs_different returns False when same."""
        t1 = {'1': '+', '2': '-', '3': '?'}
        t2 = {'1': '+', '2': '-', '3': '?'}
        
        assert _any_defined_signs_different(t1, t2) == False
    
    def test_to_similarity_result(self):
        """Test _to_similarity_result conversion."""
        stereo_class = StereoClassification.identical(
                        stereo_score=100,
                        rmsd=0.0,
                    )
        result = _to_similarity_result(stereo_class)
        
        assert isinstance(result, SimilarityResult)
        assert result.classification == "Identical structures"


class TestInChIFallbackAnalyser:
    """Test cases for InChIFallbackAnalyser class."""
    
    @pytest.fixture
    def analyser(self):
        """Create an InChIFallbackAnalyser instance."""
        return InChIFallbackAnalyser()
    
    @pytest.fixture
    def mock_mol(self):
        """Create a mock RDKit Mol object."""
        return Mock(spec=Chem.Mol)
    
    def test_init(self, analyser):
        """Test InChIFallbackAnalyser initialization."""
        assert analyser.confidence_penalty == 0.3
        assert analyser.builder is not None
    
    def test_init_custom_penalty(self):
        """Test InChIFallbackAnalyser with custom penalty."""
        analyser = InChIFallbackAnalyser(confidence_penalty=0.5)
        assert analyser.confidence_penalty == 0.5
    
    @patch('stereomapper.classification.inchi.Chem.MolToInchi')
    def test_extract_inchi_layers_success(self, mock_mol_to_inchi, analyser, mock_mol):
        """Test successful InChI layer extraction."""
        mock_mol_to_inchi.return_value = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m1/s1"
        
        result = analyser._extract_inchi_layers(mock_mol)
        
        assert result is not None
        assert result["formula"] == "C4H8O"
        assert result["connectivity"] == "1-3-4(2)5"
        assert result["hydrogen"] == "3-4H,1-2H3"
        assert result["stereochemistry_sub1"] == "3+,4-"
        assert result["stereochemistry_sub2"] == "1"
        assert result["stereochemistry_sub3"] == "1"
    
    @patch('stereomapper.classification.inchi.Chem.MolToInchi')
    def test_extract_inchi_layers_failure(self, mock_mol_to_inchi, analyser, mock_mol):
        """Test InChI layer extraction failure."""
        mock_mol_to_inchi.return_value = None
        
        result = analyser._extract_inchi_layers(mock_mol)
        
        assert result is None
    
    @patch('stereomapper.classification.inchi.Chem.MolToInchi')
    def test_extract_inchi_layers_exception(self, mock_mol_to_inchi, analyser, mock_mol):
        """Test InChI layer extraction with exception."""
        mock_mol_to_inchi.side_effect = Exception("RDKit error")
        
        result = analyser._extract_inchi_layers(mock_mol)
        
        assert result is None
    
    @patch('stereomapper.classification.inchi.Chem.MolToInchiKey')
    def test_get_inchikey_layers_success(self, mock_mol_to_inchikey, analyser, mock_mol):
        """Test successful InChIKey layer extraction."""
        mock_mol_to_inchikey.return_value = "BQJCRHHNABKAKU-KBQPJGBKSA-N"
        
        result = analyser._get_inchikey_layers(mock_mol)
        
        assert result == {
            "first": "BQJCRHHNABKAKU",
            "second": "KBQPJGBKSA", 
            "third": "N"
        }
    
    @patch('stereomapper.classification.inchi.Chem.MolToInchiKey')
    def test_get_inchikey_layers_malformed(self, mock_mol_to_inchikey, analyser, mock_mol):
        """Test InChIKey extraction with malformed key."""
        mock_mol_to_inchikey.return_value = "INVALID-KEY"
        
        result = analyser._get_inchikey_layers(mock_mol)
        
        assert result is None
    
    def test_calculate_stereo_stats_enantiomers(self, analyser):
        """Test _calculate_stereo_stats for enantiomers."""
        tA = {'3': '+', '4': '-'}
        tB = {'3': '-', '4': '+'}
        inchi_a = "InChI=1S/test/t3+,4-/m0/s1"
        inchi_b = "InChI=1S/test/t3-,4+/m1/s1"
        
        result = analyser._calculate_stereo_stats(tA, tB, inchi_a, inchi_b)
        
        assert result["num_tetra_matches"] == 0
        assert result["num_tetra_flips"] == 2
        assert result["num_stereogenic_elements"] == 2
    
    def test_calculate_stereo_stats_m0_m1_case(self, analyser):
        """Test _calculate_stereo_stats for m0/m1 enantiomer case."""
        tA = {'3': '+', '4': '-'}
        tB = {'3': '+', '4': '-'}  # Same t-layer
        inchi_a = "InChI=1S/test/t3+,4-/m0/s1"
        inchi_b = "InChI=1S/test/t3+,4-/m1/s1"  # Different m-layer
        
        result = analyser._calculate_stereo_stats(tA, tB, inchi_a, inchi_b)
        
        assert result["num_tetra_matches"] == 0  # m0/m1 case treats as flipped
        assert result["num_tetra_flips"] == 2
    
    def test_calculate_stereo_stats_with_undefined(self, analyser):
        """Test _calculate_stereo_stats with undefined centers."""
        tA = {'3': '+', '4': '?'}
        tB = {'3': '-', '4': '?'}
        inchi_a = "InChI=1S/test/t3+,4?/m0/s1"
        inchi_b = "InChI=1S/test/t3-,4?/m1/s1"
        
        result = analyser._calculate_stereo_stats(tA, tB, inchi_a, inchi_b)
        
        assert result["num_missing"] == 1  # One undefined center
        assert result["num_stereogenic_elements"] == 2
    
    def test_classify_stereo_from_inchi_enantiomers(self, analyser):
        """Test _classify_stereo_from_inchi for enantiomers."""
        inchi_a = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m1/s1"
        inchi_b = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3-,4+/m1/s1"
        
        result = analyser._classify_stereo_from_inchi(inchi_a, inchi_b)
        
        assert result == "ENANTIOMERS"
    
    def test_classify_stereo_from_inchi_m0_m1(self, analyser):
        """Test _classify_stereo_from_inchi for m0/m1 enantiomers."""
        inchi_a = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m0/s1"
        inchi_b = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m1/s1"
        
        result = analyser._classify_stereo_from_inchi(inchi_a, inchi_b)
        
        assert result == "ENANTIOMERS"
    
    def test_classify_stereo_from_inchi_diastereomers(self, analyser):
        """Test _classify_stereo_from_inchi for diastereomers."""
        inchi_a = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m0/s1"
        inchi_b = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4+/m0/s1"
        
        result = analyser._classify_stereo_from_inchi(inchi_a, inchi_b)
        
        assert result == "DIASTEREOMERS"
    
    def test_classify_stereo_from_inchi_planar_vs_stereo(self, analyser):
        """Test _classify_stereo_from_inchi for planar vs stereo."""
        inchi_a = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3"  # No stereo
        inchi_b = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m0/s1"  # With stereo
        
        result = analyser._classify_stereo_from_inchi(inchi_a, inchi_b)
        
        assert result == "PLANAR_VS_STEREO"
    
    def test_classify_stereo_from_inchi_parent_child(self, analyser):
        """Test _classify_stereo_from_inchi for putative structures."""
        inchi_a = "InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2?,3-,4+,5-,6-/m0/s1"  # Undefined
        inchi_b = "InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5-,6-/m1/s1"  # Defined
        
        result = analyser._classify_stereo_from_inchi(inchi_a, inchi_b)
        
        assert result == "PLANAR_VS_STEREO"
    
    def test_classify_stereo_from_inchi_racemic(self, analyser):
        """Test _classify_stereo_from_inchi for racemic mixtures."""
        inchi_a = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m2/s1"  # m2
        inchi_b = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m0/s1"  # m0
        
        result = analyser._classify_stereo_from_inchi(inchi_a, inchi_b)
        
        assert result == "RACEMIC_OR_MIXTURE"
    
    def test_classify_stereo_from_inchi_identical(self, analyser):
        """Test _classify_stereo_from_inchi for identical structures."""
        inchi_a = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m0/s1"
        inchi_b = "InChI=1S/C4H8O/c1-3-4(2)5/h3-4H,1-2H3/t3+,4-/m0/s1"
        
        result = analyser._classify_stereo_from_inchi(inchi_a, inchi_b)
        
        assert result == "IDENTICAL_STEREO"
    
    def test_classify_stereo_from_inchi_exception(self, analyser):
        """Test _classify_stereo_from_inchi with exception."""
        with patch('stereomapper.classification.inchi.parse_tms_with_unknowns', 
                   side_effect=Exception("Parse error")):
            result = analyser._classify_stereo_from_inchi("invalid", "invalid")
            assert result == "STEREO_UNDEFINED"
    
    @patch('stereomapper.classification.inchi.ChemistryOperations.fingerprint_tanimoto')
    def test_analyze_relationship_fallback_different_skeletons(self, mock_tanimoto, analyser):
        """Test analyze_relationship_fallback with different molecular skeletons."""
        mock_mol_a = Mock()
        mock_mol_b = Mock()
        
        with patch.object(analyser, '_get_inchikey_layers') as mock_get_layers:
            mock_get_layers.side_effect = [
                {"first": "AAAA", "second": "BBBB", "third": "C"},
                {"first": "DDDD", "second": "EEEE", "third": "F"}  # Different first block
            ]
            
            result = analyser.analyze_relationship_fallback(mock_mol_a, mock_mol_b, 0, 0)
            
            assert result.classification == "Unclassified"
    
    @patch('stereomapper.classification.inchi.ChemistryOperations.fingerprint_tanimoto')
    def test_analyze_relationship_fallback_protomers(self, mock_tanimoto, analyser):
        """Test analyze_relationship_fallback for protomers."""
        mock_mol_a = Mock()
        mock_mol_b = Mock()
        mock_tanimoto.return_value = 0.85
        
        with patch.object(analyser, '_get_inchikey_layers') as mock_get_layers, \
             patch.object(analyser, '_extract_inchi_layers') as mock_extract_layers, \
             patch.object(analyser, '_build_fallback_confidence') as mock_confidence:
            
            mock_get_layers.side_effect = [
                {"first": "AAAA", "second": "BBBB", "third": "C"},
                {"first": "AAAA", "second": "BBBB", "third": "D"}  # Same first/second, diff third
            ]
            
            mock_extract_layers.side_effect = [
                {"inchi": "InChI=1S/test/t1+/m0/s1"},
                {"inchi": "InChI=1S/test/t1+/m0/s1"}
            ]
            
            mock_conf = Mock()
            mock_conf.score = 0.85
            mock_conf.as_dict.return_value = {"confidence": 0.85}
            mock_confidence.return_value = mock_conf
            
            result = analyser.analyze_relationship_fallback(mock_mol_a, mock_mol_b, 0, 1)
            
            assert result.classification == "Protomers"
    
    @patch('stereomapper.classification.inchi.ChemistryOperations.fingerprint_tanimoto')
    def test_analyze_relationship_fallback_enantiomers(self, mock_tanimoto, analyser):
        """Test analyze_relationship_fallback for enantiomers."""
        mock_mol_a = Mock()
        mock_mol_b = Mock()
        mock_tanimoto.return_value = 0.85
        
        with patch.object(analyser, '_get_inchikey_layers') as mock_get_layers, \
             patch.object(analyser, '_extract_inchi_layers') as mock_extract_layers, \
             patch.object(analyser, '_compare_full_inchi_stereochemistry') as mock_compare, \
             patch.object(analyser, '_build_fallback_confidence') as mock_confidence:
            
            mock_get_layers.side_effect = [
                {"first": "AAAA", "second": "BBBB", "third": "C"},
                {"first": "AAAA", "second": "DDDD", "third": "C"}  # Diff second, same third
            ]
            
            mock_extract_layers.side_effect = [
                {"inchi": "InChI=1S/test/t1+/m0/s1"},
                {"inchi": "InChI=1S/test/t1-/m1/s1"}
            ]
            
            mock_compare.return_value = "ENANTIOMERS"
            
            mock_conf = Mock()
            mock_conf.score = 0.80
            mock_conf.as_dict.return_value = {"confidence": 0.80}
            mock_confidence.return_value = mock_conf
            
            result = analyser.analyze_relationship_fallback(mock_mol_a, mock_mol_b, 0, 0)
            
            assert result.classification == "Enantiomers"
    
    @patch('stereomapper.classification.inchi.ChemistryOperations.fingerprint_tanimoto')
    def test_analyze_relationship_fallback_unresolved(self, mock_tanimoto, analyser):
        """Test analyze_relationship_fallback for identical structures."""
        mock_mol_a = Mock()
        mock_mol_b = Mock()
        mock_tanimoto.return_value = 0.95
        
        with patch.object(analyser, '_get_inchikey_layers') as mock_get_layers, \
             patch.object(analyser, '_extract_inchi_layers') as mock_extract_layers, \
             patch.object(analyser, '_build_fallback_confidence') as mock_confidence:
            
            mock_get_layers.side_effect = [
                {"first": "AAAA", "second": "BBBB", "third": "C"},
                {"first": "AAAA", "second": "BBBB", "third": "C"}  # All same
            ]
            
            mock_extract_layers.side_effect = [
                {"inchi": "InChI=1S/test/t1+/m0/s1"},
                {"inchi": "InChI=1S/test/t1+/m0/s1"}
            ]
            
            mock_conf = Mock()
            mock_conf.score = 0.95
            mock_conf.as_dict.return_value = {"confidence": 0.95}
            mock_confidence.return_value = mock_conf
            
            result = analyser.analyze_relationship_fallback(mock_mol_a, mock_mol_b, 0, 0)
            
            assert result.classification == "Unresolved"
    
    def test_analyze_relationship_fallback_failed_inchikey_extraction(self, analyser):
        """Test analyze_relationship_fallback when InChIKey extraction fails."""
        mock_mol_a = Mock()
        mock_mol_b = Mock()
        
        with patch.object(analyser, '_get_inchikey_layers') as mock_get_layers:
            mock_get_layers.side_effect = [None, {"first": "AAAA", "second": "BBBB", "third": "C"}]
            
            result = analyser.analyze_relationship_fallback(mock_mol_a, mock_mol_b, 0, 0)
            
            assert result.classification == "Unclassified"
    
    def test_build_fallback_confidence_none_charges(self, analyser):
        """Test _build_fallback_confidence with None charges."""
        stereo_stats = {
            "num_stereogenic_elements": 2,
            "num_tetra_matches": 0,
            "num_tetra_flips": 2,
            "num_db_matches": 0,
            "num_db_flips": 0,
            "num_missing": 0
        }
        
        with patch.object(analyser.builder, 'build_features_for_confidence') as mock_build:
            mock_conf = Mock()
            mock_conf.score = 0.85
            mock_build.return_value = mock_conf
            
            result = analyser._build_fallback_confidence(
                "ENANTIOMERS", None, None, 0.80, True, False, True, stereo_stats
            )
            
            assert result.score == 0.55  # 0.85 - 0.3 penalty
            mock_build.assert_called_once()
    
    def test_build_fallback_confidence_none_stereo_stats(self, analyser):
        """Test _build_fallback_confidence with None stereo_stats."""
        with patch.object(analyser.builder, 'build_features_for_confidence') as mock_build:
            mock_conf = Mock()
            mock_conf.score = 0.90
            mock_build.return_value = mock_conf
            
            result = analyser._build_fallback_confidence(
                "IDENTICAL", 0, 0, 0.95, True, True, True, None
            )
            
            assert hasattr(result, 'score')
            mock_build.assert_called_once()