"""Stereo analysis operations."""

import logging
import os
from typing import Dict, List
from rdkit import Chem
from rdkit.Chem import rdmolops
from stereomapper.utils.suppress import setup_clean_logging
setup_clean_logging()

logger = logging.getLogger(__name__)

class StereoAnalyser:
    """Handles stereochemistry analysis operations."""
    
    @staticmethod
    def identify_stereogenic_elements(mol_object: Chem.Mol, mol_file: str = "mol1") -> Dict[str, List]:
        """
        Identifies all potential stereogenic elements in a molecule.
        
        Parameters:
            mol_object: RDKit Mol object
            mol_file: Name for the output dictionary key
            
        Returns:
            Dictionary with molecule name as key and list of stereogenic elements as value
        """
        if not isinstance(mol_object, Chem.Mol):
            logger.error("Input must be an RDKit Mol object.")
            raise ValueError("Input must be an RDKit Mol object.")

        base_name = os.path.splitext(os.path.basename(mol_file))[0]

        if mol_object is None:
            logger.error(f"Failed to create molecule from molFile: {mol_file}")
            raise ValueError("Failed to create molecule from molFile.")
        
        stereo_info_list = rdmolops.FindPotentialStereo(mol_object, cleanIt=True, flagPossible=False)

        if not stereo_info_list:
            logger.debug(f"No stereogenic elements found in {base_name}.")
            return {base_name: []}
        
        inner_data = []
        for info in stereo_info_list:
            stereo_type = str(info.type)
            entry = {
                "type": stereo_type,
                "centered_on": info.centeredOn,
                "specified": str(info.specified),
                "descriptor": str(info.descriptor),
                "controlling_atoms": list(info.controllingAtoms)
            }
            inner_data.append(entry)

        return {base_name: inner_data}

    @staticmethod
    def compare_stereo_elements(mol_object1: Chem.Mol, mol_object2: Chem.Mol) -> Dict:
        """
        Compares the stereogenic elements between two molecules.
        
        Returns:
            Dictionary with counts of matches, flips, penalties, and unmatched centers
        """
        if not isinstance(mol_object1, Chem.Mol) or not isinstance(mol_object2, Chem.Mol):
            logger.error("Both inputs must be RDKit Mol objects.")
            raise ValueError("Both inputs must be RDKit Mol objects.")
        
        # Identify all stereogenic elements
        stereo_elements1 = StereoAnalyser.identify_stereogenic_elements(mol_object1)
        stereo_elements2 = StereoAnalyser.identify_stereogenic_elements(mol_object2)

        if stereo_elements1 is None or stereo_elements2 is None:
            logger.error("Could not identify stereogenic elements in one or both molecules.")
            raise ValueError("Could not identify stereogenic elements in one or both molecules.")

        stereo1_info = list(stereo_elements1.items())[0][1]
        stereo2_info = list(stereo_elements2.items())[0][1]

        SPECIFIED = "Specified"

        results = {
            "total_stereo": 0,
            "total_tetra": 0,
            "tetra_matches": 0,
            "tetra_flips": 0,
            "total_db": 0,
            "db_matches": 0,
            "db_flips": 0,
            "unspecified": 0,
            "missing_centres": 0,
            "details": []
        }

        matched_indices_b = set()
        matched_indices_a = set()

        for idx1, info1 in enumerate(stereo1_info):
            best_match_idx = None
            for idx2, info2 in enumerate(stereo2_info):
                if idx2 in matched_indices_b:
                    continue
                if (
                    info1['type'] == info2['type'] and
                    sorted(info1['controlling_atoms']) == sorted(info2['controlling_atoms'])
                ):
                    best_match_idx = idx2
                    break

            if best_match_idx is not None:
                matched_indices_a.add(idx1)
                matched_indices_b.add(best_match_idx)
                info2 = stereo2_info[best_match_idx]

                if info1['specified'] == SPECIFIED and info2['specified'] == SPECIFIED:
                    if info1["type"] == "Atom_Tetrahedral":
                        if info1['descriptor'] == info2['descriptor']:
                            results["tetra_matches"] += 1.0
                        else:
                            results["tetra_flips"] += 1.0

                    elif info1["type"] == "Bond_Double":
                        if info1['descriptor'] == info2['descriptor']:
                            results["db_matches"] += 1.0
                        else:
                            results["db_flips"] += 1.0
                else:
                    results["missing_centres"] += 1.0

                results["details"].append((info1, info2))
            else:
                if info1['specified'] == SPECIFIED:
                    results["missing_centres"] += 1.0
                results["details"].append((info1, None))

        for idx2, info2 in enumerate(stereo2_info):
            if idx2 not in matched_indices_b:
                if info2['specified'] == SPECIFIED:
                    results["missing_centres"] += 1.0
                results["details"].append((None, info2))

        # Calculate total counts
        results["total_tetra"] = results["tetra_matches"] + results["tetra_flips"]
        results["total_db"] = results["db_matches"] + results["db_flips"]
        results["total_stereo"] = results["total_tetra"] + results["total_db"] + results["missing_centres"]

        return results