"""Defined rules for assigning classifications based on stereochemical information"""
from typing import Dict, Any

class StereochemicalClassifier:
    """
    Classifier for stereochemical properties of molecules.
    """
    @staticmethod
    def is_enantiomer(stereo_elements:Dict[str, Any]) -> bool:
        """
        Determine if the molecule is an enantiomer based on its stereochemical elements.
        """
        total_tetra        = stereo_elements.get('total_tetra', 0)
        tetra_matches      = stereo_elements.get('tetra_matches', 0)
        tetra_flips        = stereo_elements.get('tetra_flips', 0)

        total_db           = stereo_elements.get('total_db', 0)
        db_matches         = stereo_elements.get('db_matches', 0)
        db_flips           = stereo_elements.get('db_flips', 0)

        unspecified        = stereo_elements.get('unspecified', 0)
        missing            = stereo_elements.get('missing_centres', 0)

        # Optional sanity counts
        tetra_missing      = stereo_elements.get('tetra_missing', None)
        db_missing         = stereo_elements.get('db_missing', None)

        fail = False

        # 0) Quick disqualifiers
        if total_tetra == 0:
            fail = True
        if missing != 0:
            fail = True
        if unspecified != 0:
            fail = True

        # 1) No E/Z flips allowed
        if db_flips > 0:
            fail = True

        # 2) No tetrahedral matches allowed
        if tetra_matches > 0:
            fail = True

        # 3) All tetrahedral centres must be flipped
        if tetra_flips != total_tetra:
            fail = True

        # 4) If there are any DB sites, all must match
        if (total_db > 0) and (db_matches != total_db):
            fail = True

        # 5) Invariants
        if tetra_missing is not None:
            expected = tetra_matches + tetra_flips + tetra_missing
            assert expected == total_tetra, (
                f"Tetra counts inconsistent: expected {expected}, got {total_tetra}"
            )

        if db_missing is not None:
            assert db_matches + db_flips + db_missing == total_db, "DB counts inconsistent"

        return not fail

    @staticmethod
    def is_diastereomer(stereo_elements:Dict[str, Any]) -> bool:
        """
        Determine if the molecule is a diastereomer based on its stereochemical elements.
        """
        tetra_matches      = stereo_elements.get('tetra_matches', 0)
        tetra_flips        = stereo_elements.get('tetra_flips', 0)

        db_matches         = stereo_elements.get('db_matches', 0)
        db_flips           = stereo_elements.get('db_flips', 0)

        unspecified        = stereo_elements.get('unspecified', 0)
        missing            = stereo_elements.get('missing_centres', 0)
        total_matches    = tetra_matches + db_matches
        total_flips      = tetra_flips + db_flips

        # Optional: if you track these separately, they help with assertions
        # total_stereo       = stereo_elements.get('total_stereo', 0)

        if missing > 0 or unspecified > 0:
            return False

        # should have at least one match and one flip to be diastereomer, can be either tetra or db
        # handle double bond explicitly
        if db_flips > 0:
            return True # explicit return for E/Z diastereomers,
        # must come after enantiomers in this case in the logical flow
        # now handle cases with tetrahedral centres only
        if tetra_matches >= 1 and tetra_flips >= 1: # for cases where only terta exists,
            #again must come after enantiomers in logical flow
            return True
        # handle cases where both tetra and db exist
        if total_matches >= 1 and total_flips >= 1:
            return True
        return False

    @staticmethod
    def is_parent_child(stereo_elements: Dict[str, int]) -> bool:
        """
        True depending on two different criteria:
        1) If one structure has no stereochemical elements at all, and the other has defined
        stereochemistry, excluding the influence of E/Z stereochemistry.

        2) If one structure encapsulates all the stereochemical elements of the other, plus
        some additional ones, this time including E/Z stereochemistry. (removes need for putative
        and ambiguous classifications).
        """

        # lets check for criterion 1 first
        total_stereo      = stereo_elements.get('total_stereo', 0)
        missing           = stereo_elements.get('missing_centres', 0)

        # one piece of code to handle both criteria
        ## first check there is stereochemistry at all
        if total_stereo == 0:
            return False
        # check for criterion 1
        if missing == total_stereo: # should indicate one side has no stereochemistry at all
            return True
        # check that the defined fraction == 1, if < 1, then its parent-child
        defined = total_stereo - missing
        if defined / total_stereo < 1.0:
            return True

        return False
