"""Chemistry validation operations."""

import re
import logging
from typing import Optional
from pathlib import Path
from rdkit import Chem
from .analysis import StereoAnalyser
from stereomapper.utils.suppress import setup_clean_logging
from stereomapper.utils.logging import setup_logging

setup_clean_logging()

logger, summary_logger = setup_logging(
    console=True,
    level="INFO",           # Detailed logging to files
    quiet_console=True,     # Minimal console output during progress bar
    console_level="ERROR"   # Only errors to console
)

# Constants
V3000_ATOM_RE = re.compile(r'^M\s+V30\s+\d+\s+([A-Za-z*][A-Za-z0-9#]*)\b')
V2000_ATOM_COLS = (31, 34)
QUERYLIKE_SYMBOLS = {"*", "R", "R#"}

class ChemistryValidator:
    """Chemistry validation and detection operations."""
    
    @staticmethod
    def detect_charge_from_molfile(molfile_path: str) -> Optional[int]:
        """Detect formal charge from molfile using RDKit."""
        try:
            mol = Chem.MolFromMolFile(molfile_path, sanitize=True)
            if mol is None:
                return None
            total_charge = Chem.GetFormalCharge(mol)
            return int(total_charge or 0)
        except Exception:
            logger.exception("Error detecting charge in molfile %s", molfile_path)
            return None

    @staticmethod
    def fallback_charge_from_molfile(molfile_path: str) -> int:
        """
        Fallback charge detection by parsing M CHG lines.
        """
        if not isinstance(molfile_path, str):
            logger.error("Invalid molfile path: %s", molfile_path)
            raise ValueError("Input must be a valid molfile path as a string.")
        
        total_charge = 0
        try:
            with open(molfile_path, 'r') as f:
                for line in f:
                    if line.startswith('M  CHG'):
                        parts = line.strip().split()
                        try:
                            num_charged_atoms = int(parts[2])
                            charges = parts[3:]
                            if len(charges) != num_charged_atoms * 2:
                                raise ValueError("Mismatched number of charge entries.")
                            total_charge += sum(int(charges[i]) for i in range(1, len(charges), 2))
                        except (IndexError, ValueError):
                            logger.exception("Error parsing M CHG line: %s", line.strip())
                            raise
            return total_charge
        except FileNotFoundError:
            logger.exception("Could not find molfile: %s", molfile_path)
            return 0

    @staticmethod
    def looks_querylike_symbol(sym: str) -> bool:
        """Check if atom symbol looks like a query/wildcard."""
        s = sym.strip().upper()
        if s in QUERYLIKE_SYMBOLS:
            return True
        if re.fullmatch(r'R\d+', s):
            return True
        return False

    @staticmethod
    def looks_querylike_alias(alias: str) -> bool:
        """Check if alias looks like a query/wildcard."""
        a = alias.strip().upper()
        if "*" in a:
            return True
        if ChemistryValidator.looks_querylike_symbol(a):
            return True
        return False

    @staticmethod
    def is_wildcard_molfile(molfile_path: str) -> bool:
        """Check if molfile contains wildcard/query atoms."""
        try:
            with open(molfile_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            if not lines:
                return False

            header = "".join(lines[:10])
            is_v3000 = "V3000" in header

            # Fast textual markers
            markers = ("M  V30", "M  ALS", "M  RGP", "RGFILE", "M  APO", "M  MRV SMA")
            if any(m in header or any(m in ln for ln in lines) for m in markers):
                return True

            # Atom symbol scans
            if is_v3000:
                for ln in lines:
                    m = V3000_ATOM_RE.match(ln)
                    if m and ChemistryValidator.looks_querylike_symbol(m.group(1)):
                        return True
            else:
                # V2000 logic
                if len(lines) >= 4:
                    counts = lines[3]
                    try:
                        atom_count = int(counts[0:3])
                    except ValueError:
                        parts = counts.split()
                        atom_count = int(parts[0]) if parts else 0
                    
                    atom_start = 4
                    atom_end = min(atom_start + atom_count, len(lines))
                    for i in range(atom_start, atom_end):
                        ln = lines[i]
                        sym = ln[V2000_ATOM_COLS[0]:V2000_ATOM_COLS[1]].strip() or (ln.split()[3] if len(ln.split()) > 3 else "")
                        if ChemistryValidator.looks_querylike_symbol(sym):
                            return True

                # V2000 Atom Alias blocks
                i = 0
                while i < len(lines):
                    if re.match(r'^\s*A\s+\d+\s*$', lines[i]):
                        if i + 1 < len(lines):
                            alias = lines[i + 1].strip()
                            if ChemistryValidator.looks_querylike_alias(alias):
                                return True
                            i += 2
                            continue
                    i += 1

            # RDKit confirmation
            mol = Chem.MolFromMolFile(molfile_path, sanitize=False, removeHs=False)
            if mol is None:
                return False

            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    return True
                if atom.HasQuery():
                    return True
                if atom.HasProp("molFileAlias"):
                    alias = atom.GetProp("molFileAlias")
                    if ChemistryValidator.looks_querylike_alias(alias):
                        return True
                if ChemistryValidator.looks_querylike_symbol(atom.GetSymbol()):
                    return True

            return False

        except Exception:
            logger.exception("Error reading molfile %s", molfile_path)
            return True
        
    @staticmethod
    def is_radioactive(mol: Chem.Mol) -> bool:
        """
        Check if a molecule contains radioactive atoms.
        
        Parameters:
        mol (rdkit.Chem.Mol): The molecule to check.
        
        Returns:
        bool: True if the molecule contains radioactive atoms, False otherwise.
        """
        
        if not isinstance(mol, Chem.Mol):
            raise ValueError("Input must be an RDKit Mol object.")
        
        if mol:
            for atom in mol.GetAtoms():
                # prevent bug associated with dummy atoms i.e "R1" etc..
                if atom.GetAtomicNum() > 0: # Check for dummy atoms
                    isotope = atom.GetIsotope()
                    if isotope > 0:
                        return True
        return False

    @staticmethod
    def is_stereo_disagreement(orig_mol: Chem.Mol, tmp_mol: Chem.Mol) -> bool:
        """
        Check if there is a stereochemistry discrepancy between two molecules.
        
        Parameters:
        mol1 (rdkit.Chem.Mol): The first molecule.
        mol2 (rdkit.Chem.Mol): The second molecule.
        
        Returns:
        bool: True if there is a stereochemistry discrepancy, False otherwise.
        """
        try:
            if orig_mol is None or tmp_mol is None:
                summary_logger.debug("[is_stereo_disagreement] One of the molecules is None.")
                return False
    
            # use compare_stereo_elements to get counts
            stereo_comparison = StereoAnalyser.compare_stereo_elements(orig_mol, tmp_mol)
            # check if there are any unspecified centers or missing centers
            unspec_count = stereo_comparison.get('unspecified', 0)
            missing_centres = stereo_comparison.get('missing_centres', 0)

            summary_logger.info(f"Comparison between original and temporary molecule: {stereo_comparison}")
            logger.info("Attempting to determine stereochemistry discrepancy...")
            # if any are missing or stereo diffs occur, return 'orig'
            if unspec_count > 0 or missing_centres > 0:
                summary_logger.info("Stereochemistry discrepancy detected.")
                return 'orig'
            else:
                summary_logger.info("No stereochemistry discrepancy detected.")
                return 'canon'

        except Exception as e:
            logger.exception(f"Error in check_stereo_discrepancy: {e}")
            return 'orig'
    
    @staticmethod
    def is_SRU(ctfile_path: str | Path) -> bool:
        """
        Check if a structure contains an SRU with undefined repeat count ("n").
        Args:
            ctfile (str): full molfile/CTfile contents as a string
        Returns:
            bool: True if SRU with 'n' present, False otherwise
        """
        sru_ids = set()

        with open(ctfile_path, 'r') as f:
            ctfile = f.read()

        for line in ctfile.splitlines():
            parts = line.strip().split()
            #print(parts)
            if (line.startswith("M  STY") and "SRU" in parts) or (line.startswith("M  STY") and "MUL" in parts):
            #   print("yes")
                # collect the group IDs that are SRUs
                # format: M STY <nsgroups> <sgid> <type>
                try:
                    sgid = parts[2]  # second number after "M STY"
                    sru_ids.add(sgid)
                except IndexError:
                    pass
        
        #print(sru_ids)

        if not sru_ids:
            return False

        # now check SMT lines for those SRU ids
        for line in ctfile.splitlines():
            if line.startswith("M  SMT"):
                parts = line.strip().split(maxsplit=3)
                if len(parts) >= 3:
                    sgid = parts[2]
                    label = parts[3]
                    if sgid in sru_ids and label == "n":
                        return True

        return False
    
    @staticmethod
    def is_defined_SRU(ctfile_path: str | Path) -> tuple[bool,int | None]:
        """
        Check if a structure contains an SRU with a defined repeat count (not "n").
        Args:
            ctfile (str): full molfile/CTfile contents as a string
        Returns:
            bool: True if SRU with defined repeat count present, False otherwise
        """
        sru_ids = set()

        with open(ctfile_path, 'r') as f:
            ctfile = f.read()

        for line in ctfile.splitlines():
            parts = line.strip().split()
            #print(parts)
            if (line.startswith("M  STY") and "MUL" in parts) or (line.startswith("M  STY") and "SRU" in parts):
            #   print("yes")
                # collect the group IDs that are SRUs
                # format: M STY <nsgroups> <sgid> <type>
                try:
                    sgid = parts[2]  # second number after "M STY"
                    sru_ids.add(sgid)
                except IndexError:
                    pass

        if not sru_ids:
            return False

        # now check SMT lines for those SRU ids
        for line in ctfile.splitlines():
            if line.startswith("M  SMT"):
                parts = line.strip().split(maxsplit=3)
                if len(parts) >= 3:
                    sgid = parts[2]
                    label = parts[3]
                    if sgid in sru_ids and label != "n":
                        try:
                            return True, int(label)
                        except ValueError:
                            return True, None # in case of strange formatting

        return False, None
    
    @staticmethod
    def normalise_sru_flags(path_or_mol, logger=None, tag="[sru]"):
        """
        Returns strictly-typed SRU flags for a molecule.
        Output:
        has_sru: bool
        is_def_sru: bool
        is_undef_sru: bool
        repeat_count: int | None
        """
        # 1) Does *any* SRU exist? (undefined / generic SRU detector)
        try:
            any_sru = bool(ChemistryValidator.is_SRU(path_or_mol))
        except Exception as e:
            if logger: logger.debug(f"{tag} is_SRU failed: {e}")
            any_sru = False

        # 2) Is there a *defined* SRU (and repeat count)?
        try:
            r = ChemistryValidator.is_defined_SRU(path_or_mol)
            if isinstance(r, tuple) and len(r) == 2:
                is_def, rep = bool(r[0]), r[1]
            else:
                is_def, rep = bool(r), None
        except Exception as e:
            if logger: logger.debug(f"{tag} is_defined_SRU failed: {e}")
            is_def, rep = False, None

        # 3) Clean up (repeat count only meaningful when defined)
        if not is_def:
            rep = None

        # 4) Derive composite flags
        has_sru   = any_sru or is_def
        is_undef  = has_sru and (not is_def)

        # 5) Optional: one-shot debug line to catch type drift
        if logger:
            logger.debug(f"{tag} has_sru={has_sru} is_def={is_def} is_undef={is_undef} rep={rep!r}")

        return has_sru, is_def, is_undef, rep