"""Core chemistry operations - pure RDKit functions without I/O dependencies."""

import logging
from typing import Optional
import subprocess
import re
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdDepictor, rdMolAlign
from rdkit.Chem import rdMolDescriptors
from stereomapper.domain.exceptions.chemistry import (
    MoleculeParsingError,
    MoleculeAlignmentError,
    ChemistryError
)

logger = logging.getLogger(__name__)

INCHI_RE = re.compile(r"^InChI=1S?\/.+")  # e.g. "InChI=1S/..." or "InChI=1/...".

class ChemistryOperations:
    """Core chemistry operations that work with RDKit Mol objects."""

    @staticmethod
    def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
        """Create RDKit Mol object from SMILES string."""
        if not smiles:
            return None
        try:
            return Chem.MolFromSmiles(smiles, sanitize=True)
        except Exception:
            logger.exception("MolFromSmiles failed for: %s", smiles)
            return None

    @staticmethod
    def mol_from_molfile(molfile_path: str) -> Optional[Chem.Mol]:
        """Create RDKit Mol object from molfile path."""
        try:
            mol = Chem.MolFromMolFile(molfile_path, sanitize=True)
            if mol is None:
                raise MoleculeParsingError(
                    f"RDKit failed to parse molecule from {molfile_path}",
                    file_path=molfile_path,
                    parser="RDKit",
                    error_code="RDKIT_PARSE_FAILED"
                )
            return mol
        except Exception as e:
            if isinstance(e, MoleculeParsingError):
                raise
            raise MoleculeParsingError(
                f"RDKit failed to parse molecule from {molfile_path}",
                file_path=molfile_path,
                parser="RDKit",
                error_code="RDKIT_PARSE_FAILED"
            ) from e

    @staticmethod
    def get_formal_charge(mol: Chem.Mol) -> int:
        """Get the formal charge of a molecule."""
        if mol is None:
            return 0
        charge = Chem.GetFormalCharge(mol)
        return int(charge or 0)

    @staticmethod
    def generate_inchikey(mol: Chem.Mol) -> Optional[str]:
        """Generate InChIKey from RDKit Mol object."""
        if mol is None:
            return None
        try:
            return Chem.inchi.MolToInchiKey(mol)
        except Exception:
            logger.exception("InChIKey generation failed")
            return None

    @staticmethod
    def gen_inchikey_software(molfile: str) -> Optional[str]:
        """Generate the InChIKey using the InChI software directly from a molfile."""
        try:
            cmd = [
                "inchi",
                molfile,
                "-AMI", "-AMIOutStd", "-AMIlogStd",
                "-AMIPrbNone", "-Key", "-NoLabels"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            res = (result.stdout or "") + (result.stderr or "")

            inchikey = None
            for line in res.splitlines():
                if "InChIKey=" in line:
                    inchikey = line.split("=", 1)[1].strip()
                    break

            if inchikey and re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", inchikey):
                return inchikey

            logger.warning(f"InChIKey not found in output:\n{res}")
            return None

        except Exception as e:
            logger.warning(f"InChIKey generation via InChI software failed: {e}")
            return None

    @staticmethod
    def gen_inchistring_software(molfile: str) -> Optional[str]:
        """Generate the InChI string using the InChI software directly from a molfile."""
        try:
            cmd = [
                "inchi",
                molfile,
                "-AMI", "-AMIOutStd", "-AMIlogStd", "-AMIPrbNone",
                "-NoLabels",
            ]
            # NOTE: many builds print to stderr; capture both.
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=60
            )

            # Combine streams and normalize whitespace/newlines
            res = ((result.stdout or "") + (result.stderr or "")).replace("\r\n", "\n").strip()

            inchi = None
            for line in res.splitlines():
                line = line.strip()
                if line.startswith("InChI="):
                    inchi = line  # full line is the InChI
                    break

            # Validate shape (avoid returning banners or junk)
            if inchi and INCHI_RE.match(inchi):
                return inchi

            logger.warning("InChI string not found or invalid. Output was:\n%s", res)
            return None

        except FileNotFoundError:
            logger.warning("InChI binary ('inchi') not found on PATH.")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("InChI generation timed out.")
            return None
        except subprocess.CalledProcessError as e:
            combined = ((e.stdout or "") + (e.stderr or "")).strip()
            logger.warning("InChI generation failed (exit %s). Output:\n%s", e.returncode, combined)
            return None
        except Exception as e:
            logger.warning("InChI string generation via InChI software failed: %s", e)
            return None

    @staticmethod
    def generate_inchikey_from_file(molfile_path: str) -> Optional[str]:
        """Generate InChIKey directly from a molfile path."""
        try:
            mol = ChemistryOperations.mol_from_molfile(molfile_path)
            if mol is None:
                return None
            return ChemistryOperations.generate_inchikey(mol)
        except Exception:
            logger.exception("InChIKey generation from file failed for: %s", molfile_path)
            return None

    @staticmethod
    def generate_molecular_formula(mol: Chem.Mol) -> Optional[str]:
        """Calculate molecular formula from RDKit Mol object."""
        if mol is None:
            return None
        try:
            return rdMolDescriptors.CalcMolFormula(mol)
        except Exception:
            logger.exception("Molecular formula calculation failed")
            return None

    @staticmethod
    def align_molecules(mol_object1: Chem.Mol, mol_object2: Chem.Mol, cid1=None, cid2=None) -> Optional[float]:
        """Align two molecules and return the best RMSD, or None if alignment fails."""
        if not isinstance(mol_object1, Chem.Mol) or not isinstance(mol_object2, Chem.Mol):
            # Log the error but return None instead of raising
            logger.warning(f"Invalid input types for molecule alignment: {type(mol_object1)}, {type(mol_object2)}")
            return None

        try:
            mol_object1 = Chem.RemoveHs(mol_object1)
            mol_object2 = Chem.RemoveHs(mol_object2)

            rdDepictor.Compute2DCoords(mol_object1)
            rdDepictor.Compute2DCoords(mol_object2)

            rmsd1 = None
            rmsd2 = None

            # Try substructure alignment
            match = mol_object1.GetSubstructMatch(mol_object2)
            if match:
                atom_map = [(i, j) for j, i in enumerate(match)]
                try:
                    rmsd1 = rdMolAlign.AlignMol(mol_object1, mol_object2, atomMap=atom_map)
                except Exception:
                    rmsd1 = None

            # Try direct alignment
            try:
                rmsd2 = rdMolAlign.AlignMol(mol_object1, mol_object2)
            except Exception:
                rmsd2 = None

            if rmsd1 is None and rmsd2 is None:
                # Log the alignment failure but return None instead of raising
                logger.warning(f"""Both substructure and direct alignment failed for molecules {cid1} vs {cid2}, fallback comparison will be attempted.""")
                return None

            return min(rmsd1, rmsd2) if rmsd1 is not None and rmsd2 is not None else (rmsd1 or rmsd2)

        except Exception as e:
            # Log unexpected errors but return None instead of raising
            logger.warning(f"Unexpected error during molecule alignment {cid1} vs {cid2}: {str(e)}")
            return None
    @staticmethod
    def fingerprint_tanimoto(mol1: Chem.Mol, mol2: Chem.Mol) -> Optional[float]:
        """Calculate the Tanimoto similarity between two molecules using Morgan fingerprints."""
        if not isinstance(mol1, Chem.Mol) or not isinstance(mol2, Chem.Mol):
            return None
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception:
            logger.exception("Tanimoto similarity calculation failed")
            return None
        