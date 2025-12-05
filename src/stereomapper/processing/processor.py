"""Modular molecule processing with proper error handling and separation of concerns."""

import os
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import logging
from rdkit import Chem
from stereomapper.utils.timing import section_timer, timeit
from stereomapper.results import assemblers
from stereomapper.data.cache_repo import get_cached_entry, ingest_one
from stereomapper.domain.models import ProcessingResult, CacheEntry
from stereomapper.domain.chemistry import (
    OpenBabelOperations,
    ChemistryUtils,
    ChemistryOperations,
    ChemistryValidator
    )
from stereomapper.processing.sdf import SDFPropertyExtractor, CurieExtractor
from stereomapper.domain.exceptions import (
    MoleculeParsingError,
    CanonicalizationError,
    WildcardMoleculeError,
    ChemistryError
)

logger = logging.getLogger(__name__)


class MoleculeProcessor:
    """
    Handles individual molecule processing operations.

    Processes single molecules including metadata extraction, chemistry validation,
    and unique key generation.
    """

    def __init__(self, std_version: int = 1):
        """
        Initialize the MoleculeProcessor.

        Args:
            std_version: Standardization version to use (default: 1).

        Raises:
            EnvironmentError: If required tools are not available.
        """
        self.std_version = std_version
        self._validate_environment()
        self.curie_extractor = CurieExtractor()

    def _validate_environment(self) -> None:
        """
        Validate that required tools are available.

        Raises:
            EnvironmentError: If OpenBabel is not available or version is too old.
        """
        if not OpenBabelOperations.is_obabel_available():
            raise EnvironmentError(
                "OpenBabel (obabel) is not available in the system PATH. "
                "Please install OpenBabel to use this function."
            )

        obabel_version = OpenBabelOperations.get_obabel_version()
        if obabel_version is None or obabel_version < "2.1.0":
            raise EnvironmentError(
                "OpenBabel version 2.1.0 or higher is required. "
                "Detected version: %s" % (obabel_version if obabel_version else 'not found')
            )

    def process_molecule_metadata(self, path_str: str) -> Dict[str, Any]:
        """
        Extract SDF properties and compute metadata for a molecule file.

        Args:
            path_str: Path to the molecule file.

        Returns:
            Dictionary containing metadata including accession CURIE, charge, and SRU flags.
        """
        result = {
            'accession_curie': None,
            'charge': 0,
            'is_def_sru': 0,
            'is_undef_sru': 0,
            'sru_repeat_count': None
        }

        # Extract SDF properties and accession curie
        try:
            props_iter = SDFPropertyExtractor.extract_properties(path_str)
            try:
                sdf_props = next(props_iter)
            except StopIteration:
                sdf_props = None

            curies = self.curie_extractor.infer_curies(sdf_props) if sdf_props else []
            primary_curie = (
                self.curie_extractor.pick_primary_curie(sdf_props, curies)
                if curies else None
                )

            if primary_curie:
                result['accession_curie'] = primary_curie
            else:
                result['accession_curie'] = (
                    self.curie_extractor.fallback_accession(file_path=path_str) or
                    assemblers.prefixed_identifier(
                        path_str, os.path.basename(path_str).rsplit(".", 1)[0]
                    )
                )
        except Exception:
            logger.exception("[sdf-props] Failed to extract properties from %s", path_str)

        # Detect charge
        try:
            result['charge'] = ChemistryUtils.detect_charge(path_str)
        except Exception:
            logger.debug("[charge] Failed to detect charge for %s", path_str)

        # Extract SRU flags
        try:
            _, is_def, is_undef, rep = ChemistryValidator.normalise_sru_flags(
                path_str, logger, tag="[cache_sru]"
            )
            result['is_def_sru'] = 1 if is_def else 0
            result['is_undef_sru'] = 1 if is_undef else 0
            result['sru_repeat_count'] = rep
        except Exception as e:
            logger.debug("[cache_sru] Failed to normalize SRU flags for %s: %s", path_str, e)

        return result

    def process_molecule_chemistry(
            self,
            path_str: str,
            canon_smiles: Optional[str] = None
            ) -> Tuple[
                Optional[str],
                Optional[str],
                Optional[str]
            ]:
        """
        Process molecule chemistry to get SMILES and InChIKey.

        Args:
            path_str: Path to the molecule file.
            canon_smiles: Optional canonical SMILES string.

        Returns:
            Tuple of (smiles, inchikey_first, inchikey_full).

        Raises:
            MoleculeParsingError: If molecule parsing fails.
            CanonicalizationError: If canonicalization fails.
            WildcardMoleculeError: If wildcard atoms are detected.
            ChemistryError: If unexpected chemistry errors occur.
        """
        try:
            # Get original molecule and SMILES
            m_orig = ChemistryOperations.mol_from_molfile(path_str)
            smiles_orig = Chem.MolToSmiles(m_orig, isomericSmiles=True) if m_orig else None

            # Use canonical SMILES if available
            m_canon = ChemistryOperations.mol_from_smiles(canon_smiles) if canon_smiles else None

            # Choose best SMILES representation
            chosen_source = None
            if m_orig and m_canon:
                choice = ChemistryValidator.is_stereo_disagreement(m_orig, m_canon)
                chosen_source = choice if choice in ('canon', 'orig') else 'canon'
            elif m_canon:
                chosen_source = 'canon'
            elif m_orig:
                chosen_source = 'orig'

            smiles = (
                canon_smiles if chosen_source == 'canon' else
                smiles_orig if chosen_source == 'orig' else None
                )

            if not smiles:
                return None, None, None

            # Generate InChIKey
            inchikey_full = None
            inchikey_first = None
            try:
                ik = ChemistryOperations.generate_inchikey_from_file(path_str)
                if ik and '-' in ik:
                    inchikey_full = ik
                    inchikey_first = ik.split("-")[0]
            except Exception as e:
                logger.debug("[inchi] Failed to generate InChIKey for %s: %s", path_str, e)

            return smiles, inchikey_first, inchikey_full

        except (MoleculeParsingError, CanonicalizationError, WildcardMoleculeError, ChemistryError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap unexpected chemistry errors
            raise ChemistryError("Unexpected chemistry error: %s" % str(e)) from e

    def generate_molecule_key(
            self,
            inchikey_full: Optional[str],
            smiles: Optional[str],
            charge: int,
            file_hash: str
            ) -> str:
        """
        Generate a unique molecule key.

        Args:
            inchikey_full: Full InChIKey string.
            smiles: SMILES string.
            charge: Formal charge.
            file_hash: Hash of the source file.

        Returns:
            Unique molecule key string.
        """
        if inchikey_full:
            return assemblers.make_molecule_key(
                std_version=str(self.std_version),
                inchikey_full=inchikey_full,
                isomeric_smiles=None,
                formal_charge=charge,
            )
        if smiles:
            return assemblers.make_molecule_key(
                std_version=str(self.std_version),
                inchikey_full=None,
                isomeric_smiles=smiles,
                formal_charge=charge,
            )
        else:
            return hashlib.blake2b(
                ("%s|PARSE-ERROR|%s" % (self.std_version, file_hash)).encode(),
                digest_size=16
            ).hexdigest()


class CacheManager:
    """
    Handles cache operations and validation.

    Manages database cache lookups and metadata preparation for molecule processing.
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize the CacheManager.

        Args:
            conn: SQLite database connection.
        """
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    def get_cache_metadata(
            self,
            molfile_paths: List[str]
            ) -> Tuple[
                List[str],
                List[ProcessingResult],
                Dict[
                    str,
                    Dict[str, str]
                ]
            ]:
        """
        Check cache for existing entries and prepare metadata.

        Args:
            molfile_paths: List of molecule file paths.

        Returns:
            Tuple of (files_to_process, cached_results, metadata_by_path).
        """
        to_process = []
        cached_results = []
        meta_by_path = {}

        with section_timer("[cache] checking %d files", logger, len(molfile_paths)):
            for path_str in molfile_paths:
                if not os.path.exists(path_str):
                    continue

                fhash = assemblers.hash_file(path_str)
                meta_by_path[path_str] = {
                    "file_hash": fhash,
                    "source_id": fhash,
                    "source_ref": path_str
                }

                cached_entry: Optional[CacheEntry] = get_cached_entry(fhash, self.conn)
                if cached_entry is not None and cached_entry.is_valid:
                    result = ProcessingResult(
                        molecule_id=cached_entry.molecule_id,
                        smiles=cached_entry.smiles,
                        error=cached_entry.error,
                        file_path=path_str,
                    )
                    cached_results.append(result)
                else:
                    to_process.append(path_str)

        return to_process, cached_results, meta_by_path


class BulkMoleculeProcessor:
    """
    Orchestrates bulk molecule processing with proper error handling.

    Processes multiple molecules efficiently with caching and batch operations.
    """

    def __init__(self, std_version: int = 1):
        """
        Initialize the BulkMoleculeProcessor.

        Args:
            std_version: Standardization version to use (default: 1).
        """
        self.molecule_processor = MoleculeProcessor(std_version)
        self.std_version = std_version

    @timeit(logger, "process_and_cache_molecules")
    def process_and_cache_molecules(
        self,
        molfile_list: List[str],
        conn: sqlite3.Connection,
        *,
        namespace: str = "default",
        source_kind: str = "file",
    ) -> List[ProcessingResult]:
        """
        Process many molfiles efficiently and cache them.

        Args:
            molfile_list: List of molecule file paths.
            conn: SQLite database connection.
            namespace: Namespace for organizing results (default: "default").
            source_kind: Type of source data (default: "file").

        Returns:
            List of ProcessingResult objects.
        """
        results: List[ProcessingResult] = []

        # Filter valid files
        files = [
            str(Path(p).expanduser().resolve()) for p in molfile_list if os.path.exists(p)
            ]
        if not files:
            return results

        # Initialize cache manager
        cache_manager = CacheManager(conn)
        to_process, cached_results, meta_by_path = cache_manager.get_cache_metadata(files)
        results.extend(cached_results)

        if not to_process:
            return results

        # Batch canonicalization
        with section_timer("[bulk] canonicalizing %d files", logger, len(to_process)):
            canon_map = OpenBabelOperations.canonicalise_molfiles_batch(to_process)

        # Process each file
        try:
            for path_str in to_process:
                try:
                    result = self._process_single_molecule(
                        path_str, canon_map, meta_by_path, conn, namespace, source_kind
                    )
                    if result:
                        results.append(result)

                except (
                    MoleculeParsingError,
                    CanonicalizationError,
                    WildcardMoleculeError,
                    ChemistryError
                    ) as e:
                    # Handle chemistry errors gracefully
                    logger.warning("Chemistry error for %s: %s", path_str, e.message)
                    error_result = ProcessingResult(
                        molecule_id=None,
                        smiles=None,
                        error="%s: %s" % (e.__class__.__name__, str(e)),
                        file_path=path_str,
                    )
                    results.append(error_result)
                    continue

                except Exception as e:
                    # Handle unexpected errors gracefully
                    logger.error("Unexpected error processing %s: %s", path_str, e)
                    error_result = ProcessingResult(
                        molecule_id=None,
                        smiles=None,
                        error="Unexpected error: %s" % str(e),
                        file_path=path_str,
                    )
                    results.append(error_result)
                    continue

            conn.commit()

        except Exception:
            logger.exception("[bulk] Transaction failed; rolling back")
            conn.rollback()

        return results

    def _process_single_molecule(
        self,
        path_str: str,
        canon_map: Dict[str, str],
        meta_by_path: Dict[str, Dict[str, str]],
        conn: sqlite3.Connection,
        namespace: str,
        source_kind: str
    ) -> Optional[ProcessingResult]:
        """
        Process a single molecule file.

        Args:
            path_str: Path to the molecule file.
            canon_map: Dictionary mapping paths to canonical SMILES.
            meta_by_path: Dictionary of metadata by file path.
            conn: Database connection.
            namespace: Namespace for results.
            source_kind: Type of source data.

        Returns:
            ProcessingResult or None if ingestion fails.
        """
        file_hash = meta_by_path[path_str]["file_hash"]
        source_id = meta_by_path[path_str]["source_id"]
        source_ref = meta_by_path[path_str]["source_ref"]

        error = None
        molecule_key = None

        if not os.path.exists(path_str):
            error = "file_error: not found: %s" % path_str
            molecule_key = hashlib.blake2b(
                ("%s|MISSING|%s" % (self.std_version, file_hash)).encode(),
                digest_size=16
            ).hexdigest()
            smiles = None
            inchikey_first = None
            metadata = {
                'accession_curie': None,
                'charge': 0,
                'is_def_sru': 0,
                'is_undef_sru': 0,
                'sru_repeat_count': None
            }
        else:
            # Extract metadata
            metadata = self.molecule_processor.process_molecule_metadata(path_str)

            # Process chemistry - debug the canon_map lookup
            canon_smiles = canon_map.get(path_str)
            if canon_smiles is None:
                # Try looking up with different path representations
                abs_path = os.path.abspath(path_str)
                canon_smiles = canon_map.get(abs_path)
                if canon_smiles is None:
                    # Try with original path if it was normalized
                    for key in canon_map.keys():
                        if os.path.samefile(key, path_str):
                            canon_smiles = canon_map[key]
                            break

            # process the metabolite to get structural identifiers
            smiles, inchikey_first, inchikey_full = (
                self.molecule_processor.process_molecule_chemistry(
                    path_str,
                    canon_smiles,
                )
            )

            if not smiles:
                error = "smiles_error: failed to derive SMILES from %s" % path_str

            # Generate molecule key
            molecule_key = self.molecule_processor.generate_molecule_key(
                inchikey_full, smiles, metadata['charge'], file_hash
            )

        # Ingest into database
        try:
            mol_id = ingest_one(
                conn,
                namespace=namespace,
                molecule_key=molecule_key,
                smiles=smiles,
                inchikey_first=inchikey_first,
                is_undef_sru=metadata['is_undef_sru'],
                is_def_sru=metadata['is_def_sru'],
                sru_repeat_count=metadata['sru_repeat_count'],
                error=error,
                feature_version=0,
                feature_blob=None,
                source_id=source_id,
                source_kind=source_kind,
                source_ref=source_ref,
                accession_curie=metadata['accession_curie'],
                file_hash=file_hash,
            )

            return ProcessingResult(
                molecule_id=mol_id,
                smiles=smiles,
                error=error,
                file_path=path_str,
            )

        except Exception:
            logger.exception("[bulk] Failed to ingest %s", path_str)
            return None


# Backward compatibility function
def process_and_cache_molecules(
    molfile_list: List[str],
    conn: sqlite3.Connection,
    *,
    std_version: int = 1,
    namespace: str = "default",
    source_kind: str = "file",
) -> List[ProcessingResult]:
    """
    Legacy function for backward compatibility.

    Args:
        molfile_list: List of molecule file paths.
        conn: SQLite database connection.
        std_version: Standardization version (default: 1).
        namespace: Namespace for organizing results (default: "default").
        source_kind: Type of source data (default: "file").

    Returns:
        List of ProcessingResult objects.
    """
    processor = BulkMoleculeProcessor(std_version)
    return processor.process_and_cache_molecules(
        molfile_list,
        conn,
        namespace=namespace,
        source_kind=source_kind,
    )
