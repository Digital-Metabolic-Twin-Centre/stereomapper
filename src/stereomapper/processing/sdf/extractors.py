"""Functionality for extracting metabolite identifiers for annotation."""
import os
import re
from typing import Dict, List, Iterator, Any, Optional, Tuple
from rdkit import Chem

def _add(out: List[Tuple[str, str]], namespace: str, local_id: str) -> None:
    """Helper function to add namespace:local_id pairs to output list."""
    out.append((namespace, local_id))


def _norm_keys(d: Dict[str, Any]) -> Dict[str, str]:
    """Normalize keys; coerce ALL values to strings."""
    return {
        k.lower().replace(" ", "").replace("_", ""): _to_text(v).strip()
        for k, v in d.items()
    }


def _to_text(v: Any) -> str:
    """Coerce RDKit property values to a clean string."""
    if v is None:
        return ""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", "ignore")
        except Exception:
            return ""
    if isinstance(v, (list, tuple, set)):
        # Most SD props are scalars; if not, take the first non-empty item
        for x in v:
            s = _to_text(x).strip()
            if s:
                return s
        return ""
    # int/float/bool/str and misc: stringify
    return str(v)


def _strip_exts(path: str) -> str:
    """Strip file extensions from path."""
    name = os.path.basename(path)
    root, ext = os.path.splitext(name)
    if ext.lower() in (".gz", ".bz2", ".xz", ".zip"):
        root, _ = os.path.splitext(root)
        name = root
    # one more pass for .mol / .sdf
    root, ext = os.path.splitext(name)
    return root if ext.lower() in (".mol", ".sdf") else name


class SDFPropertyExtractor:
    """
    Extract properties from SDF files. This will be used to help in naming
    of structures in the output database.
    """

    @staticmethod
    def extract_properties(molfile_path: str) -> Iterator[Dict[str, str]]:
        # strictParsing=False helps skip minor format issues instead of crashing
        suppl = Chem.SDMolSupplier(molfile_path, sanitize=False, removeHs=False, strictParsing=False)
        for mol in suppl:
            if mol is None:
                continue  # skip unparseable records
            yield mol.GetPropsAsDict()


class CurieExtractor:
    """ Extract and infer identifiers from SDF properties """

    # patterns to match against for id assignment
    # note kegg not included due to clashes with vmh, causes confusion + duplication
    PATTERNS = [
        (re.compile(r"^CHEBI:(\d+)$", re.I),            lambda m: ("chebi",          m.group(1))),
        (re.compile(r"^HMDB(\d{5})$", re.I),            lambda m: ("hmdb",           f"HMDB{m.group(1)}")),
        (re.compile(r"^LM[A-Z]{2}\d{5,}$", re.I),       lambda m: ("lipidmaps",      m.string)),
        (re.compile(r"^SLM:(\d+)$", re.I),              lambda m: ("slm", m.group(1))),
        (re.compile(r"^cpd(\d{5})$", re.I),             lambda m: ("modelseed",      f"cpd{m.group(1)}")),
        (re.compile(r"^sabiork_(\d+)$", re.I),          lambda m: ("sabiork", m.group(1))),
    ]

    def infer_curies(self, props: Dict[str, Any]) -> List[str]:
        """
        Return CURIE-like ids from a single SDF record's props.
        Safe against numeric/list/bytes values.
        """
        n = _norm_keys(props)
        out: List[Tuple[str, str]] = []

        # --- ChEBI ---
        chebi_raw = n.get("chebiid") or n.get("chebi")
        if chebi_raw:
            m = re.search(r"(\d+)$", chebi_raw)
            if m:
                _add(out, "chebi", m.group(1))

        # --- HMDB ---
        hmdb_id = n.get("hmdbid")
        if not hmdb_id and n.get("databasename", "").lower() == "hmdb":
            hmdb_id = n.get("databaseid")
        if hmdb_id:
            _add(out, "hmdb", hmdb_id.upper())

        # --- SwissLipids ---
        # --- SwissLipids ---
        slm = n.get("lipidid") or n.get("swisslipidsid") or n.get("slm")
        if slm:
            s = str(slm).strip()
            # Prefer explicit SLM:NNN... match
            m = re.search(r"(?i)^SLM:(\d+)$", s)
            if m:
                _add(out, "slm", m.group(1))  # no formatting, keep all leading zeros
            else:
                # Fallback: take trailing digits exactly as-is
                m = re.search(r"(\d+)$", s)
                if m:
                    _add(out, "slm", m.group(1))  # no formatting


        # --- KEGG ---
        entry = n.get("entry")
        if entry:
            e = entry.strip()
            el = e.lower()
            if el.startswith("cpd:"):
                _add(out, "kegg.compound", e.split(":", 1)[1].upper())
            elif el.startswith("dr:"):
                _add(out, "kegg.drug", e.split(":", 1)[1].upper())
            elif el.startswith("gl:") or el.startswith("glycan:"):
                _add(out, "kegg.glycan", e.split(":", 1)[1].upper())
            elif re.fullmatch(r"[CDG]\d{5}", e.upper()):
                head = e[0].upper()
                ns = {"C": "kegg.compound", "D": "kegg.drug", "G": "kegg.glycan"}[head]
                _add(out, ns, e.upper())

        # --- LIPIDMAPS ---
        lm_id = n.get("lmid") or n.get("lipidmaps")
        if lm_id:
            _add(out, "lipidmaps", lm_id.upper())

        # --- ModelSEED ---
        ms = n.get("id") or n.get("modelseedid") or n.get("modelseed")
        if ms and re.fullmatch(r"cpd\d{5}", ms.lower()):
            _add(out, "modelseed", ms.lower())

        # Dedup, stable
        seen, curies = set(), []
        for ns, local in out:
            curie = f"{ns}:{local}"
            if curie not in seen:
                seen.add(curie)
                curies.append(curie)
        return curies

    def pick_primary_curie(self, props: Dict[str, str], curies: List[str]) -> Optional[str]:
        """
        Choose one CURIE as 'primary' based on which provider's native tag is present.
        props must already be normalized to strings (i.e., via your _norm_keys/_to_text).
        """
        n = _norm_keys(props)  # reuse your existing normalizer

        # Helper: does curies list contain any with namespace X?
        def pick(ns_prefix: str) -> Optional[str]:
            for c in curies:
                if c.lower().startswith(ns_prefix.lower() + ":"):
                    return c
            return None

        # Priority by native source fields present
        if n.get("lmid") or n.get("lipidmaps"):
            if (c := pick("lipidmaps")): return c

        entry = n.get("entry", "")
        if entry:
            if (c := pick("kegg.compound")): return c
            if (c := pick("kegg.drug")):     return c
            if (c := pick("kegg.glycan")):   return c

        if n.get("hmdbid") or n.get("databasename", "").lower() == "hmdb":
            if (c := pick("hmdb")): return c

        if n.get("chebiid") or n.get("chebi"):
            if (c := pick("chebi")): return c

        if n.get("lipidid") or n.get("swisslipidsid") or n.get("slm"):
            if (c := pick("slm")): return c

        if n.get("id") or n.get("modelseedid") or n.get("modelseed"):
            if (c := pick("modelseed")): return c

        # Fall back to first inferred
        return curies[0] if curies else None

    @staticmethod
    def fallback_accession(file_path: str) -> str:
        """
        Returns CURIE-like 'ns:id' or 'local:<basename>'.
        Never throws; never guesses ambiguous.
        """
        base = _strip_exts(file_path)

        # 2) Basename patterns (unambiguous only)
        for rx, maker in CurieExtractor.PATTERNS:
            m = rx.match(base)
            if m:
                provider, local = maker(m)
                if provider.startswith("ambiguous."):
                    return f"local:{base}"
                return f"{provider}:{local}"

        # 3) Already namespaced like 'chebi:15377' (case-insensitive)
        if ":" in base:
            ns, local = base.split(":", 1)
            if ns and local:
                return f"{ns.lower()}:{local}"

        # 4) Fallback
        return f"local:{base}"
