from pathlib import Path
from typing import Optional, Union

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


def mol_from_input(
    structure: Union[str, Path],
    input_type: str = "auto",
    sanitize: bool = True,
) -> Chem.Mol:
    """
    Load an RDKit molecule from SMILES, molfile text, or molfile path.

    Parameters
    ----------
    structure:
        SMILES string, molfile text, or path to a .mol / .sdf file.
    input_type:
        "auto", "smiles", "molblock", or "file".
    sanitize:
        Whether RDKit should sanitise the molecule.

    Returns
    -------
    rdkit.Chem.Mol
    """

    structure = str(structure)

    if input_type == "auto":
        path = Path(structure)
        if path.exists():
            input_type = "file"
        elif "\n" in structure and ("V2000" in structure or "V3000" in structure):
            input_type = "molblock"
        else:
            input_type = "smiles"

    if input_type == "smiles":
        mol = Chem.MolFromSmiles(structure, sanitize=sanitize)

    elif input_type == "molblock":
        mol = Chem.MolFromMolBlock(
            structure,
            sanitize=sanitize,
            removeHs=False,
        )

    elif input_type == "file":
        path = Path(structure)
        if path.suffix.lower() == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=False)
            mol = supplier[0] if supplier and len(supplier) > 0 else None
        else:
            mol = Chem.MolFromMolFile(
                str(path),
                sanitize=sanitize,
                removeHs=False,
            )

    else:
        raise ValueError("input_type must be one of: auto, smiles, molblock, file")

    if mol is None:
        raise ValueError("RDKit failed to parse the input structure.")

    return mol


def structure_to_publication_svg(
    structure: Union[str, Path],
    output_svg: Union[str, Path],
    input_type: str = "auto",
    width: int = 500,
    height: int = 400,
    kekulize: bool = True,
    add_hs: bool = False,
    remove_hs: bool = False,
    wedge_bonds: bool = True,
    centre_molecule: bool = True,
    atom_indices: bool = False,
    bond_indices: bool = False,
    legend: Optional[str] = None,
    line_width: float = 2.0,
    font_size: float = 0.8,
    fixed_bond_length: float = 35,
    transparent_background: bool = True,
) -> str:
    """
    Generate a publication-quality SVG of a molecule using RDKit.

    Suitable for:
    - PowerPoint
    - Illustrator
    - Inkscape
    - Affinity Designer
    - journal figures

    Parameters
    ----------
    structure:
        SMILES string, molfile text, or path to .mol / .sdf.
    output_svg:
        Output SVG file path.
    input_type:
        "auto", "smiles", "molblock", or "file".
    width, height:
        SVG canvas size in pixels.
    kekulize:
        Draw aromatic systems in Kekulé form where possible.
    add_hs:
        Add explicit hydrogens before drawing.
    remove_hs:
        Remove explicit hydrogens before drawing.
    wedge_bonds:
        Preserve wedge/dash stereochemistry.
    centre_molecule:
        Centre molecule on the canvas.
    atom_indices:
        Show atom indices for debugging.
    bond_indices:
        Show bond indices for debugging.
    legend:
        Optional text below the structure.
    line_width:
        Bond line width.
    font_size:
        Atom label font scaling.
    fixed_bond_length:
        Controls apparent size of the molecule.
    transparent_background:
        Use transparent SVG background.

    Returns
    -------
    SVG text.
    """

    mol = mol_from_input(structure, input_type=input_type)

    if remove_hs:
        mol = Chem.RemoveHs(mol)

    if add_hs:
        mol = Chem.AddHs(mol)

    mol = Chem.Mol(mol)

    if mol.GetNumConformers() == 0:
        Draw.rdDepictor.Compute2DCoords(mol)

    if wedge_bonds:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        Draw.rdDepictor.StraightenDepiction(mol)

    try:
        draw_mol = rdMolDraw2D.PrepareMolForDrawing(
            mol,
            kekulize=kekulize,
            wedgeBonds=wedge_bonds,
            addChiralHs=False,
        )
    except Exception:
        draw_mol = rdMolDraw2D.PrepareMolForDrawing(
            mol,
            kekulize=False,
            wedgeBonds=wedge_bonds,
            addChiralHs=False,
        )

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()

    options.clearBackground = not transparent_background
    options.centreMoleculesBeforeDrawing = centre_molecule
    options.addAtomIndices = atom_indices
    options.addBondIndices = bond_indices
    options.bondLineWidth = line_width
    options.fixedBondLength = fixed_bond_length
    options.minFontSize = int(12 * font_size)
    options.maxFontSize = int(18 * font_size)

    # Good defaults for publication figures.
    options.padding = 0.08
    options.multipleBondOffset = 0.18
    options.additionalAtomLabelPadding = 0.08
    options.prepareMolsBeforeDrawing = False

    drawer.DrawMolecule(draw_mol, legend=legend or "")
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()

    output_svg = Path(output_svg)
    output_svg.write_text(svg, encoding="utf-8")

    return svg

def main():

    chebi_16349 = "/home/jackmcgoldrick/2026_05_20_full_chebi_vs_vmh_stereomapper/chebi_molfiles/CHEBI_16349.mol"

    structure_to_publication_svg(
        structure=chebi_16349,
        output_svg="chebi_16349.svg",
        width=700,
        height=500,
        line_width=2.2,
        fixed_bond_length=38,
        font_size=1.0,
        transparent_background=True,
    )

    # # diff versions of L-alanine
    # vmh_ala_l = "/media/JACK/repos/ctf/mets/molFiles/ala_L.mol"
    # chebi_ala_L = "C[C@H](N)C(=O)O"
    # hmdb_ala = "C[C@H](N)C(O)=O"

    # structure_to_publication_svg(
    #     structure=vmh_ala_l,
    #     output_svg="vmh_ala_L.svg",
    #     width=700,
    #     height=500,
    #     line_width=2.2,
    #     fixed_bond_length=38,
    #     font_size=1.0,
    #     transparent_background=True,    
    # )

    # structure_to_publication_svg(
    #     structure=chebi_ala_L,
    #     output_svg="chebi_ala_L.svg",
    #     width=700,
    #     height=500,
    #     line_width=2.2,
    #     fixed_bond_length=38,
    #     font_size=1.0,
    #     transparent_background=True,    
    # )

    # structure_to_publication_svg(
    #     structure=hmdb_ala,
    #     output_svg="hmdb_ala.svg",
    #     width=700,
    #     height=500,
    #     line_width=2.2,
    #     fixed_bond_length=38,   
    #     font_size=1.0,
    #     transparent_background=True,
    # )

    # chebi_192708 = "[2H]C([2H])(CC[C@H](N)C(=O)O)NC(N)=O"
    # chebi_16349 = "NC(=O)NCCC[C@H](N)C(=O)O"

    # structure_to_publication_svg(
    #     structure=chebi_192708,
    #     output_svg="chebi_192708.svg",
    #     width=700,
    #     height=500,
    #     line_width=2.2,
    #     fixed_bond_length=38,
    #     font_size=1.0,
    #     transparent_background=True,
    # )

    # structure_to_publication_svg(
    #     structure=chebi_16349,
    #     output_svg="chebi_16349.svg",
    #     width=700,
    #     height=500,
    #     line_width=2.2,
    #     fixed_bond_length=38,
    #     font_size=1.0,
    #     transparent_background=True,
    # )

    # chebi_61553 = "O=P(O)(O)OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"
    # chebi_57634 = "O=P([O-])([O-])OC[C@H]1O[C@](O)(CO)[C@@H](O)[C@@H]1O"


    # structure_to_publication_svg(
    #     structure=chebi_61553,
    #     output_svg="chebi_61553.svg",
    #     width=700,
    #     height=500,
    #     line_width=2.2,
    #     fixed_bond_length=38,
    #     font_size=1.0,
    #     transparent_background=True,
    # )

    # structure_to_publication_svg(
    #     structure=chebi_57634,
    #     output_svg="chebi_57634.svg",
    #     width=700,
    #     height=500,
    #     line_width=2.2,
    #     fixed_bond_length=38,
    #     font_size=1.0,
    #     transparent_background=True,
    # )
#     d_ala_smiles = "C[C@@H](N)C(=O)O"
#     l_ala_smiles = "C[C@H](N)C(=O)O"

#     structure_to_publication_svg(
#     structure=d_ala_smiles,
#     output_svg="d_ala.svg",
#     width=700,
#     height=500,
#     line_width=2.2,
#     fixed_bond_length=38,
#     font_size=1.0,
#     transparent_background=True,
# )
    
#     structure_to_publication_svg(
#     structure=l_ala_smiles,
#     output_svg="l_ala.svg",
#     width=700,
#     height=500,
#     line_width=2.2,
#     fixed_bond_length=38,
#     font_size=1.0,
#     transparent_background=True,
# )
    
if __name__ == "__main__":    
    main()