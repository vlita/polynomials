import qcelemental as qcel
from typing import Dict
from pprint import pprint as pp
import apnet_pt
import qcml_mcp


def estimate_timing_for_qcel_molecule(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
units angstrom
    """,
    method: str = "MP2",
    basis_set: str = "aug-cc-pVDZ",
    manybody: bool = True,
) -> Dict:
    """
    Estimate computational timing for a molecular system.

    Computes the necessary variables for timing estimation for monomers and
    dimers. When manybody is True, it computes the timing for the dimer and
    each monomer separately to mimic a supermolecular interaction energy
    calculation. Manybody should also be set when counterpoise correction (CP)
    is requested.

    Allowed methods: 'mp2', 'hf', 'b2plyp-d3', 'b3lyp-d3', 'pbe-d3', 'm05-2x',
    'wb97x-v', 'wb97x-d', 'fno-ccsd', 'fno-ccsd(t)'.

    Parameters
    ----------
    p4_string : str, optional
        Molecular geometry in Psi4 format with charge, multiplicity, atomic
        symbols, coordinates, and units. Format:
        '''
        <charge_mon1> <multiplicity_mon1>
        <atom_symbol> <x> <y> <z>
        <atom_symbol> <x> <y> <z>
        --
        <charge_mon2> <multiplicity_mon2>
        units <unit>
        '''
        Note that "--" separates different molecules in the input string.
        Default is a water molecule geometry.
    method : str, optional
        Computational method to use. Default is 'MP2'.
    basis_set : str, optional
        Basis set to use. Default is 'aug-cc-pVDZ'.
    manybody : bool, optional
        Whether to compute timing for dimer and monomers separately.
        Default is True.

    Returns
    -------
    dict
        Dictionary containing:
        - "estimated_compute_time_seconds" : float
            Estimated computational time in seconds.
        - "geometry" : str
            Molecular geometry in Psi4 format.
    """
    qcel_molecule = qcel.models.Molecule.from_data(p4_string)
    if qcml_mcp.timings.is_psi4_installed() is False:
        print("Psi4 is not installed. Please install Psi4 to use this function.")
    mols = [qcel_molecule]
    if manybody and qcel_molecule.fragments_:
        for n, i in enumerate(qcel_molecule.fragments_):
            mols.append(qcel_molecule.get_fragment(n))

    method = method.lower()
    time_seconds = 0.0
    for mol in mols:
        n_occupied, n_virtual, np_total, nbf_aux = (
            qcml_mcp.timings.estimate_timings.compute_psi4_time_estimation_variables(
                mol,
                basis_set,
            )
        )
        input_vars = {
            "nocc": n_occupied,
            "nvirt": n_virtual,
            "nbf_aux": nbf_aux,
            "np_total": np_total,
        }
        result = qcml_mcp.timings.estimate_timings.predict_timing(method, input_vars)
        time_seconds += result["time_seconds"]
    return {
        "estimated_compute_time_seconds": time_seconds,
        "geometry": mol.to_string("psi4"),
    }


def predict_dAPNet2_error_estimates_QCMLForge(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
units angstrom
    """,
    starting_level_of_theory: str = "MP2/aug-cc-pVTZ/CP",
) -> Dict:
    """
    Predict error between a starting level of theory and CCSD(T)/CBS/CP reference.

    Estimates the interaction energy error using the dAPNet2 model in QCMLForge
    for a single molecular complex. The p4_string defines the molecular geometry
    in Psi4 format.

    Acceptable starting_level_of_theory values currently only include:
    [
    "B3LYP-D3/aug-cc-pVTZ/unCP",
    "B2PLYP-D3/aug-cc-pVTZ/unCP",
    "wB97X-V/aug-cc-pVTZ/CP",
    "SAPT0/aug-cc-pVDZ/SA",
    "MP2/aug-cc-pVTZ/CP",
    "HF/aug-cc-pVDZ/CP",
    ]

    Parameters
    ----------
    p4_string : str, optional
        Molecular geometry in Psi4 format with charge, multiplicity, atomic
        symbols, coordinates, and units. Format:
        '''
        <charge_mon1> <multiplicity_mon1>
        <atom_symbol> <x> <y> <z>
        <atom_symbol> <x> <y> <z>
        --
        <charge_mon2> <multiplicity_mon2>
        units <unit>
        '''
        Note that "--" is used to separate different molecules in the input string
        but is not required for monomers. Default is a water dimer geometry.
    starting_level_of_theory : str, optional
        Level of theory for which to estimate error relative to CCSD(T)/CBS/CP.
        Default is "MP2/aug-cc-pVTZ/CP".

    Returns
    -------
    dict
        Dictionary containing:
        - "ERROR ESTIMATES (kcal/mol)" : array
            Predicted error between starting_level_of_theory and CCSD(T)/CBS/CP.
    """
    mol = qcel.models.Molecule.from_data(p4_string)
    IE_pred = apnet_pt.pretrained_models.dapnet2_model_predict(
        [mol],
        compile=False,
        m1=starting_level_of_theory,
        m2="CCSD(T)/CBS/CP",
    )
    return {
        "ERROR ESTIMATES (kcal/mol)": IE_pred.tolist(),
    }


def main():
    # Example usage of the timing estimation function
    timing_estimate = estimate_timing_for_qcel_molecule()
    print("Timing Estimate:")
    pp(timing_estimate)
    # Example usage of the dAPNet2 error estimation function
    error_estimate = predict_dAPNet2_error_estimates_QCMLForge()
    print("\ndAPNet2 Error Estimate:")
    print(error_estimate)
    return


if __name__ == "__main__":
    main()
