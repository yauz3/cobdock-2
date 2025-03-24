"""
obrms
obrms: Computes the heavy-atom RMSD of identical compound structures.
Usage: obrms reference_file [test_file]
Options:
	 -o, --out        re-oriented test structure output
	 -f, --firstonly  use only the first structure in the reference file
	 -m, --minimize   compute minimum RMSD
	 -x, --cross      compute all n^2 RMSDs between molecules of reference file
	 -s, --separate   separate reference file into constituent molecules and report best RMSD
	 -h, --help       help message
Command line parse error: test file is required but missing

"""

import os
import shutil
import subprocess
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))


def rmsd_mti():
    # List of ligands from the MTi test set
    mti_test = [
        '3erd-A', '4asd-A', '1gs4-A', '1ki2-A', '1rv1-A', '1fm6-ABD', '1z95-A', '3h0a-ABD', '2p16-AL',
        '3mxf-A', '2hyy-A', '1m48-A', '3vn2-AC', '2gqg-A', '3clx-A', '2yek-A', '2ydo-A', '1fm9-ABD',
        '4ag8-A', '2w26-AB', '4mxo-A', '3dzy-ADE', '4agd-A', '1t4e-A', '4ey7-A', '4ey6-A'
    ]

    rmsd_list = []  # Will hold RMSD values for all ligands

    # Loop through each ligand
    for ligand in mti_test:
        # Define input path (reference/native ligand pose)
        input_path = f"{current_dir}/local_docking/data/mti/{ligand}_ligand.mol2"

        # Define output path (predicted pose after docking)
        output_path = f"{current_dir}/cobdock_2/local_docking/poses/mti_multimodel_1/{ligand}_protonated_pH=7_4/{ligand}_protonated_pH=7_4_conf_001.mol2"

        try:
            # Step 1: Run obrms in match mode (-m)
            subprocess.run(["obrms", input_path, output_path], capture_output=True, text=True, check=True)

            # Step 2: Run obrms again to capture output for parsing RMSD
            completed_process = subprocess.run(["obrms", input_path, output_path],
                                               capture_output=True, text=True, check=True)

            # Step 3: Parse RMSD value from the command output
            output_lines = completed_process.stdout.strip().split("\n")
            rmsd_line = output_lines[-1]  # Assume last line contains the RMSD info
            rmsd_value = float(rmsd_line.split()[-1])  # Get the numeric RMSD value

            # Print the result for this ligand
            print(f"{ligand} - RMSD value: {rmsd_value}")

            # Store in list
            rmsd_list.append(rmsd_value)

        except subprocess.CalledProcessError as e:
            # Handle obrms errors gracefully and store None
            print(f"Error calculating RMSD for {ligand}: {e}")
            rmsd_list.append(None)

        except Exception as e:
            # Catch unexpected errors (e.g., file not found)
            print(f"Unexpected error with {ligand}: {e}")
            rmsd_list.append(None)

    # Final output of all RMSD values
    print("\nAll RMSD values:")
    print(rmsd_list)