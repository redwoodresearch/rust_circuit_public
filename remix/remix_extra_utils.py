import subprocess


def check_rust_circuit_version():
    correct_version_string = open("requirements_rust_circuit.txt", "rb").read().strip()
    current_version_string = subprocess.check_output("conda run pip freeze | grep rust_circuit", shell=True).strip()
    if correct_version_string != current_version_string:
        raise Exception(
            "You have the wrong rust_circuit version. Please run `./update_remix_machines.sh` and restart python, or contact a TA."
        )
