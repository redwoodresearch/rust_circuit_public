# Cursed import hack file
# rust_circuit/python/rust_circuit is a legit python package called rust_circuit. When you `maturin dev`, rust is compiled and put into this folder, next to this file
# If you installed rust_circuit seperately, and have cloned this repo without compiling
# then you might accidentally import this version
# this script imports another installed compiled version of rust_circuit if it's installed
import os
import site

installed_rust_circuit_path = site.getsitepackages()[0] + "/rust_circuit"
if os.path.exists(installed_rust_circuit_path):
    import warnings

    warnings.warn("trying to rust_circuit local version without compiled rust. Using another installed version.")
    import importlib.util
    import sys

    rust_circuit_files = os.listdir(installed_rust_circuit_path)
    rust_so_file = [x for x in rust_circuit_files if x.startswith("_rust.cpython")][0]
    spec = importlib.util.spec_from_file_location("_rust", installed_rust_circuit_path + "/" + rust_so_file)
    foo = importlib.util.module_from_spec(spec)  # type: ignore
    print(foo)
    sys.modules["_rust"] = foo
    from _rust import *  # type: ignore
else:
    raise ImportError("rust_circuit is not compiled, run `maturin dev` to compile and install")
