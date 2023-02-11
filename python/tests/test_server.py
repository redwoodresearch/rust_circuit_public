import rust_circuit

if __name__ == "__main__":
    rust_circuit.circuit_server_serve(
        "0.0.0.0:9876", rust_circuit.TensorCacheRrfs(10_000_000, 500_000_000, 40_000_000_000, "cuda:0")
    )
