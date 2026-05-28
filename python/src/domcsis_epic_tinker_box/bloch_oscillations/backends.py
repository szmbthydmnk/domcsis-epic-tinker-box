from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


def get_fake_backend(name: str):
    from qiskit_ibm_runtime import fake_provider

    candidates = {
        "brisbane": ["FakeBrisbaneV2", "FakeBrisbane"],
        "sherbrooke": ["FakeSherbrookeV2", "FakeSherbrooke"],
        "almaden": ["FakeAlmadenV2", "FakeAlmaden"],
    }

    key = name.lower()
    if key not in candidates:
        raise ValueError(f"Unknown fake backend name: {name}")

    for cls_name in candidates[key]:
        cls = getattr(fake_provider, cls_name, None)
        if cls is not None:
            return cls()

    raise RuntimeError(
        f"Could not find a fake backend class for '{name}' in qiskit_ibm_runtime."
    )


def make_noisy_density_sim(fake_backend):
    noise_model = NoiseModel.from_backend(fake_backend)
    return AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
    )
