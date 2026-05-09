from domcsis_epic_tinker_box.pauli_algebra import generate_pauli_operators
from memory_profiler import profile
import tracemalloc
import cProfile

@profile
def my_func():
    ops = generate_pauli_operators(5)
    return ops

if __name__ == "__main__":
    tracemalloc.start()
    my_func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Peak memory: {peak / 1024**2:.1f} MB")
    
    cProfile.run("generate_pauli_operators(5)")