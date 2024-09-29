import tracemalloc
import time
import random

def random_math_operations(n):
    """Performs random mathematical operations."""
    result = 0
    for _ in range(n):
        a = random.random()
        b = random.random()
        c = random.choice([a + b, a - b, a * b, a / b if b != 0 else 0])
        result += c
    return result

def measure_memory_cpu_usage(func, *args, **kwargs):
    # Start measuring CPU time
    cpu_start_time = time.process_time()

    # Start measuring memory allocation
    tracemalloc.start()

    # Execute the function
    result = func(*args, **kwargs)

    # Stop measuring memory allocation
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Stop measuring CPU time
    cpu_end_time = time.process_time()

    memory_usage = current / 10**6  # Convert to MB
    peak_memory_usage = peak / 10**6  # Convert to MB
    cpu_time_used = cpu_end_time - cpu_start_time  # In seconds

    return {
        "result": result,
        "memory_usage_mb": memory_usage,
        "peak_memory_usage_mb": peak_memory_usage,
        "cpu_time_used_sec": cpu_time_used
    }

if __name__ == "__main__":
    n_operations = 1000
    usage_stats = measure_memory_cpu_usage(random_math_operations, n_operations)

    print(f"Result of computation: {usage_stats['result']}")
    print(f"Memory usage: {usage_stats['memory_usage_mb']} MB")
    print(f"Peak memory usage: {usage_stats['peak_memory_usage_mb']} MB")
    print(f"CPU time used: {usage_stats['cpu_time_used_sec']} seconds")
