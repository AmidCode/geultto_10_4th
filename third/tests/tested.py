import hashlib
import os
import time
from typing import (
    Callable,
    Dict,
)

import numpy as np


def run_benchmark(func: Callable, iterations: int = 5) -> Dict[str, float]:
    """
    주어진 함수에 대해 벤치마크를 실행하고 결과를 반환합니다.
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def svd_benchmark(matrix_size: int = 1000) -> None:
    """
    SVD 연산 벤치마크
    """
    matrix = np.random.random((matrix_size, matrix_size))
    np.linalg.svd(matrix)


def matrix_mult_benchmark(matrix_size: int = 1000) -> None:
    """
    행렬 곱셈 벤치마크
    """
    matrix1 = np.random.random((matrix_size, matrix_size))
    matrix2 = np.random.random((matrix_size, matrix_size))
    np.matmul(matrix1, matrix2)


def prime_benchmark(n: int = 100000) -> None:
    """
    소수 계산 벤치마크
    """

    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    primes = [num for num in range(n) if is_prime(num)]


def hash_benchmark(size: int = 1000000) -> None:
    """
    해시 계산 벤치마크
    """
    data = os.urandom(size)
    for _ in range(1000):
        hashlib.sha256(data).digest()


if __name__ == "__main__":
    benchmarks = {
        "SVD": lambda: svd_benchmark(3000),
        "Matrix Multiplication": lambda: matrix_mult_benchmark(3000),
        "Prime Numbers": lambda: prime_benchmark(100000),
        "Hash Calculation": lambda: hash_benchmark(1000000),
    }

    results = {}
    for name, func in benchmarks.items():
        print(f"Running {name} benchmark...")
        results[name] = run_benchmark(func)
        print(f"{name} Results:", {k: f"{v:.3f}" for k, v in results[name].items()})
        print("-" * 50)
