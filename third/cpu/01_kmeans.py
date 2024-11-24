import threading
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import psutil
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


@dataclass
class BenchmarkResult:
    dataset_size: int
    n_clusters: int
    execution_time: float
    iterations: int
    inertia: float
    cpu_usage: List[float]
    memory_usage: List[float]


class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self._stop_flag = threading.Event()
        self.process = psutil.Process()

    def start(self):
        self.cpu_usage = []
        self.memory_usage = []
        self._stop_flag.clear()

        def monitor():
            while not self._stop_flag.is_set():
                try:
                    # CPU 사용률 측정 (interval=None으로 설정하여 즉시 측정)
                    cpu = self.process.cpu_percent(interval=None)
                    # 메모리 사용률 측정 (RSS를 전체 메모리로 나눔)
                    mem = (
                        self.process.memory_info().rss
                        / psutil.virtual_memory().total
                        * 100
                    )

                    self.cpu_usage.append(cpu)
                    self.memory_usage.append(mem)
                    time.sleep(0.1)  # 100ms 간격으로 측정
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    break

        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()

    def stop(self):
        self._stop_flag.set()
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join()

        # 빈 결과 처리
        if not self.cpu_usage:
            self.cpu_usage = [0.0]
        if not self.memory_usage:
            self.memory_usage = [0.0]


class KMeansBenchmark:
    def __init__(self):
        self.resource_monitor = ResourceMonitor()

    def generate_dataset(
        self,
        n_samples: int,
        n_features: int = 10,
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> np.ndarray:
        """테스트용 데이터셋 생성"""
        X, _ = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            random_state=random_state,
        )
        return X

    def run_single_benchmark(
        self,
        X: np.ndarray,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> tuple:
        """단일 K-means 클러스터링 실행"""
        # 모니터링 시작
        self.resource_monitor.start()

        # K-means 실행
        start_time = time.time()
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        kmeans.fit(X)
        execution_time = time.time() - start_time

        # 모니터링 중지
        self.resource_monitor.stop()

        return (
            execution_time,
            kmeans.n_iter_,
            kmeans.inertia_,
            self.resource_monitor.cpu_usage,
            self.resource_monitor.memory_usage,
        )

    def run_benchmark(
        self,
        n_samples: List[int] = [1000, 10000, 100000],
        n_features: int = 10,  # 차원의 수.
        n_clusters: List[int] = [3, 5, 10],
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> List[BenchmarkResult]:
        results = []

        for samples in n_samples:
            for clusters in n_clusters:
                print(
                    f"\nRunning benchmark for {samples} samples, {clusters} clusters..."
                )

                # 데이터셋 생성
                X = self.generate_dataset(
                    n_samples=samples,
                    n_features=n_features,
                    n_clusters=clusters,
                    random_state=random_state,
                )

                # 벤치마크 실행
                execution_time, iterations, inertia, cpu_usage, memory_usage = (
                    self.run_single_benchmark(
                        X, clusters, n_init, max_iter, random_state
                    )
                )

                # 결과 저장
                result = BenchmarkResult(
                    dataset_size=samples,
                    n_clusters=clusters,
                    execution_time=execution_time,
                    iterations=iterations,
                    inertia=inertia,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                )
                results.append(result)

                # 결과 출력
                print(f"Execution time: {execution_time:.2f} seconds")
                print(f"Iterations: {iterations}")
                print(f"Inertia: {inertia:.2f}")
                print(f"Average CPU Usage: {np.mean(cpu_usage):.2f}%")
                print(f"Average Memory Usage: {np.mean(memory_usage):.2f}%")
                print(f"Peak CPU Usage: {max(cpu_usage):.2f}%")
                print(f"Peak Memory Usage: {max(memory_usage):.2f}%")

        return results

    def plot_results(
        self,
        results: List[BenchmarkResult],
    ):
        """벤치마크 결과 시각화"""
        plt.style.use("seaborn-v0_8")  # 가독성 좋은 스타일 적용
        fig = plt.figure(figsize=(20, 15))

        # 1. 실행 시간 그래프
        ax1 = plt.subplot(2, 2, 1)
        cluster_sizes = sorted(set(r.n_clusters for r in results))
        for n_clusters in cluster_sizes:
            data = [
                (r.dataset_size, r.execution_time)
                for r in results
                if r.n_clusters == n_clusters
            ]
            if data:  # 데이터가 있는 경우에만 플롯
                sizes, times = zip(
                    *sorted(data)
                )  # 정렬하여 선 그래프가 교차하지 않도록
                ax1.plot(
                    sizes,
                    times,
                    "o-",
                    linewidth=2,
                    label=f"{n_clusters} clusters",
                    markersize=8,
                )

        ax1.set_xlabel("Dataset Size", fontsize=12)
        ax1.set_ylabel("Execution Time (s)", fontsize=12)
        ax1.set_title("Execution Time vs Dataset Size", fontsize=14, pad=20)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")  # 로그 스케일로 변경하여 가독성 향상

        # 2. CPU 사용률 그래프
        ax2 = plt.subplot(2, 2, 2)
        for result in results:
            if result.cpu_usage:  # CPU 사용률 데이터가 있는 경우에만 플롯
                time_points = np.arange(len(result.cpu_usage)) * 0.1
                ax2.plot(
                    time_points,
                    result.cpu_usage,
                    linewidth=2,
                    label=f"{result.dataset_size} samples, {result.n_clusters} clusters",
                )

        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("CPU Usage (%)", fontsize=12)
        ax2.set_title("CPU Usage Over Time", fontsize=14, pad=20)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. 메모리 사용률 그래프
        ax3 = plt.subplot(2, 2, 3)
        for result in results:
            if result.memory_usage:  # 메모리 사용률 데이터가 있는 경우에만 플롯
                time_points = np.arange(len(result.memory_usage)) * 0.1
                ax3.plot(
                    time_points,
                    result.memory_usage,
                    linewidth=2,
                    label=f"{result.dataset_size} samples, {result.n_clusters} clusters",
                )

        ax3.set_xlabel("Time (seconds)", fontsize=12)
        ax3.set_ylabel("Memory Usage (%)", fontsize=12)
        ax3.set_title("Memory Usage Over Time", fontsize=14, pad=20)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. 반복 횟수 vs 데이터셋 크기
        ax4 = plt.subplot(2, 2, 4)
        for n_clusters in cluster_sizes:
            data = [
                (r.dataset_size, r.iterations)
                for r in results
                if r.n_clusters == n_clusters
            ]
            if data:  # 데이터가 있는 경우에만 플롯
                sizes, iters = zip(*sorted(data))
                ax4.plot(
                    sizes,
                    iters,
                    "o-",
                    linewidth=2,
                    label=f"{n_clusters} clusters",
                    markersize=8,
                )

        ax4.set_xlabel("Dataset Size", fontsize=12)
        ax4.set_ylabel("Number of Iterations", fontsize=12)
        ax4.set_title("Iterations vs Dataset Size", fontsize=14, pad=20)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale("log")

        plt.tight_layout(pad=3.0)  # 여백 조정

        # 그래프를 저장하고 표시
        try:
            plt.savefig("kmeans_benchmark_results.png", dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"Warning: Could not save plot: {e}")

        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot: {e}")

        # 결과 데이터 출력 (디버깅용)
        print("\nPlotting Data Summary:")
        for result in results:
            print(
                f"\nDataset Size: {result.dataset_size}, Clusters: {result.n_clusters}"
            )
            print(f"Execution Time: {result.execution_time:.2f}s")
            print(
                f"CPU Usage: {len(result.cpu_usage)} points, "
                f"Range: [{min(result.cpu_usage):.1f}, {max(result.cpu_usage):.1f}]"
            )
            print(
                f"Memory Usage: {len(result.memory_usage)} points, "
                f"Range: [{min(result.memory_usage):.1f}, {max(result.memory_usage):.1f}]"
            )


# 사용 예시
def main():
    benchmark = KMeansBenchmark()

    # 벤치마크 실행
    results = benchmark.run_benchmark(
        n_samples=[100_000, 500_000],  # 샘플 크기
        n_features=128,  # 특성 수
        n_clusters=[5, 7],  # 클러스터 수
        n_init=10,  # 초기화 횟수
        max_iter=300,  # 최대 반복 횟수
        random_state=42,  # 랜덤 시드
    )

    # 결과 시각화
    benchmark.plot_results(results)


if __name__ == "__main__":
    main()
