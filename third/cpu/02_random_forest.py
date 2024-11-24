import time
from dataclasses import dataclass
from typing import (
    List,
    Dict,
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class BenchmarkResult:
    n_estimators: int
    training_time: float
    prediction_time: float
    total_time: float


class RandomForestBenchmark:
    def __init__(
            self,
            n_samples: int = 100000,
            n_features: int = 100,
            random_state: int = 42
    ):
        # 데이터셋 생성
        np.random.seed(random_state)
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randint(0, 2, n_samples)
        print(f"Dataset generated: {n_samples} samples with {n_features} features")

    def run_benchmark(self, n_estimators_list: List[int]) -> Dict[int, BenchmarkResult]:
        results = {}

        for n_trees in n_estimators_list:
            print(f"\nTesting RandomForest with {n_trees} trees...")

            # 모델 생성
            model = RandomForestClassifier(
                n_estimators=n_trees,
                n_jobs=-1,  # 모든 CPU 코어 사용
                random_state=42
            )

            # 학습 시간 측정
            train_start = time.time()
            model.fit(self.X, self.y)
            train_time = time.time() - train_start

            # 예측 시간 측정
            pred_start = time.time()
            y_pred = model.predict(self.X)
            pred_time = time.time() - pred_start

            # 총 수행시간
            total_time = train_time + pred_time

            results[n_trees] = BenchmarkResult(
                n_estimators=n_trees,
                training_time=train_time,
                prediction_time=pred_time,
                total_time=total_time
            )

            print(f"Training time: {train_time:.2f} seconds")
            print(f"Prediction time: {pred_time:.2f} seconds")
            print(f"Total time: {total_time:.2f} seconds")

        return results

    def plot_results(self, results: Dict[int, BenchmarkResult]):
        """벤치마크 결과 시각화"""
        n_trees = [r.n_estimators for r in results.values()]
        train_times = [r.training_time for r in results.values()]
        pred_times = [r.prediction_time for r in results.values()]
        total_times = [r.total_time for r in results.values()]

        plt.figure(figsize=(12, 8))

        # 시간 비교 그래프
        plt.plot(n_trees, train_times, 'b-o', label='Training Time')
        plt.plot(n_trees, pred_times, 'r-o', label='Prediction Time')
        plt.plot(n_trees, total_times, 'g-o', label='Total Time')

        plt.xlabel('Number of Trees (n_estimators)')
        plt.ylabel('Time (seconds)')
        plt.title('Random Forest Performance vs Number of Trees')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # x축 눈금을 실제 n_estimators 값으로 설정
        plt.xticks(n_trees)

        try:
            plt.savefig('random_forest_benchmark.png', dpi=300, bbox_inches='tight')
            print("\nPlot saved as 'random_forest_benchmark.png'")
        except Exception as e:
            print(f"\nWarning: Could not save plot: {e}")

        try:
            plt.show()
        except Exception as e:
            print(f"\nWarning: Could not display plot: {e}")


def main():
    # 벤치마크 파라미터
    params = {
        'n_samples': 100000,
        'n_features': 100,
        'random_state': 42
    }

    # 테스트할 트리 개수 목록
    n_estimators_list = [1, 2, 5, 10, 25, 50, 100]

    # 벤치마크 실행
    benchmark = RandomForestBenchmark(**params)
    results = benchmark.run_benchmark(n_estimators_list)

    # 결과 요약 출력
    print("\nSummary:")
    print(f"{'Trees':>6} {'Train(s)':>10} {'Predict(s)':>10} {'Total(s)':>10}")
    print("-" * 40)
    for n_trees, result in results.items():
        print(
            f"{n_trees:>6} {result.training_time:>10.2f} {result.prediction_time:>10.2f} {result.total_time:>10.2f}"
        )

    # 결과 시각화
    benchmark.plot_results(results)


if __name__ == "__main__":
    main()
