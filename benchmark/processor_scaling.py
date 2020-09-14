from benchmark import get_models
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
from joblib import Memory
import multiprocessing.dummy
import threading
import os
import sys

memory = Memory('/tmp/cachedir', verbose=1)
matplotlib.rcParams['text.usetex'] = True
plt.rc('font', family='serif')


@memory.cache
def cpu_experiment(model_name, test_rows, iterations):
    model = get_models(model_name)[0]
    n_cpu_cores = [x for x in range(2, 42)]

    # warmup
    model.xgb_model.set_param({"predictor": "gpu_predictor"})
    dtest = model.dataset.get_test_dmat(test_rows)
    model.xgb_model.predict(dtest, pred_contribs=True)
    results = pd.DataFrame(columns=["CPU Cores", "Time", "Throughput(rows/s)"])
    dtest = model.dataset.get_test_dmat(test_rows)
    for n in n_cpu_cores:
        for i in range(iterations):
            model.xgb_model.set_param({"predictor": "cpu_predictor", "nthread": n})
            start = time.perf_counter()
            shap = model.xgb_model.predict(dtest, pred_contribs=True)
            cpu_time = time.perf_counter() - start
            results = results.append(
                {"CPU Cores": n, "Time": cpu_time, "Throughput(rows/s)": test_rows / cpu_time},
                ignore_index=True)
            print(results)

    print(results)
    return results


def plot_cpu_results(results):
    plt.clf()
    sns.lineplot(data=results, x="CPU Cores", y="Throughput(rows/s)")
    plt.tight_layout()
    plt.savefig("cpu_scaling.pdf")


def run_part(idx, dtest, xgb_model):
    print("Computing shap for {} rows on device {}".format(dtest.num_row(), idx))
    # dtest = xgb.DMatrix(X)
    # warmup
    start = time.perf_counter()
    xgb_model.set_param({"gpu_id": idx})
    xgb_shap = xgb_model.predict(dtest, pred_contribs=True)
    print("shap time {}".format(time.perf_counter() - start))


def run_multi_gpu(n_gpus, xgb_models, dtest_parts):
    pool = [threading.Thread(target=run_part, args=(idx, dtest_parts[idx], xgb_models[idx])) for idx
            in
            range(n_gpus)]
    for p in pool:
        p.start()
    for p in pool:
        p.join()


@memory.cache
def gpu_experiment(model_name, test_rows, iterations):
    model = get_models(model_name)[0]
    model.xgb_model.set_param({"predictor": "gpu_predictor"})
    n_gpus = [x for x in range(1, 9)]
    results = pd.DataFrame(columns=["GPUs", "Time", "Throughput(rows/s)"])
    print("Copying models...")
    pool = multiprocessing.dummy.Pool(n_gpus[-1])
    xgb_models = pool.map(lambda x: model.xgb_model.copy(), range(n_gpus[-1]))
    for n in n_gpus:
        dtest_parts = model.dataset.get_test_dmat_parts(test_rows, n)
        # warmup
        run_multi_gpu(n, xgb_models, dtest_parts)
        for i in range(iterations):
            start = time.perf_counter()
            run_multi_gpu(n, xgb_models, dtest_parts)
            gpu_time = time.perf_counter() - start
            results = results.append(
                {"GPUs": n, "Time": gpu_time, "Throughput(rows/s)": test_rows / gpu_time},
                ignore_index=True)
            print(results)

    results["GPUs"] = results["GPUs"].astype(int)
    print(results)
    return results


def plot_gpu_results(results):
    plt.clf()
    sns.barplot(x="GPUs", y="Throughput(rows/s)", color="royalblue", data=results, ci=0.95)
    plt.tight_layout()
    plt.savefig("gpu_scaling.pdf")


def main():
    iterations = 5
    cpu_test_rows = 100000
    gpu_test_rows = 1000000
    cpu_df = cpu_experiment("cal_housing-med", cpu_test_rows, iterations)
    plot_cpu_results(cpu_df)
    gpu_df = gpu_experiment("cal_housing-med", gpu_test_rows, iterations)
    plot_gpu_results(gpu_df)


if __name__ == "__main__":
    main()
