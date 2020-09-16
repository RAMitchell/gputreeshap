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
import numpy as np
import sys
from contextlib import redirect_stdout
import io
import re
from wurlitzer import pipes

memory = Memory('/tmp/cachedir', verbose=1)
matplotlib.rcParams['text.usetex'] = True
plt.rc('font', family='serif')


def check_accuracy(shap, margin):
    if not np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-1, 1e-1):
        print("Warning: Failed 1e-1 accuracy")
    if not np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-3, 1e-3):
        print("Warning: Failed 1e-3 accuracy")


@memory.cache
def get_stats():
    models = get_models(
        "all")
    df = pd.DataFrame(columns=["Model", "Algorithm", "Time", "Utilisation", "Bins used"])
    for m in models:
        print(m.name)
        dtest = m.dataset.get_test_dmat(100)
        algorithms = ["none", "nf", "ffd", "bfd"]
        for alg in algorithms:
            print("Algorithm: " + alg)
            m.xgb_model.set_param({"predictor": "gpu_predictor", "bin_packing_algorithm": alg})
            with pipes() as (out, err):
                xgb_shap = m.xgb_model.predict(dtest, pred_contribs=True)
            s = out.read()
            for item in s.split("\n"):
                if "Time" in item:
                    runtime = float(re.findall("\d+\.\d+", item)[0])
                if "Utilisation" in item:
                    utilisation = float(re.findall("\d+\.\d+", item)[0])
                if "Bins" in item:
                    bins_used = int(re.findall("\d+", item)[0])

            margin = m.xgb_model.predict(dtest, output_margin=True)
            check_accuracy(xgb_shap, margin)
            df = df.append(
                {"Model": m.name, "Algorithm": alg, "Time": runtime, "Utilisation": utilisation,
                 "Bins used": bins_used},
                ignore_index=True)
            print(df)
    print(df)
    return df


def main():
    df = get_stats()
    print(df.to_latex(index=False))


if __name__ == "__main__":
    main()
