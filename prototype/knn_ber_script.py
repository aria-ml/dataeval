from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

from dataeval.metrics.estimators import BER


def generate_toy_data(num_classes=2, num_feat=32, num_samples_per_class=1000):
    mu_0 = np.ones(num_feat)
    var_0 = 4
    cov = var_0 * np.eye(num_feat)
    samples_per_class = num_samples_per_class

    if num_classes == 2:
        pi_0 = pi_1 = 0.5
        # assumes we increment mu by 1
        delta = np.sqrt(np.ones(num_feat).dot(np.linalg.solve(cov, np.ones(num_feat))))
        arg0 = -delta / 2 + 1 / delta * np.log(pi_0 / pi_1)
        arg1 = -delta / 2 - 1 / delta * np.log(pi_0 / pi_1)
        ber_0 = pi_0 * NormalDist().cdf(arg0) + pi_1 * NormalDist().cdf(arg1)
    else:
        ber_0 = -1

    data = np.concatenate(
        [np.random.multivariate_normal(mean=mu_0 + cls, cov=cov, size=samples_per_class) for cls in range(num_classes)]
    )
    labels = np.concatenate([np.ones(samples_per_class) * cls for cls in range(num_classes)])
    return data, labels, ber_0


if __name__ == "__main__":
    # toy data
    num_classes = 2
    num_feat = 32
    ks = [1, 3, 5, 7, 9, 15, 21]
    num_samp = 2500  # per class
    data, labels, ber_0 = generate_toy_data(
        num_classes=num_classes,
        num_feat=num_feat,
        num_samples_per_class=num_samp,
    )

    # BER
    m_mst = BER(data, labels, "MST")
    ber_mst = m_mst.evaluate()

    ber_knn = []
    for kdx, k in enumerate(ks):
        # BER
        m_knn = BER(data, labels, "KNN", k=k)
        ber_knn.append(m_knn.evaluate())

    # some linter didn't like that call to rcParams, so now hardcoded
    cols = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    plt.figure()
    plt.plot(ks, [bk["ber"] for bk in ber_knn], ":", marker=".", lw=2, color=cols[0])
    plt.plot(ks, [bk["ber_lower"] for bk in ber_knn], marker=".", lw=2, label="KNN", color=cols[0])
    plt.plot(plt.xlim(), ber_mst["ber"] * np.ones(2), ":", marker=".", lw=2, color=cols[1])
    plt.plot(plt.xlim(), ber_mst["ber_lower"] * np.ones(2), marker=".", lw=2, label="MST", color=cols[1])
    plt.plot(plt.xlim(), ber_0 * np.ones(2), marker=".", lw=2, label="True BER", color="k")
    plt.title(f"{num_classes} classes, {num_feat} features")
    plt.legend()
    plt.savefig(f"src/dataeval/_prototype/figs/ber_knn_convergence_{num_feat}feat.png")
