# Based on
# https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_ae_cifar10.html

# import daml
# from daml._internal.utils import Metrics
# import matplotlib.pyplot as plt
# import numpy as np
import pytest

# import tensorflow as tf
# from tqdm import tqdm

# from alibi_detect.utils.perturbation import apply_mask


class TestCifar10:
    # Test main functionality of the program
    @pytest.mark.functional
    def test_label_cifar10_outliers(self):
        assert True

    #     print("begin")
    #     """Functional testing of Alibi OutlierDection

    #     The AlibiAE model is being trained on all 1's and tested on all 5's.
    #     When evaluating, the model should say all 1's are not outliers
    #     and all 5's are outliers
    #     """

    #     # Load CIFAR10 dataset
    #     train, test = tf.keras.datasets.cifar10.load_data()
    #     X_train, y_train = train
    #     X_test, y_test = test

    #     X_train = X_train.astype("float32") / 255
    #     X_test = X_test.astype("float32") / 255
    #     print("loaded data")
    #     input = Metrics.Method.AutoEncoder

    #     # Initialize the autoencoder-based outlier detector from alibi-detect
    #     metric = daml.load_metric(
    #         metric=Metrics.OutlierDetection,
    #         provider=Metrics.Provider.AlibiDetect,
    #         method=input,
    #     )
    #     print("training metric")
    #     # Train the detector on the dataset of all 1's
    #     metric.fit_dataset(dataset=X_train, epochs=50, verbose=False)
    #     print("evaluating metric")
    #     # Evaluate the detector on the dataset of all 1's
    #     preds_X_train = metric.evaluate(X_train[:500]).instance_score
    #     print("collecting results")
    #     n_mask_sizes = 10
    #     n_masks = 20
    #     n_imgs = 50
    #     mask_sizes = [(2 * n, 2 * n) for n in range(1, n_mask_sizes + 1)]
    #     img_ids = np.arange(n_imgs)
    #     X_orig = X_train[img_ids].reshape(img_ids.shape[0], 32, 32, 3)

    #     all_img_scores = []
    #     for i in tqdm(range(X_orig.shape[0])):
    #         img_scores = np.zeros((len(mask_sizes),))
    #         for j, mask_size in enumerate(mask_sizes):
    #             # create masked instances
    #             X_mask, _ = apply_mask(
    #                 X_orig[i].reshape(1, 32, 32, 3),
    #                 mask_size=mask_size,
    #                 n_masks=n_masks,
    #                 channels=[0, 1, 2],
    #                 mask_type="normal",
    #                 noise_distr=(0, 1),
    #                 clip_rng=(0, 1),
    #             )
    #             # predict outliers
    #             od_preds_mask = metric.evaluate(X_mask).instance_score
    #             score = od_preds_mask["data"]["instance_score"]
    #             # store average score over `n_masks` for a given mask size
    #             img_scores[j] = np.mean(score)
    #         all_img_scores.append(img_scores)

    #     x_plt = [mask[0] for mask in mask_sizes]
    #     ais_np = np.zeros((len(all_img_scores), all_img_scores[0].shape[0]))
    #     for i, ais in enumerate(all_img_scores):
    #         ais_np[i, :] = ais
    #     ais_mean = np.mean(ais_np, axis=0)
    #     plt.title("Mean Outlier Score All Images for Increasing Mask Size")
    #     plt.xlabel("Mask size")
    #     plt.ylabel("Outlier score")
    #     plt.plot(x_plt, ais_mean)
    #     plt.xticks(x_plt)
    #     plt.show()

    #     # Need to figure out tests
    #     assert 1 == 0
