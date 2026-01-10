from itertools import product

import numpy as np
import pytest

from dataeval.shift._update_strategies import LastSeenUpdateStrategy, ReservoirSamplingUpdateStrategy


@pytest.mark.required
class TestUpdateReference:
    n = [3, 50]
    n_features = [1, 10]
    update_method = [LastSeenUpdateStrategy, ReservoirSamplingUpdateStrategy]
    tests_update = list(product(n, n_features, update_method))
    n_tests_update = len(tests_update)

    @pytest.fixture(scope="class")
    def update_params(self, request):
        return self.tests_update[request.param]

    @pytest.mark.parametrize("update_params", list(range(n_tests_update)), indirect=True)
    def test_update_reference(self, update_params):
        n, n_features, update_method = update_params
        n_ref = np.random.randint(1, n)
        n_test = np.random.randint(1, 2 * n)
        X_ref = np.random.rand(n_ref * n_features).reshape(n_ref, n_features)
        X = np.random.rand(n_test * n_features).reshape(n_test, n_features)
        update_method = update_method(n)
        X_ref_new = update_method(X_ref, X)

        assert X_ref_new.shape[0] <= n
        if isinstance(update_method, LastSeenUpdateStrategy):
            assert (X_ref_new[-1] == X[-1]).all()
