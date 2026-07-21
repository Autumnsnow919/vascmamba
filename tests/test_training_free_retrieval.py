import numpy as np

from training_free_retrieval import (
    l2_normalize,
    patient_similarity_matrix,
    select_threshold,
)


def test_patient_set_similarity_is_invariant_to_view_permutation():
    rng = np.random.default_rng(2)
    bmode = l2_normalize(rng.normal(size=(5, 4, 8)).astype("float32"))
    ulm = l2_normalize(rng.normal(size=(5, 4, 8)).astype("float32"))
    density = rng.random(size=(5, 4)).astype("float32")
    valid = np.ones((5, 4), dtype=bool)
    expected = patient_similarity_matrix(
        bmode, ulm, density, valid, bmode_weight=0.5, density_penalty=0.25
    )

    permutation = np.asarray([2, 0, 3, 1])
    actual = patient_similarity_matrix(
        bmode[:, permutation], ulm[:, permutation], density[:, permutation],
        valid[:, permutation], bmode_weight=0.5, density_penalty=0.25,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_threshold_is_selected_from_scores_not_a_fixed_grid():
    labels = np.asarray([0, 0, 1, 1, 1, 1])
    scores = np.asarray([-0.8, 0.2, -0.1, 0.3, 0.5, 0.7])
    threshold = select_threshold(labels, scores)
    prediction = scores >= threshold

    assert prediction.tolist() == [False, True, True, True, True, True]
