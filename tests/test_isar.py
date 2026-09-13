import numpy as np

from csi.isar import (
    Radar,
    _ImageMeasure,
    f1_score,
    image_from_rect,
    keep_random,
    phase_history,
    pipeline_image,
    pipeline_zero_fill,
    polar_to_rect,
    pseudo_aircraft,
    render_scene,
    sparsity_energy,
)

N = 64


def test_image_measure_adjoint():
    idx = keep_random(N * N, 0.2, seed=0)
    assert _ImageMeasure((N, N), idx).adjoint_check() < 1e-10


def test_full_phase_history_images_the_target():
    radar, scene = Radar(), pseudo_aircraft()
    rect = polar_to_rect(phase_history(scene, radar), radar, N)
    truth = render_scene(scene, radar, N)
    f1, err = f1_score(image_from_rect(rect), truth)
    assert f1 == 1.0 and err < 0.2


def test_h2_image_domain_l1_recovers_at_paper_sampling():
    radar, scene = Radar(), pseudo_aircraft()
    rect = polar_to_rect(phase_history(scene, radar), radar, N)
    truth = render_scene(scene, radar, N)
    idx = keep_random(rect.size, 0.225, seed=0)
    f1, err = f1_score(pipeline_image(rect, idx), truth)
    assert f1 >= 0.9 and err < 1.0
    # and it beats zero-filling by a wide margin on energy concentration
    zf = pipeline_zero_fill(rect, idx)
    m = np.abs(truth) > 0
    e = lambda im: (np.abs(im)[m] ** 2).sum() / (np.abs(im) ** 2).sum()  # noqa: E731
    assert e(pipeline_image(rect, idx)) > 3 * e(zf)


def test_h3_data_is_not_sparse_image_is():
    radar, scene = Radar(), pseudo_aircraft()
    rect = polar_to_rect(phase_history(scene, radar), radar, N)
    truth = render_scene(scene, radar, N)
    assert sparsity_energy(rect, scene.n) < 0.5
    assert sparsity_energy(truth, scene.n) > 0.99
