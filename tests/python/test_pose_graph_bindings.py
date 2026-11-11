"""Python bindings used by external schedulers (rls_gbp): pose-graph types, accessors and the
incremental message-passing API. Run with the built module on the path, e.g.
    PYTHONPATH=<dir containing gsolver/ with __init__.py and _gsolver*.so> pytest tests/python
"""

import numpy as np
import pytest

g = pytest.importorskip("gsolver")


def se2(x, y, t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def se3(t, w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / (th if th else 1.0)
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def se2_loop(n=6, seed=0):
    rng = np.random.default_rng(seed)
    fg, truth = g.FactorGraph(), [se2(0, 0, 0)]
    for _ in range(1, n):
        truth.append(truth[-1] @ se2(1, 0, 2 * np.pi / n))
    for k in range(n):
        fg.add_variable(g.SE2Pose(f"P{k}", truth[k] @ se2(*rng.normal(0, 0.1, 3)), np.eye(3) * 1e8))
    fg.add_factor(g.SE2Prior("PR", truth[0], np.eye(3) * 1e-2), ["P0"])
    for k in range(n):
        j = (k + 1) % n
        fg.add_factor(g.SE2Odometry(f"O{k}", np.linalg.inv(truth[k]) @ truth[j], np.eye(3) * 1e-2), [f"P{k}", f"P{j}"])
    return fg, truth


def tick(fg):
    for v in fg.variable_nodes:
        for i in range(v.num_edges):
            v.send_message(i)
        v.mark_messages_processed()
    sent = 0
    for f in fg.factor_nodes:
        f.update_factor_if_new()
        sent += f.send_all_messages_if_new()
    for v in fg.variable_nodes:
        v.update_mu_sigma()
    return sent


def test_se2_incremental_tick_converges():
    fg, truth = se2_loop()
    for _ in range(100):
        sent = tick(fg)
    assert sent == 1 + 2 * 6
    assert max(np.abs(fg.variable_nodes[k].mu - truth[k]).max() for k in range(6)) < 1e-9


def test_nothing_new_nothing_sent_and_accessors():
    fg, _ = se2_loop()
    tick(fg)
    f, v = fg.factor_nodes[1], fg.variable_nodes[1]
    assert not f.update_factor_if_new() and f.send_all_messages_if_new() == 0
    assert np.asarray(f.error).shape == (3,) and np.asarray(f.jacobian).shape == (3, 6)
    assert np.allclose(np.diag(f.covariance), 1e-2)
    assert [e.factor_id for e in v.edges] == ["O0", "O1"]
    v.send_message(0)
    assert [x.has_unprocessed_messages() for x in fg.factor_nodes].count(True) == 1


def test_se3_pose_graph_converges():
    rng, n = np.random.default_rng(1), 6
    fg, truth = g.FactorGraph(), [np.eye(4)]
    for _ in range(1, n):
        truth.append(truth[-1] @ se3([1, 0, 0.1], [0.05, 0.1, 2 * np.pi / n]))
    for k in range(n):
        fg.add_variable(g.SE3Pose(f"P{k}", truth[k] @ se3(rng.normal(0, 0.05, 3), rng.normal(0, 0.05, 3)), np.eye(6) * 1e8))
    fg.add_factor(g.SE3PosePrior("PR", truth[0], np.eye(6) * 1e-2), ["P0"])
    for k in range(n):
        j = (k + 1) % n
        fg.add_factor(g.SE3PosePose(f"E{k}", np.linalg.inv(truth[k]) @ truth[j], np.eye(6) * 1e-2), [f"P{k}", f"P{j}"])
    for _ in range(200):
        tick(fg)
    assert max(np.abs(fg.variable_nodes[k].mu - truth[k]).max() for k in range(n)) < 1e-6
