"""Unit tests for the BatSLAM 2.0 building blocks (no dataset needed)."""
import sys
from pathlib import Path

import gtsam
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batslam.backend.pose_graph import PoseGraph  # noqa: E402
from batslam.config import BatSLAMConfig, GraphConfig, RecognitionConfig, TemplateConfig  # noqa: E402
from batslam.recognition.sequence import SequenceVerifier  # noqa: E402
from batslam.views.templates import Candidate, TemplateStore  # noqa: E402


def test_config_roundtrip(tmp_path):
    cfg = BatSLAMConfig()
    cfg.recognition.min_pairs = 11
    cfg.save(tmp_path / "c.json")
    back = BatSLAMConfig.load(tmp_path / "c.json")
    assert back.recognition.min_pairs == 11
    assert back.graph.loop_sigma == cfg.graph.loop_sigma


def test_template_shift_sign():
    """A query whose echoes arrive later than the template's -> positive shift."""
    rng = np.random.default_rng(0)
    base = np.zeros((2, 4, 100), np.float32)
    base[..., 30:33] = 1.0
    base[..., 60:62] = 0.5
    base += 0.01 * rng.random(base.shape, dtype=np.float32)
    store = TemplateStore(TemplateConfig(max_shift_bins=6, exclude_recent_m=0.0), base.shape, 0.03)
    store.add(0, base, 0.0)
    later = np.roll(base, 4, axis=-1)
    sim, shift = store.similarities(later)
    assert sim[0] > 0.95
    assert shift[0] == 4


def _straight_line_verifier(cfg, anchors_offset=0.0):
    # query node k at x = 0.15 k (odometry); anchor node a at x = 0.15 a + offset
    odom = lambda k: gtsam.Pose2(0.15 * k, 0, 0)            # noqa: E731
    mp = lambda a: gtsam.Pose2(0.15 * a + anchors_offset, 0, 0)  # noqa: E731
    return SequenceVerifier(cfg, odom, mp, use_shift_offset=False)


def test_sequence_commits_consistent_run():
    cfg = RecognitionConfig(min_pairs=5, min_templates=3, commit_evidence=5.0, ambiguity_margin=2.0)
    v = _straight_line_verifier(cfg)
    released = []
    for k in range(100, 115):
        a = k - 100                       # revisiting anchors 0, 1, 2, ...
        r = v.step(k, [Candidate(k, a, a, 0.9, 0.0)])
        released += r.release
    assert released, "a long consistent run of good matches must commit"


def test_sequence_ambiguity_blocks_commit():
    """Two equally good, mutually inconsistent explanations -> no commit."""
    cfg = RecognitionConfig(min_pairs=5, min_templates=3, commit_evidence=5.0, ambiguity_margin=2.0)
    v = _straight_line_verifier(cfg)
    released = []
    for k in range(100, 115):
        a = k - 100
        cands = [Candidate(k, a, a, 0.9, 0.0),                 # place A
                 Candidate(k, 1000 + a, 50 + a, 0.9, 0.0)]      # alias B, 7.5 m away
        released += v.step(k, cands).release
    assert not released


def test_sequence_rejects_inconsistent_jumps():
    """Good similarities that jump around geometrically never commit."""
    cfg = RecognitionConfig(min_pairs=5, min_templates=3, commit_evidence=5.0, ambiguity_margin=2.0)
    v = _straight_line_verifier(cfg)
    rng = np.random.default_rng(1)
    released = []
    for k in range(100, 130):
        a = int(rng.integers(0, 90))
        released += v.step(k, [Candidate(k, a, a, 0.95, 0.0)]).release
    assert not released


def _square_then_retrace(n_links):
    """Drive a biased square, then retrace its first side; link the retrace
    to the first side with ``n_links`` weak view links."""
    g = PoseGraph(GraphConfig())
    g.add_first((0, 0, 0))
    for side in range(4):
        for _ in range(10):
            g.add_odometry((0.2, 0, 0))
            g.update()
        g.add_odometry((0, 0, np.pi / 2 + 0.15))       # biased turn
        g.update()
    start = g.n - 1                                     # truly back at the origin
    for _ in range(10):
        g.add_odometry((0.2, 0, 0))
        g.update()
    for k in range(n_links):
        g.add_view_link(start + k, k, (0, 0, 0))
    g.update(extra_iterations=3)
    end = g.n - 1
    return np.hypot(g.pose(end).x() - 2.0, g.pose(end).y())   # truth: (2, 0)


def test_weak_links_need_a_sequence():
    """One weak link barely moves the graph; a sequence of them closes the loop."""
    none, one, seq = _square_then_retrace(0), _square_then_retrace(1), _square_then_retrace(11)
    assert one < none                     # it does pull...
    assert one > 0.5 * none               # ...but a single view is not decisive
    assert seq < 0.25 * none              # a consistent sequence is


def test_short_link_group_is_withdrawn_for_good():
    """Length rule: a group whose hypothesis ends with too few links is removed
    from the graph permanently (brief corridor aliases); a long one stays."""
    g = PoseGraph(GraphConfig(link_management=True, min_group_links=16, short_group_grace=10, gnc_every=0))
    g.add_first((0, 0, 0))
    for _ in range(120):
        g.add_odometry((0.15, 0, 0))
        g.update()
    for k in range(5):                      # short group: 5 links, then silence
        g.add_view_link(60 + k, 10 + k, (0, 0, 0), hypothesis_id=1)
    for k in range(20):                     # long group: 20 links
        g.add_view_link(80 + k, 30 + k, (0, 0, 0), hypothesis_id=2)
    g.update()
    removed = g.manage(119, new_links=False)
    assert 1 in removed and 2 not in removed
    assert g.groups[1].permanent_reject and g.groups[1].status == "rejected"
    assert all(l.hypothesis_id == 2 for l in g.loops)
