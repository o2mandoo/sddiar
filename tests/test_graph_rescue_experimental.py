import unittest
from time import perf_counter
from types import SimpleNamespace

from sddiar.contracts import Tracklet
from sddiar.graph_rescue_experimental import (
    GraphRescueError,
    GraphRescueConfig,
    GraphRescueResourceError,
    _build_adjacency,
    build_redacted_receipt,
    rescue_unknowns,
)


def t(tracklet_id, group, start, end=None):
    end = start + 1_000_000 if end is None else end
    return Tracklet(tracklet_id, f"r-{tracklet_id}", group, start, end, end - start, "ANCHOR" if tracklet_id.startswith(("a", "b")) else "MICRO")


class GraphRescueExperimentalTests(unittest.TestCase):
    def setUp(self):
        self.tracklets = (
            t("a0", "block-a0", 0),
            t("a1", "block-a1", 1_000_000),
            t("a2", "block-a2", 2_000_000),
            t("b0", "block-b0", 3_000_000),
            t("b1", "block-b1", 4_000_000),
            t("b2", "block-b2", 5_000_000),
            t("u", "unknown", 6_000_000),
        )
        self.labels = ("SPEAKER_00", "SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_01", "UNKNOWN")
        self.embeddings = {
            "a0": (1.0, 0.0, 0.0),
            "a1": (0.98, 0.2, 0.0),
            "a2": (0.99, -0.1, 0.0),
            "b0": (-1.0, 0.0, 0.0),
            "b1": (-0.98, -0.2, 0.0),
            "b2": (-0.99, 0.1, 0.0),
            "u": (0.99, 0.1, 0.0),
        }
        self.config = GraphRescueConfig(
            enabled=True,
            k_neighbors=4,
            max_edge_distance=0.7,
            propagation_steps=1,
            min_posterior=0.6,
            posterior_margin_min=0.15,
            leave_block_margin_min=0.05,
        )

    def test_h2_gate_and_default_off_do_not_compute_candidates(self):
        disabled = rescue_unknowns(self.tracklets, self.labels, self.embeddings)
        self.assertEqual(disabled.labels, self.labels)
        self.assertEqual(disabled.candidates, ())
        self.assertFalse(disabled.diagnostics["candidate_computation"])

        uncertain = rescue_unknowns(
            self.tracklets, self.labels, self.embeddings,
            decision_state="UNCERTAIN_1_OR_2", config=self.config,
        )
        self.assertEqual(uncertain.labels, self.labels)
        self.assertEqual(uncertain.diagnostics["skip_reason"], "H2_REQUIRED")

    def test_two_independent_blocks_and_leave_block_stability_rescue_unknown_only(self):
        original = list(self.labels)
        result = rescue_unknowns(
            self.tracklets, original, self.embeddings,
            decision_state="H2_CONFIRMED", config=self.config,
        )
        self.assertEqual(result.labels[-1], "SPEAKER_00")
        self.assertEqual(result.applied_count, 1)
        self.assertEqual(result.candidates[0].supporting_anchor_blocks, ("block-a0", "block-a1", "block-a2"))
        self.assertTrue(result.candidates[0].leave_block_stable)
        self.assertEqual(original, list(self.labels))
        # Anchor-clamping means no existing assignment can be overwritten.
        self.assertEqual(result.labels[:4], self.labels[:4])

    def test_ambiguous_unknown_abstains_on_posterior_margin(self):
        embeddings = dict(self.embeddings)
        embeddings["u"] = (0.0, 1.0, 0.0)
        result = rescue_unknowns(
            self.tracklets, self.labels, embeddings,
            decision_state="H2_CONFIRMED",
            config=self.config.__class__(
                enabled=True, k_neighbors=6, max_edge_distance=1.1,
                min_posterior=0.5, posterior_margin_min=0.2,
                leave_block_margin_min=0.05,
            ),
        )
        self.assertEqual(result.labels[-1], "UNKNOWN")
        self.assertEqual(result.candidates, ())

    def test_explicit_independent_blocks_and_resource_bound(self):
        result = rescue_unknowns(
            self.tracklets, self.labels, self.embeddings,
            decision_state="H2_CONFIRMED", config=self.config,
            anchor_block_ids={"a0": "A", "a1": "A", "a2": "A", "b0": "B0", "b1": "B1", "b2": "B2"},
        )
        self.assertEqual(result.labels[-1], "UNKNOWN")
        too_small = self.config.__class__(enabled=True, max_nodes=4)
        with self.assertRaises(GraphRescueResourceError):
            rescue_unknowns(self.tracklets, self.labels, self.embeddings,
                            decision_state="H2_CONFIRMED", config=too_small)

    def test_seed_default_is_anchor_only_and_support_requires_explicit_opt_in(self):
        from dataclasses import replace

        support_tracklets = self.tracklets[:2] + (replace(self.tracklets[2], kind="SUPPORT"),) + self.tracklets[3:]
        default = rescue_unknowns(
            support_tracklets, self.labels, self.embeddings,
            decision_state="H2_CONFIRMED", config=self.config,
        )
        self.assertEqual(default.labels[-1], "UNKNOWN")
        self.assertEqual(default.diagnostics["seed_eligibility"]["support_seed_count"], 0)
        self.assertEqual(default.diagnostics["seed_eligibility"]["excluded_assigned_count"], 1)

        opted_in = rescue_unknowns(
            support_tracklets, self.labels, self.embeddings,
            decision_state="H2_CONFIRMED", seed_tracklet_ids=("a2",), config=self.config,
        )
        self.assertEqual(opted_in.labels[-1], "SPEAKER_00")
        self.assertEqual(opted_in.diagnostics["seed_eligibility"]["support_seed_count"], 1)

    def test_input_and_decision_contracts_are_fail_closed(self):
        mismatched = dict(self.embeddings)
        mismatched["u"] = (1.0, 0.0, 0.0, 0.0)
        with self.assertRaises(GraphRescueError):
            rescue_unknowns(self.tracklets, self.labels, mismatched,
                            decision_state="H2_CONFIRMED", config=self.config)
        invalid_labels = self.labels[:-1] + ("INVENTED",)
        with self.assertRaises(GraphRescueError):
            rescue_unknowns(self.tracklets, invalid_labels, self.embeddings,
                            decision_state="H2_CONFIRMED", config=self.config)
        with self.assertRaises(GraphRescueError):
            rescue_unknowns(self.tracklets, self.labels, self.embeddings,
                            decision=SimpleNamespace(state="H1_CONFIRMED"),
                            h2_confirmed=True, config=self.config)

    def test_diagnostics_are_deep_immutable_and_receipt_is_redacted(self):
        result = rescue_unknowns(
            self.tracklets, self.labels, self.embeddings,
            decision_state="H2_CONFIRMED", config=self.config,
        )
        with self.assertRaises(TypeError):
            result.diagnostics["resource_bounds"]["max_nodes"] = 0
        receipt = build_redacted_receipt(result)
        self.assertEqual(receipt["schema"], "graph_rescue_redacted_receipt_v1")
        self.assertEqual(len(receipt["candidate_identity_digest"]), 64)
        self.assertNotIn("a0", str(receipt))
        self.assertNotIn("block-a0", str(receipt))

    def test_chunked_numpy_and_python_adjacency_have_same_output(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy is not installed")
        vectors = {
            "a": tuple(np.asarray((1.0, 0.0, 0.0))),
            "b": tuple(np.asarray((0.99, 0.1, 0.0))),
            "c": tuple(np.asarray((-1.0, 0.0, 0.0))),
            "u": tuple(np.asarray((0.995, 0.05, 0.0))),
        }
        cfg = GraphRescueConfig(enabled=True, k_neighbors=2, max_edge_distance=0.8)
        python_graph = _build_adjacency(tuple(vectors), vectors, cfg, use_numpy=False)
        numpy_graph = _build_adjacency(tuple(vectors), vectors, cfg, use_numpy=True)
        self.assertEqual(python_graph, numpy_graph)

    def test_1000_node_256d_wall_budget_when_numpy_is_available(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy is not installed")
        rng = np.random.default_rng(20260826)
        matrix = rng.normal(size=(1000, 256))
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        tracklets = tuple(
            SimpleNamespace(
                tracklet_id=f"n-{index:04d}", continuity_group_id=f"block-{index:04d}",
                kind="ANCHOR" if index < 999 else "MICRO",
            ) for index in range(1000)
        )
        labels = tuple("SPEAKER_00" if index % 2 == 0 else "SPEAKER_01" for index in range(999)) + ("UNKNOWN",)
        embeddings = {item.tracklet_id: matrix[index] for index, item in enumerate(tracklets)}
        cfg = GraphRescueConfig(
            enabled=True, k_neighbors=4, max_nodes=1200, max_edges=6000,
            max_dimension=256, max_distance_evaluations=1_500_000,
            max_edge_distance=2.0,
        )
        started = perf_counter()
        result = rescue_unknowns(tracklets, labels, embeddings,
                                 decision_state="H2_CONFIRMED", config=cfg)
        elapsed = perf_counter() - started
        self.assertEqual(result.diagnostics["node_count"], 1000)
        self.assertLess(elapsed, 30.0, f"1000-node graph took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
