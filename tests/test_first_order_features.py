# Checks for the first-order (SPARTA_ORDER=1) baseline. Plain asserts: runs under pytest or directly,
# from anywhere (it locates the repo root itself) with the SPaRTA_env interpreter:
#   /Users/brunodeprez/anaconda3/envs/SPaRTA_env/bin/python tests/test_first_order_features.py
#
# What is verified
#   1. schema     feature_columns(2) is the legacy 90-column list, unchanged; order 1 is 20 columns.
#   2. extraction the 20 first-order columns that measure_calculation.process_node emits equal an
#                 INDEPENDENT edge-list computation of the node's own in-/out-edge statistics (random
#                 digraphs with isolated nodes, reciprocal edges, echo-style weights); the order-2
#                 columns still line up with their names.
#   3. namespace  run_tag / static_stem add `_o1` only for order 1 and are byte-identical for order 2.
#   4. reporting  collect_results / extra_metrics parse the `_o1` infix, keep the two orders in
#                 disjoint rows, and never leave a file unparsed.
#   5. windows    a first-order window store round-trips with 20 columns and is refused under order 2.
#   6. static     the first-order static split is identical to the order-2 split even when the two
#                 feature files list the rows in a different order.
#   7. comparison scripts/compare_orders.py pairs the two orders, re-aligns permuted prediction rows
#                 for the bootstrap, and refuses pairs whose test rows differ.
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import xgboost  # noqa: F401  (import BEFORE torch: OpenMP clash segfaults otherwise)

import contextlib
import itertools
import tempfile

import networkx as nx
import numpy as np
import pandas as pd


@contextlib.contextmanager
def order(n):
    """Run a block with SPARTA_ORDER=n (None unsets it)."""
    old = os.environ.get("SPARTA_ORDER")
    if n is None:
        os.environ.pop("SPARTA_ORDER", None)
    else:
        os.environ["SPARTA_ORDER"] = str(n)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SPARTA_ORDER", None)
        else:
            os.environ["SPARTA_ORDER"] = old


def _legacy_keys_to_include():
    """The 90-column list exactly as measure_calculation.py built it before the schema module."""
    keys = ['measure_00', 'measure_01', 'measure_02', 'measure_10', 'measure_11', 'measure_12',
            'measure_20', 'measure_21', 'measure_22', 'size_00', 'size_01', 'size_02',
            'size_10', 'size_11', 'size_12', 'size_20', 'size_21', 'size_22']
    for weight in ['amount_trans', 'num_trans']:
        for agg in ['sum', 'mean', 'max', 'std']:
            for suff in ['00', '01', '02', '10', '11', '12', '20', '21', '22']:
                keys.append(f'transaction_{weight}_summary_{agg}_{suff}')
    return keys


def _random_graph(seed, n=40, p=0.07, echo=False):
    """DiGraph shaped like load_network's output: weight / amount_trans / num_trans per edge."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n + 3))                  # the last three never get an edge: isolated nodes
    for u, v in itertools.permutations(range(n), 2):    # no self-loops (the loaders drop them)
        if rng.random() < p:
            G.add_edge(u, v,
                       weight=float(rng.uniform(0.05, 1.0)) if echo else 1,
                       amount_trans=float(rng.lognormal(3, 1)),
                       num_trans=int(rng.integers(1, 9)))
    return G


def _reference_own_edge_columns(G, v):
    """Independent first-order columns of node v, straight from its edges (no networkx ego graphs,
    no frozen block functions): cell 01 = v's out-edges, cell 10 = v's in-edges, over its n1
    distinct neighbours (zero where only the other direction exists)."""
    nbrs = sorted(set(G.successors(v)) | set(G.predecessors(v)))
    n1 = len(nbrs)
    cols = {}

    def vec(attr, direction):
        if direction == "out":
            return np.array([G[v][x][attr] if G.has_edge(v, x) else 0.0 for x in nbrs], dtype=float)
        return np.array([G[x][v][attr] if G.has_edge(x, v) else 0.0 for x in nbrs], dtype=float)

    for cell, direction in (("01", "out"), ("10", "in")):
        cols[f"measure_{cell}"] = vec("weight", direction).sum() / n1 if n1 else (1.0 if cell == "01" else 0.0)
        cols[f"size_{cell}"] = float(n1)
        for weight in ("amount_trans", "num_trans"):
            a = vec(weight, direction)
            for agg, fn in (("sum", np.sum), ("mean", np.mean), ("max", np.max), ("std", np.std)):
                cols[f"transaction_{weight}_summary_{agg}_{cell}"] = float(fn(a)) if n1 else 0.0
    return cols


def _row(G, v, order_):
    """process_node's output for node v, in-process (no Pool), as {column: value}."""
    from src.data.feature_schema import feature_columns
    from src.methods import measure_calculation as mc
    mc.init_worker(G, G.reverse(copy=True), G.to_undirected(), order_)
    out = mc.process_node(v)
    assert out[0] == v
    cols = feature_columns(order_)
    assert len(out) == 1 + len(cols), (len(out), len(cols))
    return dict(zip(cols, out[1:]))


# ------------------------------------------------------------------------------------------ 1. schema
def test_schema():
    from src.data.feature_schema import feature_columns, cell_indices, n_features
    assert feature_columns(2) == _legacy_keys_to_include()
    assert n_features(2) == 90 and cell_indices(2) == tuple(range(9))
    c1 = feature_columns(1)
    assert n_features(1) == 20 and len(set(c1)) == 20
    assert cell_indices(1) == (1, 3)                         # cells 01 and 10 of the row-major 3x3
    assert set(c1) <= set(feature_columns(2))                # same names as their order-2 counterparts
    assert all(c.endswith(("_01", "_10")) for c in c1)       # nothing of cells 00/02/11/12/2x
    assert c1[:4] == ["measure_01", "measure_10", "size_01", "size_10"]
    from src.methods.measure_calculation import keys_to_include
    assert keys_to_include == _legacy_keys_to_include()      # legacy import path unchanged


# -------------------------------------------------------------------------------------- 2. extraction
def test_first_order_matches_independent_computation():
    for seed, echo in ((1, False), (2, True), (3, False)):
        G = _random_graph(seed, echo=echo)
        assert any(G.degree(v) == 0 for v in G)              # isolated nodes are exercised
        assert any(G.has_edge(v, u) and G.has_edge(u, v) for u, v in G.edges)   # reciprocal edges too
        for v in G.nodes:
            got = _row(G, v, 1)
            ref = _reference_own_edge_columns(G, v)
            assert got.keys() == ref.keys()
            for name in ref:
                assert np.isclose(got[name], ref[name], rtol=1e-9, atol=1e-12), (seed, v, name, got[name], ref[name])


def test_order2_columns_line_up_with_names():
    """Each named order-2 column equals the frozen measure/summary output at that cell's position."""
    from src.data.feature_schema import CELLS
    from src.methods.neighbourhood_picture import node_measures
    G = _random_graph(7)
    G_und, G_rev = G.to_undirected(), G.reverse(copy=True)
    for v in list(G.nodes)[:15]:
        direct = node_measures(v, G, G_und, G_rev, output_tensor=False)       # default radius=2
        got = _row(G, v, 2)
        for name, val in got.items():
            cell = name[-2:]
            pos = CELLS.index(cell)
            if name.startswith("measure_"):
                ref = direct["measure"][pos]
            elif name.startswith("size_"):
                ref = direct["size"][pos]
            else:
                ref = direct[name.rsplit("_", 1)[0]][pos]
            assert np.isclose(val, ref, rtol=0, atol=0, equal_nan=True), (v, name)


# --------------------------------------------------------------------------------------- 3. namespace
def test_namespacing_is_a_noop_for_order_2():
    from src.utils.setup import run_tag, static_stem, resolve_order, order_suffix
    echo = dict(time_type="days", time_step=1, echo=True, days_echo=3, time_width=1)
    plain = dict(time_type="hours", time_step=6, echo=False, days_echo=3, time_width=24)
    with order(None):
        assert resolve_order() == 2 and order_suffix() == ""
        assert run_tag(echo) == "d1_echo3" and run_tag(plain) == "h6_w24"
        assert static_stem("HI-Small") == "HI-Small"
    with order(2):
        assert run_tag(echo) == "d1_echo3"
    with order(1):
        assert run_tag(echo) == "d1_echo3_o1" and run_tag(plain) == "h6_w24_o1"
        assert static_stem("HI-Small") == "HI-Small_o1"
    assert run_tag(echo, order=1) == "d1_echo3_o1"            # explicit argument beats the env
    with order(3):
        try:
            run_tag(echo)
        except ValueError:
            pass
        else:
            raise AssertionError("SPARTA_ORDER=3 must be rejected")


def test_cnn_style_entry_points_refuse_order_1():
    from src.utils.setup import require_default_order
    with order(None):
        require_default_order("x", "why")
    with order(1):
        try:
            require_default_order("x", "why")
        except SystemExit:
            pass
        else:
            raise AssertionError("must exit under SPARTA_ORDER=1")


# --------------------------------------------------------------------------------------- 4. reporting
_DUMP = "Model: {m}\nAUC PR: {pr}\nAUC ROC: 0.9\nPrecision: [0.01 0.5]\n"


def test_reporting_keeps_the_orders_apart():
    from scripts import collect_results as cr
    from scripts.extra_metrics import describe
    names = {
        "HI-Small_d1_echo3_K7_nowcast_timeseries_LSTM.txt": ("LSTM", 0.30),
        "HI-Small_d1_echo3_o1_K7_nowcast_timeseries_LSTM.txt": ("LSTM", 0.20),
        "HI-Small_d1_echo3_K7_forecast_local.txt": ("XGBoost", 0.31),
        "HI-Small_d1_echo3_o1_K7_forecast_local.txt": ("XGBoost", 0.21),
        "HI-Small_h6_w24_o1_K13_nowcast_baselines_XGBoost.txt": ("XGBoost", 0.22),
        "HI-Small_features_XGBoost.txt": None,
        "HI-Small_o1_features_XGBoost.txt": None,
        "HI-Small_HPT_features_XGBoost.txt": None,
        "HI-Small_o1_HPT_features_XGBoost.txt": None,
    }
    old_dir = cr.EXP_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cr.EXP_DIR = tmp
            for fname, spec in names.items():
                with open(os.path.join(tmp, fname), "w") as fh:
                    if spec is None:
                        fh.write("AUC PR: 0.4\nAUC ROC: 0.8\nPrecision: [0.01 0.5]\n" if "HPT" not in fname
                                 else "best params only\n")
                    else:
                        fh.write(_DUMP.format(m=spec[0], pr=spec[1]))
            df, unparsed = cr.collect()
    finally:
        cr.EXP_DIR = old_dir
    assert unparsed == [], unparsed
    assert set(df.neighbourhood_order) == {1, 2}
    # the order-1 tag keeps its infix, so a consumer that ignores the new column cannot mix orders
    assert set(df[df.neighbourhood_order == 1].run_tag) == {"d1_echo3_o1", "h6_w24_o1", "static_o1"}
    assert set(df[df.neighbourhood_order == 2].run_tag) == {"d1_echo3", "static"}
    assert set(df.dataset) == {"HI-Small"}                    # static '_o1' peeled off the dataset
    assert not df.duplicated(["dataset", "run_tag", "K", "task", "model", "split"]).any()
    lstm = df[(df.model == "LSTM") & (df.task == "nowcast")].set_index("neighbourhood_order").AUC_PR
    assert lstm.loc[2] == 0.30 and lstm.loc[1] == 0.20       # o1 never overwrote the order-2 row
    stat = df[df.task.isna()]
    assert sorted(zip(stat.neighbourhood_order, stat.run_tag)) == [(1, "static_o1"), (2, "static")]

    assert describe("HI-Small_d1_echo3_o1_K7_nowcast_timeseries.parquet") == \
        ("HI-Small", "d1_echo3_o1", 7, "nowcast", "", 1)
    assert describe("HI-Small_d1_echo3_K7_forecast_local.parquet") == \
        ("HI-Small", "d1_echo3", 7, "forecast", "Local ", 2)
    assert describe("HI-Small_o1_features.parquet") == ("HI-Small", "static_o1", None, None, "", 1)
    assert describe("HI-Small_features.parquet") == ("HI-Small", "static", None, None, "", 2)


def test_order1_names_cannot_match_order2_globs():
    """The experiment scripts glob `results/{experiments,tuning}/<stem>_<kind>*` with the stem of
    their own order; the `_o1` infix sits before `_K`, so neither order's glob sees the other's."""
    import fnmatch
    stem2, stem1 = "HI-Small_d1_echo3_K7_nowcast", "HI-Small_d1_echo3_o1_K7_nowcast"
    for kind in ("timeseries", "baselines", "local"):
        assert not fnmatch.fnmatch(f"{stem1}_{kind}_LSTM.txt", f"{stem2}_{kind}*.txt")
        assert not fnmatch.fnmatch(f"{stem2}_{kind}_LSTM.txt", f"{stem1}_{kind}*.txt")


# ----------------------------------------------------------------------------------------- 5. windows
def _write_snapshots(features_dir, cols, T=3, n=12, seed=0):
    from src.utils.feature_io import save_table
    rng = np.random.default_rng(seed)
    os.makedirs(features_dir, exist_ok=True)
    for i in range(T):
        nodes = [f"n{k}" for k in range(n - i)]               # a node drops out each snapshot
        feat = pd.DataFrame(rng.normal(size=(len(nodes), len(cols))), columns=cols)
        feat.insert(0, "node", nodes)
        save_table(feat, os.path.join(features_dir, f"TST_dynamic_{i}_features"))
        lab = pd.DataFrame({"Account": nodes, "is_laundering": (rng.random(len(nodes)) < 0.2).astype(int)})
        save_table(lab, os.path.join(features_dir, f"TST_dynamic_{i}_labels"))


def test_window_store_roundtrip_and_order_guard():
    from src.data import window_store
    from src.data.feature_schema import feature_columns
    from src.data.sequence_data import _load_snapshot
    with tempfile.TemporaryDirectory() as tmp:
        feats, store = os.path.join(tmp, "f"), os.path.join(tmp, "s")
        _write_snapshots(feats, feature_columns(1))
        with order(1):
            meta = window_store.build_store(store, "TST", "", feats, T=3, verbose=False)
            assert meta["n_features"] == 20 and meta["neighbourhood_order"] == 1
            assert meta["feature_columns"] == feature_columns(1)
            st = window_store.load_window_store(store, "TST", "")
            values, idx, y, anchors, nodes = window_store.windows_from_store(st, K=2, task="nowcast")
            assert values.shape == (sum(12 - i for i in range(3)), 20)
            assert idx.shape[1] == 2 and len(y) == len(anchors) == len(nodes) == len(idx)
        with order(2):                                         # a first-order store is refused ...
            try:
                window_store.load_window_store(store, "TST", "")
            except ValueError as e:
                assert "order-1" in str(e)
            else:
                raise AssertionError("order-2 job must refuse an order-1 store")
            try:                                               # ... and so are order-1 feature files
                _load_snapshot(0, "TST", "", feats)
            except ValueError as e:
                assert "SPARTA_ORDER" in str(e)
            else:
                raise AssertionError("order-2 job must refuse 20-column feature files")


# ------------------------------------------------------------------------------------------ 6. static
def test_static_split_is_shared_across_orders():
    from scripts.experiment_features import data_prep_features
    from src.utils.feature_io import save_table
    rng = np.random.default_rng(5)
    n = 300
    nodes = np.array([f"acc{k}" for k in range(n)])
    labels = pd.DataFrame({"Account": nodes, "is_laundering": (rng.random(n) < 0.1).astype(int)})
    with tempfile.TemporaryDirectory() as tmp:
        f2 = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
        f2.insert(0, "node", nodes)
        perm = rng.permutation(n)                              # imap_unordered gives another row order
        f1 = pd.DataFrame(rng.normal(size=(n, 2)), columns=list("ab"))
        f1.insert(0, "node", nodes[perm])
        save_table(f2, os.path.join(tmp, "TST_static_features"))
        save_table(labels, os.path.join(tmp, "TST_static_labels"))
        save_table(f1, os.path.join(tmp, "TST_o1_static_features"))
        save_table(labels, os.path.join(tmp, "TST_o1_static_labels"))

        *_, node_test_2 = data_prep_features(tmp, "TST")
        *_, node_test_naive = data_prep_features(tmp, "TST_o1")                 # no alignment
        *_, node_test_1 = data_prep_features(tmp, "TST_o1", ref_type="TST")     # aligned to order 2
    assert set(node_test_naive) != set(node_test_2)            # the hazard is real ...
    assert list(node_test_1) == list(node_test_2)              # ... and the alignment removes it


# ------------------------------------------------------------------------------------ 7. comparison
def test_compare_orders_pairs_cells_and_guards_the_bootstrap():
    from scripts import compare_orders as co
    from src.methods import evaluation

    cols = ["dataset", "run_tag", "K", "task", "model", "split", "AUC_PR", "AUC_ROC", "PREVALENCE",
            "neighbourhood_order"]
    summary = pd.DataFrame([
        ("A", "d1_echo3", 7, "nowcast", "LSTM", "test", 0.40, 0.90, 0.01, 2),
        ("A", "d1_echo3_o1", 7, "nowcast", "LSTM", "test", 0.30, 0.85, 0.01, 1),
        ("A", "static", None, None, "XGBoost", "test", 0.50, 0.95, 0.01, 2),
        ("A", "static_o1", None, None, "XGBoost", "test", 0.55, 0.94, 0.01, 1),
        ("A", "d1_echo3", 7, "nowcast", "CNN-LSTM", "test", 0.45, 0.91, 0.01, 2),   # no order-1 twin by design
    ], columns=cols)
    paired, unpaired = co.paired_table(summary)
    assert len(paired) == 2 and set(paired.model) == {"LSTM", "XGBoost"}
    d = paired.set_index("model").dAUC_PR
    assert np.isclose(d["LSTM"], 0.10) and np.isclose(d["XGBoost"], -0.05)      # o2 - o1
    assert list(unpaired.model) == ["CNN-LSTM"] and list(unpaired.only_in) == ["order 2"]

    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.1).astype(int)
    nodes = np.array([f"x{k}" for k in range(n)])
    anchors = np.full(n, 9)
    s2, s1 = y + rng.normal(0, 0.6, n), y + rng.normal(0, 1.5, n)             # order 2 is the better scorer
    old_pred_dir = evaluation.PRED_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation.PRED_DIR = tmp
            evaluation.save_predictions(os.path.join(tmp, "A_d1_echo3_K7_nowcast_timeseries.parquet"),
                                        {"LSTM": s2}, y, anchors=anchors, nodes=nodes)
            perm = rng.permutation(n)                                         # order 1 scored the rows in another order
            evaluation.save_predictions(os.path.join(tmp, "A_d1_echo3_o1_K7_nowcast_timeseries.parquet"),
                                        {"LSTM": s1[perm]}, y[perm], anchors=anchors[perm], nodes=nodes[perm])
            evaluation.save_predictions(os.path.join(tmp, "A_d1_echo3_K7_forecast_timeseries.parquet"),
                                        {"LSTM": s2}, y, anchors=anchors, nodes=nodes)
            evaluation.save_predictions(os.path.join(tmp, "A_d1_echo3_o1_K7_forecast_timeseries.parquet"),
                                        {"LSTM": s1[:-1]}, y[:-1], anchors=anchors[:-1], nodes=nodes[:-1])
            boot, skipped = co.bootstrap_cells(n_boot=30, seed=1)
    finally:
        evaluation.PRED_DIR = old_pred_dir
    assert len(boot) == 1 and boot.iloc[0]["task"] == "nowcast"               # permuted rows are re-aligned and paired
    assert boot.iloc[0]["dAP"] > 0 and boot.iloc[0]["n_test"] == n
    assert any("different test rows" in why for _, why in skipped)           # truncated rows are refused


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")
