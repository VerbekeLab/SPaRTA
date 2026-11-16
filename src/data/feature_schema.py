# Canonical SPaRTA feature-column schema, per NEIGHBOURHOOD ORDER.
#
# Pure python on purpose (no torch / networkx / multiprocessing): the window store, the sequence
# builder and the extraction script all need the column list, and importing it from
# measure_calculation.py used to drag in the whole extraction stack (and its os.chdir side effect).
#
# ORDER 2 (the paper's SPaRTA features): each node's 2-hop ego network is split into three groups
# G0 (ego + off-path 2-hop nodes), G1 (1-hop neighbours), G2 (2-hop nodes) and summarised as a
# 3x3 block picture over the cells 00..22 (row = source group, column = target group) x 10 channels
# (measure, size, 4 amount summaries, 4 count summaries) = 90 columns.
#
# ORDER 1 (the first-order baseline): the same code run on a radius-1 ego graph. There G0 = {ego},
# G2 = {} and only the cells that hold the node's OWN edges carry information:
#   01  ego -> neighbours   (out-edges)
#   10  neighbours -> ego   (in-edges)
# The neighbour<->neighbour block 11 is deliberately dropped (it is triangle information, not the
# node's own transactions), and every other cell is constant zero. That leaves 2 cells x 10
# channels = 20 columns, named exactly like their order-2 counterparts.

DEFAULT_ORDER = 2
ORDERS = (1, 2)

# Row-major position of each cell in the flat 3x3 picture the extraction code emits.
CELLS = ("00", "01", "02", "10", "11", "12", "20", "21", "22")

# Cells that carry information for each order (a subset of CELLS, in CELLS order).
ORDER_CELLS = {2: CELLS, 1: ("01", "10")}

WEIGHTS = ("amount_trans", "num_trans")
AGGREGATIONS = ("sum", "mean", "max", "std")


def check_order(order):
    """Return `order` as an int, or raise ValueError if it is not a supported order."""
    order = int(order)
    if order not in ORDERS:
        raise ValueError(f"neighbourhood order must be one of {ORDERS}, got {order!r}")
    return order


def order_cells(order=DEFAULT_ORDER):
    """Informative cells for `order`, e.g. ('01', '10') for order 1."""
    return ORDER_CELLS[check_order(order)]


def cell_indices(order=DEFAULT_ORDER):
    """Positions of the informative cells inside the flat row-major 3x3 picture:
    (0..8) for order 2, (1, 3) for order 1."""
    return tuple(CELLS.index(c) for c in order_cells(order))


def feature_columns(order=DEFAULT_ORDER):
    """Feature column names for `order`, in the order measure_calculation writes them:
    measure_<cell>..., size_<cell>..., then weight x aggregation x cell summaries.
    feature_columns(2) is the legacy 90-column `keys_to_include`, unchanged."""
    cells = order_cells(order)
    cols = [f"measure_{c}" for c in cells] + [f"size_{c}" for c in cells]
    for weight in WEIGHTS:
        for agg in AGGREGATIONS:
            for c in cells:
                cols.append(f"transaction_{weight}_summary_{agg}_{c}")
    return cols


def n_features(order=DEFAULT_ORDER):
    """Number of feature columns for `order` (90 for order 2, 20 for order 1)."""
    return len(feature_columns(order))
