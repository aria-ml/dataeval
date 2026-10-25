__all__ = []

import numpy as np

# Values marked provisional may change after review, hence one place to edit.

PANEL_SIZE = 5  # panelists per disjoint panel; provisional
N_CELLS_SCHEDULE = ((20_000, 10), (50_000, 30), (None, 300))  # planned audit units below bound -> cells; provisional
FIT_FRACTION = 0.5  # fraction of groups used to fit partitions and witnesses; provisional
WITNESS_POS_PREVALENCE = 0.15  # pos thresholds below this fit-split prevalence; provisional guess in (0.1, 0.2)
WITNESS_QUANTILES = np.linspace(0.05, 0.95, 19)  # candidate witness threshold quantiles
ALLOCATION = "equal_per_claim"  # equal split per (target, evidence source); provisional
# {0} U {k/(m-k)}: the cost grid of the AP and recall ceilings, where the consensus changes slope; provisional
COST_GRID = "kinks"
AP_INTERVALS = 1000  # intervals in the AP envelope integration grid
COVERAGE_SEEDS = 346  # seeds in the coverage simulation
COVERAGE_CRITICAL = 318  # a coverage claim fails at this many covered seeds or fewer: passing needs >= 319
SPEC_VERSION = 1  # version of the canonical target specification
EVIDENCE_SCHEMA = "feasibility-evidence/1"  # schema tag carried by evidence items


def n_cells_for(planned_audit_units: int) -> int:
    """Return the cell count for a planned audit size: the first bound it is below (None = no bound)."""
    return next(n for bound, n in N_CELLS_SCHEDULE if bound is None or planned_audit_units < bound)
