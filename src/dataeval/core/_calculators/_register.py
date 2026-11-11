from dataeval.core._calculators._blurstats import BlurStatCalculator
from dataeval.core._calculators._dimensionstats import DimensionStatCalculator
from dataeval.core._calculators._hashstats import HashStatCalculator
from dataeval.core._calculators._pixelstats import PixelStatCalculator
from dataeval.core._calculators._visualstats import VisualStatCalculator

# Ensure all calculators have been registered
all(
    c is not None
    for c in [
        BlurStatCalculator,
        DimensionStatCalculator,
        HashStatCalculator,
        PixelStatCalculator,
        VisualStatCalculator,
    ]
)
