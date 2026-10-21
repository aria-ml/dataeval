__all__ = []

from dataeval.core._calculators._descriptivestats import DescriptiveStatCalculator
from dataeval.core._calculators._egostats import EgoStatCalculator
from dataeval.core._calculators._redundancystats import RedundancyStatCalculator

# Ensure all video calculators have been registered
all(c is not None for c in [DescriptiveStatCalculator, EgoStatCalculator, RedundancyStatCalculator])
