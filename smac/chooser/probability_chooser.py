from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import logging

import numpy as np

from smac.chooser import RandomConfigurationChooser

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class ProbabilityConfigurationChooser(RandomConfigurationChooser):
    """Interleave a random configuration according to a given probability.

    Parameters
    ----------
    prob : float
        Probility of a random configuration
    rng : np.random.RandomState
        Random state
    """

    def __init__(self, prob: float, seed: int = 0):
        super().__init__(seed)
        self.prob = prob

    def next_smbo_iteration(self) -> None:
        """Does nothing."""
        ...

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random."""
        if self.rng.rand() < self.prob:
            return True
        else:
            return False


class ProbabilityCoolDownConfigurationChooser(RandomConfigurationChooser):
    """Interleave a random configuration according to a given probability which is decreased over
    time.

    Parameters
    ----------
    prob : float
        Probility of a random configuration
    cool_down_fac : float
        Multiply the ``prob`` by ``cool_down_fac`` in each iteration
    rng : np.random.RandomState
        Random state
    """

    def __init__(self, prob: float, cool_down_fac: float, seed: int = 0):
        super().__init__(seed)
        self.prob = prob
        self.cool_down_fac = cool_down_fac

    def next_smbo_iteration(self) -> None:
        """Set the probability to the current value multiplied by the `cool_down_fac`."""
        self.prob *= self.cool_down_fac

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random."""
        if self.rng.rand() < self.prob:
            return True
        else:
            return False
