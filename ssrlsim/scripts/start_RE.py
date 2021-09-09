import os
import matplotlib

# get_ipython().run_line_magic("matplotlib", "widget")  # i.e. %matplotlib widget
import matplotlib.pyplot as plt

from ophyd import Device, Component, EpicsSignal
from ophyd.signal import EpicsSignalBase
from ophyd.areadetector.filestore_mixins import resource_factory
import uuid
import os
from pathlib import Path
import numpy as np
from IPython import get_ipython

# Set up a RunEngine and use metadata backed by a sqlite file.
from bluesky import RunEngine
from bluesky.utils import PersistentDict

RE = RunEngine({})
# RE.md = PersistentDict(str(Path("~/.bluesky_history").expanduser()))

# Set up SupplementalData.
from bluesky import SupplementalData

sd = SupplementalData()
RE.preprocessors.append(sd)

# Set up a Broker.
from databroker import Broker

db = Broker.named("temp") #mongo-intake")
print(f'Using databroker: {db.name}')

# and subscribe it to the RunEngine
RE.subscribe(db.insert)

# Add a progress bar.
from bluesky.utils import ProgressBarManager

pbar_manager = ProgressBarManager()
RE.waiting_hook = pbar_manager

# # Register bluesky IPython magics.
# from bluesky.magics import BlueskyMagics

# get_ipython().register_magics(BlueskyMagics)

# Set up the BestEffortCallback.
from bluesky.callbacks.best_effort import BestEffortCallback

bec = BestEffortCallback()
RE.subscribe(bec)
peaks = bec.peaks

# Make plots update live while scans run.
from bluesky.utils import install_nb_kicker

install_nb_kicker()

# convenience imports
# some of the * imports are for 'back-compatibility' of a sort -- we have
# taught BL staff to expect LiveTable and LivePlot etc. to be in their
# namespace
import numpy as np

from bluesky.callbacks.mpl_plotting import LivePlot, LiveGrid

import bluesky.plans as bp

import bluesky.plan_stubs as bps

import bluesky.preprocessors as bpp
