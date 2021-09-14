'''
OPHYD classes for simulated beamline SSRL1-5, in hitp mode of operation
Should mimic ophyd devices as closely as possible for operation

Many ideas stolen from various bluesky tutorials

Device List:
Stage (px, py, pz, th, vx, vy)
Laser Range Finder (linked to stage v?)
xspress3
an area detector (MarCCD)
I0, I1, shutter?

Plans:
basic bluesky plans
grid scan

Framework:
subscribe stage to baseline

To-Do: 
- make I1, I0, reactive to shutter

'''
import os
import numpy as np
from pathlib import Path

from ophyd import Signal, Device, Component as Cpt, MotorBundle
from ophyd.sim import SynAxis 

import bluesky.plan_stubs as bps

from . import ArraySynSignal, gen_wafer_locs, SynTiffFilestore, SynHDF5Filestore
from .images import make_random_peaks, generate_image

# initialize RunEngine, temp databroker
from ssrlsim.scripts.start_RE import *

# general beamline components (motors, shutter, beam)

class SynLaserRangeFinder(Signal):
    """SynLaserRangeFinder simulates height sensor.  
    
    Represents stage as perfectly flat with 4 randomly initialized heights at 
    cardinal directions.  
    """
    def __init__(self, stage_x, stage_y, plate_x, plate_y, *args, **kwargs):
        self.stage_x = stage_x # should be ophyd SynAxis's
        self.stage_y = stage_y
        self.plate_x = plate_x
        self.plate_y = plate_y

        # Stage limits
        self.x_min = -30
        self.x_max = 30
        self.y_min = -60
        self.y_max = 60
        
        # initial heights, could be different from motor readouts
        # located at x_max and y_max, with x_min/y_min at 0
        self.real_plate_x = int(np.random.uniform(200, 500)) * np.random.choice([-1, 1])
        self.real_plate_y = int(np.random.uniform(20, 500)) * np.random.choice([-1, 1])

        # Sample footprint.  wafer thicknes = 0.5mm = 1.07V
        # spike to 10 after off stage

        super(SynLaserRangeFinder, self).__init__(*args, **kwargs)

    def get(self):
        '''
        returns value based on position of motors
        TODO: Fix issues with units here.  level stage plan is hardcoded to 
               work with LRF at 1-5
        TODO: Issues with pico motor operation.  Each tweak results in some small amount.
            --> See calibration curves
            --> Typically values around 4
            --> std of 0.00815 (1/s for 1min) 
            --> 0.0727 V / 50 steps = 0.001454 V/step
            --> 4.28 / 2 mm = 2.14 V/mm
        Assumes pico motor is at x_min, y_min points
        '''
        st2V = 0.001454

        # 'ims' motors (mm)
        x_pos = self.stage_x.read()[self.stage_x.name]['value']
        y_pos = self.stage_y.read()[self.stage_y.name]['value']

        # 'pico' motors (steps)
        x_vert = self.plate_x.read()[self.plate_x.name]['value']
        y_vert = self.plate_y.read()[self.plate_y.name]['value']
        
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min

        mx = (x_vert - self.real_plate_x) / x_range # steps / mm 
        my = (y_vert - self.real_plate_y) / y_range # steps / mm 
        
        # y = mx + b (steps)
        # step = (steps / mm * mm) * steps
        x_disp = mx * x_pos - (x_vert - self.real_plate_x) / 2
        y_disp = my * y_pos - (y_vert - self.real_plate_y) / 2
        
        # sample shape.  For simplicity assume:
        # - wafer is in center of stage.... 4" = 101mm
        # - stage is square
        if (x_pos**2 + y_pos**2) <= 5**2:
            # on wafer
            offset = -1
        else:
            offset = 0

        # average displacements 
        self._readback = 4 + offset - ((x_disp + y_disp) * st2V / 2)
        
        if (np.abs(x_pos) > self.x_max) or (np.abs(y_pos) > self.y_max):
            # outside stage bounds
            self._readback = 10

        return self._readback # in "V"

class SynHiTpStage(MotorBundle):
    """
    HiTp Sample Stage
    """
    #stage x, y
    px = Cpt(SynAxis, name='stage_x', kind='hinted')
    py = Cpt(SynAxis, name='stage_y', kind='hinted')
    pz = Cpt(SynAxis, name='stage_z', kind='hinted')

    # plate vert adjust motor 1, 2
    vx = Cpt(SynAxis, name='plate_x')
    vy = Cpt(SynAxis, name='plate_y')

    th = Cpt(SynAxis, name='theta')

s_stage = SynHiTpStage('', name='s_stage')

lrf = SynLaserRangeFinder(s_stage.px, s_stage.py, s_stage.vx, 
                            s_stage.vy, name='lrf')

shutter = SynAxis(name='FastShutter')
I1 = SynAxis(name='I1', value=1)
I0 = SynAxis(name='I0', value=1)


class SynBeamStopDetector(Signal):
    def __init__(self, motor_z, I = 5, *args, **kwargs):
        self.stage_z = motor_z
        self.height = np.random.uniform(-3, 3)
        self.I = I
        super().__init__(*args, **kwargs)

    def get(self):
        # Simulate seen intensity with sigmoid.  
        # Brighter if stage below "height"
        h = self.stage_z.read()[self.stage_z.name]['value']
        val = 1 - (self.I / (1 + np.exp(-3 * (h - self.height))))

        self._readback = val
        return self._readback

ptDet = SynBeamStopDetector(s_stage.pz, name='ptDet')

# Create simulated image for dexela detector
def dex_func():
    """imfunc is a function that produces a simulated dexela image
    """
    x = np.linspace(1, 6, num=301)
    intensity = make_random_peaks(x, peak_chance=0.05)*100
    image = generate_image(x, intensity, (512, 512))
    return image

def xsp3_func():
    '''
    Return a simulated MCA array
    '''
    x = np.linspace(1, 2000, num=2000)
    intensity = make_random_peaks(x)
    return intensity

class SynMar(ArraySynSignal, SynTiffFilestore):
    pass

class SynXsp3(ArraySynSignal, SynHDF5Filestore):
    pass


fpath = Path(os.getcwd()) / 'fstore'
print(f'Filestore path: {fpath}')
dexDet = SynMar(name='MarCCD', fstore_path=fpath, func=dex_func)

xsp3 = SynXsp3(name='Xspress3EXAMPLE', fstore_path=fpath, func=xsp3_func)

from ophyd.sim import SynGauss, motor
import time

class DelaySynGauss(SynGauss):
    def trigger(self, *args, **kwargs):
        print('exposure time')
        # time.sleep(20)
        # yield from bps.sleep(5)
        return self.val.trigger(*args, **kwargs)
from ophyd.sim import det
# det = DelaySynGauss('det', motor, 'motor', center=0, Imax=5, sigma=0.5, labels={'detectors'})

def test_rock(det, motor, min, max, *, md=None):
    uid = yield from bps.open_run(md)

    # read exposure time from detector, depends on detector implementation
    try:
        exposure_time = det.exposure_time
    except:
        exposure_time = 30

    yield from bps.trigger(det, wait=False)
    start = time.time()
    now = time.time()
    while (now-start) < exposure_time:
        # this logic is reasonable, but for normal area detectors the trigger won't be waitable
        # will have to read exposure time, then wiggle for that amount of time?
        print('1 cycle')
        yield from bps.sleep(1)
        yield from bps.mov(motor, min)
        yield from bps.mov(motor, max)
        now = time.time()
    yield from bps.create('primary')
    reading = (yield from bps.read(det))
    yield from bps.save()
    yield from bps.close_run()
    return uid


# Looking again at ramp_plan, might be a better option?  Looks a lot like past attempts
# looks like this is more suited toward taking periodic measurements as something moves,
# Rather than taking one measurement as something moves 
from bluesky.plans import ramp_plan
def wiggle_plan(det, motor, start, stop):
    timeout = 60
    def go_plan():
        # Plan to start ramp.  yields a generator and ophyd.StatusBase object
        # Need to correctly finish status object
        # Involves starting the trigger/acquisition
        yield from bps.count()

    def inner_plan():
        # plan to run in the midst of the ramp.  Should save events....?
        # in our case there's no events during this inner plan, since we want 
        # only one exposure
        yield from bps.trigger_and_read(det, name='primary')


    yield from ramp_plan(go_plan, inner_plan, timeout=timeout, take_pre_data=False,
                                period=1)

# finalize imports, namespace
px = s_stage.px
py = s_stage.py
pz = s_stage.pz

vx = s_stage.vx
vy = s_stage.vy

th = s_stage.th

sd.baseline.append(s_stage)