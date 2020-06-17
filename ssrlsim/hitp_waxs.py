'''
OPHYD classes for simulated beamline SSRL1-5, in hitp mode of operation
Should mimic ophyd devices as closely as possible for operation

Many ideas stolen from various bluesky tutorials
'''
import os
import numpy as np
from pathlib import Path

from ophyd import Signal, Device, Component as Cpt
from ophyd.sim import SynAxis

import bluesky.plan_stubs as bps
from ssrltools.devices.hitp import HiTpStage

from . import ArraySynSignal, gen_wafer_locs, SynTiffFilestore, SynHDF5Filestore
from .images import make_random_peaks, generate_image

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

class SynHiTpStage(HiTpStage):
    """
    Combined class for HiTp stage.  
    * Gathers stage and plate motors
    * Stores sample locations

    Simplifies task of aligning and remembering sample positions
    """
    #stage x, y
    stage_x = Cpt(SynAxis, name='stage_x', kind='hinted')
    stage_y = Cpt(SynAxis, name='stage_y', kind='hinted')
    stage_z = Cpt(SynAxis, name='stage_z', kind='hinted')

    # plate vert adjust motor 1, 2
    plate_x = Cpt(SynAxis, name='plate_x')
    plate_y = Cpt(SynAxis, name='plate_y')

    theta = Cpt(SynAxis, name='theta')

stage = SynHiTpStage(name='HiTp_Stage')

lrf = SynLaserRangeFinder(stage.stage_x, stage.stage_y, stage.plate_x, 
                            stage.plate_y, name='lrf')


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

ptDet = SynBeamStopDetector(stage.stage_z, name='ptDet')

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
    x = np.linspace(1, 6, num=301)
    intensity = make_random_peaks(x)
    return intensity

class SynDex(ArraySynSignal, SynTiffFilestore):
    pass

class SynXsp3(ArraySynSignal, SynHDF5Filestore):
    pass


fpath = Path(os.getcwd()) / 'fstore'
print(f'Filestore path: {fpath}')
dexDet = SynDex(name='Dexela 2923', fstore_path=fpath, func=dex_func)

xsp3 = SynXsp3(name='Xspress3EXAMPLE', fstore_path=fpath, func=xsp3_func)