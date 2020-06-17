
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import os
import numpy as np
import random
from pathlib import Path
import uuid

from ophyd import Signal
from ophyd.sim import SynSignal
from ophyd.areadetector.filestore_mixins import resource_factory

import tifffile
import h5py

# Basic signals 
# TODO: sort out file formats into trigger mixins?
class SynTiffFilestore(SynSignal):
    def trigger(self):
        # not running at the moment.... but super.trigger() is.  
        tmpRoot = Path(self.fstore_path)
        tmpPath = 'tmp'
        os.makedirs(tmpRoot / tmpPath, exist_ok=True)
        st = super().trigger() # re-evaluates self._func, puts into value
        # Returns NullType
        ret = super().read()    # Signal.read() exists, not SynSignal.read()
        # But using Signal.read() does not allow uid's to be passed into mem.
        val = ret[self.name]['value']
        
        # AD_TIFF handler generates filename by populating template
        # self.template % (self.path, self.filename, self.point_number)
        self.point_number += 1 
        resource, datum_factory = resource_factory(
                spec='AD_TIFF',
                root=tmpRoot,
                resource_path=tmpRoot / tmpPath,
                resource_kwargs={'template': '%s%s_%d.tiff' , 
                                    'filename': f'{uuid.uuid4()}'},
                path_semantics='windows')
        datum = datum_factory({'point_number': self.point_number})
        
        self._asset_docs_cache.append(('resource', resource))
        self._asset_docs_cache.append(('datum', datum))

        fname = (resource['resource_kwargs']['filename'] 
                    + f'_{self.point_number}.tiff')
        fpath = Path(resource['root']) / resource['resource_path'] / fname
        # for tiff spec
        tifffile.imsave(fpath, val)
        
        # replace 'value' in read dict with some datum id
        ret[self.name]['value'] = datum['datum_id']
        self._last_ret = ret
        return st

class SynHDF5Filestore(SynSignal):
    def trigger(self):
        # not running at the moment.... but super.trigger() is.  
        tmpRoot = Path(self.fstore_path)
        tmpPath = 'tmp'
        os.makedirs(tmpRoot / tmpPath, exist_ok=True)
        st = super().trigger() # re-evaluates self._func, puts into value
        # Returns NullType
        ret = super().read()    # Signal.read() exists, not SynSignal.read()
        # But using Signal.read() does not allow uid's to be passed into mem.
        val = ret[self.name]['value']

        self.point_number += 1 
        fn = f'{uuid.uuid4()}.h5'
        resource, datum_factory = resource_factory(
                spec='XSP3',
                root=tmpRoot,
                resource_path=tmpRoot / tmpPath,
                resource_kwargs={'filename' : fn},
                path_semantics='windows')
        datum = datum_factory({'point_number': self.point_number})
             
        self._asset_docs_cache.append(('resource', resource))
        self._asset_docs_cache.append(('datum', datum))

        fpath = Path(resource['root']) / resource['resource_path'] / fn
        # for h5 spec
        with h5py.File(fpath, 'w') as f:
            e = f.create_group('/entry/instrument/detector')
            dset = e.create_dataset('data', data=val)
        
        # replace 'value' in read dict with some datum id
        ret[self.name]['value'] = datum['datum_id']
        self._last_ret = ret
        return st

class ArraySynSignal(SynSignal):
    """
    Base class for synthetic array signals. 
    Same interface as a normal ArraySignal, but with simulated data and 
    filestore
    """
    _asset_docs_cache = []
    _last_ret = None
    point_number = 0

    def __init__(self, fstore_path=None, *args, **kwargs):
        self.fstore_path = fstore_path
        super(ArraySynSignal, self).__init__(*args, **kwargs)
    
    def describe(self):
        ret = super().describe()
        ret[self.name]['external'] = 'FILESTORE:'
        return ret
    
    def read(self):
        '''Put the status of the signal into a simple dictionary format
        for data acquisition

        Returns
        -------
            dict
        '''
        # Appears to break things, throw resource sentinel issue... 
        # need to initialize sentinel when starting RunEngine
        # Is ostensibly the same as Signal.read()?...
        if self._last_ret is not None:
            return self._last_ret
            # return {self.name: {'value': self._last_ret,
            #                     'timestamp': self.timestamp}}
        else: # If detector has not been triggered already
            raise Exception('read before being triggered')
            # return {self.name: {'value': self.get(),
            #                      'timestamp': self.timestamp}}

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        for item in items:
            yield item

# position generator
def gen_wafer_locs(shape='circle', radius=10):
    """
    Create square grid of locations, with spacing of 1 between
    return lists of x, y locations
    """
    vals = np.arange(-radius, radius+1, 1)

    xv, yv = np.meshgrid(vals, vals)

    x = xv.flatten()
    y = yv.flatten()
    if shape == 'circle':
        xout = [] #np.array([])
        yout = [] # np.array([])
        for i in range(len(x)):
            if (x[i]**2 + y[i]**2) <= radius**2:
                xout.append(x[i])
                yout.append(y[i])
        
        return xout, yout
    else:
        return x, y