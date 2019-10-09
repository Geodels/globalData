import gc
import sys
import glob
import h5py
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
from operator import itemgetter
from scipy.interpolate import interp1d
from pykdtree.kdtree import KDTree
from pyevtk.hl import gridToVTK
from scipy.ndimage import gaussian_filter

class stratal():

    def __init__(self, filename=None, layer=None):

        # Check input file exists
        try:
            with open(filename) as finput:
                pass
        except IOError as exc:
            print("Unable to open file: ",filename)
            raise IOError('The input file is not found...')

        # Open YAML file
        with open(filename, 'r') as finput:
            self.input = yaml.load(finput, Loader=yaml.Loader)

        self.radius = 6378137.
        self._inputParser()

        self.nbCPUs = len(glob.glob1(self.outputDir+"/h5/","topology.p*"))

        if layer is not None:
            self.layNb = layer
        else:
            self.layNb = len(glob.glob1(self.outputDir+"/h5/","stratal.*.p0.h5"))

        return

    def _inputParser(self):

        try:
            timeDict = self.input['time']
        except KeyError as exc:
            print("Key 'time' is required and is missing in the input file!")
            raise KeyError('Key time is required in the input file!')

        try:
            self.tStart = timeDict['start']
        except KeyError as exc:
            print("Key 'start' is required and is missing in the 'time' declaration!")
            raise KeyError('Simulation start time needs to be declared.')

        try:
            self.tEnd = timeDict['end']
        except KeyError as exc:
            print("Key 'end' is required and is missing in the 'time' declaration!")
            raise KeyError('Simulation end time needs to be declared.')

        try:
            self.strat = timeDict['strat']
        except KeyError as exc:
            print("Key 'strat' is required to build the stratigraphy in the input file!")
            raise KeyError('Simulation stratal time needs to be declared.')

        try:
            outDict = self.input['output']
            try:
                self.outputDir = outDict['dir']
            except KeyError as exc:
                self.outputDir = 'output'
        except KeyError as exc:
            self.outputDir = 'output'

        return

    def _xyz2lonlat(self):

        lons = np.arctan2(self.y/self.radius, self.x/self.radius)
        lats = np.arcsin(self.z/self.radius)

        # Convert spherical mesh longitudes and latitudes to degrees
        self.lonlat = np.empty((self.nbPts,2))
        self.lonlat[:,0] = np.mod(np.degrees(lons)+180.0, 360.0)[:,0]
        self.lonlat[:,1] = np.mod(np.degrees(lats)+90, 180.0)[:,0]

        self.tree = KDTree(self.lonlat,leafsize=10)

        return

    def _lonlat2xyz(self, lon, lat, elev):
        """
        Convert lon / lat (radians) for the spherical triangulation into x,y,z
        on the unit sphere
        """

        xs = np.cos(lat) * np.cos(lon) * self.radius
        ys = np.cos(lat) * np.sin(lon) * self.radius
        zs = np.sin(lat) * self.radius + elev

        return xs, ys, zs


    def _getCoordinates(self):

        for k in range(self.nbCPUs):
            df = h5py.File('%s/h5/topology.p%s.h5'%(self.outputDir,k), 'r')
            coords = np.array((df['/coords']))
            if k == 0:
                self.x, self.y, self.z = np.hsplit(coords, 3)
            else:
                self.x = np.append(self.x, coords[:,0])
                self.y = np.append(self.y, coords[:,1])
                self.z = np.append(self.z, coords[:,2])
            df.close()

        del coords
        self.nbPts = len(self.x)
        self._xyz2lonlat()

        gc.collect()

        return

    def _getData(self,layNb):

        for k in range(self.nbCPUs):
            sf = h5py.File('%s/h5/stratal.%s.p%s.h5'%(self.outputDir,layNb,k), 'r')
            if k == 0:
                self.elev[:,layNb] = np.array((sf['/elev']))[:,0]
                self.erodep[:,layNb] = np.array((sf['/erodep']))[:,0]
                self.sea[layNb] = np.array((sf['/sea']))
            else:
                self.elev[:,layNb] = np.append(self.elev[:,layNb], sf['/elev'])[:,0]
                self.erodep[:,layNb] = np.append(self.erodep[:,layNb], sf['/erodep'])[:,0]
            sf.close()

        return

    def readStratalData(self):

        self._getCoordinates()
        self.sea = np.empty(self.layNb)
        self.elev = np.empty((self.nbPts,self.layNb))
        self.erodep = np.empty((self.nbPts,self.layNb))

        for l in range(self.layNb):
            self._getData(l)

        return

    def _test_progress(self, job_title, progress):

        length = 20
        block = int(round(length*progress))
        msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
        if progress >= 1: msg += " DONE\r\n"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def buildLonLatMesh(self, res=0.1, nghb=3):

        self.nx = int(360./res)
        self.ny = int(180./res)
        self.lon = np.linspace(0., 360., self.nx)
        self.lat = np.linspace(0, 180, self.ny)

        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        xyi = np.dstack([self.lon.flatten(), self.lat.flatten()])[0]
        self.zi = np.empty((self.ny,self.nx,self.layNb))
        self.edi = np.empty((self.ny,self.nx,self.layNb))
        self.thi = np.empty((self.ny,self.nx,self.layNb))

        th = np.zeros((len(xyi),self.layNb))
        th[:,0] = 1.e5

        distances, indices = self.tree.query(xyi, k=nghb)
        weights = 1.0 / distances**2
        denum = 1.0 / np.sum(weights, axis=1)
        onIDs = np.where(distances[:,0] == 0)[0]

        print("Start building regular data arrays")

        for k in range(self.layNb):

            zz = self.elev[:,k]
            ed = self.erodep[:,k]
            self._test_progress("Percentage of arrays built ", k/self.layNb)
            zi = np.sum(weights*zz[indices],axis=1)*denum
            ei = np.sum(weights*ed[indices],axis=1)*denum

            if len(onIDs)>0:
                zi[onIDs] = zz[indices[onIDs,0]]
                ei[onIDs] = ed[indices[onIDs,0]]

            self.zi[:,:,k] = np.reshape(zi,(self.ny,self.nx))
            self.edi[:,:,k] = np.reshape(ei,(self.ny,self.nx))

            if k>0 :
                diff_ed = (self.edi[:,:,k]-self.edi[:,:,k-1]).flatten()
                depoID = np.where(diff_ed>0.)[0]
                th[depoID,k] += diff_ed[depoID]
                eroID = np.where(diff_ed<0.)[0]
                if len(eroID)>0:

                    # Compute cumulative stratal thicknesses
                    cumThick = np.cumsum(th[eroID,k::-1],axis=1)[:,::-1]

                    # Find nodes with no remaining stratigraphic thicknesses
                    ero = -diff_ed[eroID]
                    tmpIDs = np.where(ero>=cumThick[:,0])[0]
                    th[eroID[tmpIDs],:k+1] = 0.

                    # Erode remaining stratal layers
                    if len(tmpIDs) < len(eroID):
                        ero[tmpIDs] = 0.

                        # Clear all stratigraphy points which are eroded
                        cumThick[cumThick < ero.reshape((len(eroID),1))] = 0.
                        mask = (cumThick > 0).astype(int) == 0
                        tmpH = th[eroID,:k+1]
                        tmpH[mask] = 0
                        th[eroID,:k+1] = tmpH

                        # Update thickness of top stratigraphic layer
                        eroIDs = np.bincount(np.nonzero(cumThick)[0])-1
                        eroVals = cumThick[np.arange(len(eroID)),eroIDs]-ero
                        eroVals[tmpIDs] = 0.
                        th[eroID,eroIDs] = eroVals

                th[:,0] = 1.e5

        for k in range(self.layNb):
            if k == 0:
                self.thi[:,:,k] = 0.
            else:
                self.thi[:,:,k] = np.reshape(th[:,k],(self.ny,self.nx))

        del weights, denum, onIDs, zz, zi, ei, ed, xyi, th
        gc.collect()

        return

    def writeMesh(self, vtkfile='mesh', lons=None, lats=None, hscale=1000., sigma=1.):
        """
        Create a vtk unstructured grid based on current time step stratal parameters.
        Parameters
        ----------
        variable : outfolder
            Folder path to store the stratal vtk mesh.
        """

        lon = np.linspace(0., 360., self.nx)
        lat = np.linspace(0., 180., self.ny)

        if lons is None:
            lons = [lon[0],lon[1]]
        if lats is None:
            lats = [lat[0],lat[1]]

        x = np.zeros((lons[1]-lons[0], lats[1]-lats[0], self.layNb))
        y = np.zeros((lons[1]-lons[0], lats[1]-lats[0], self.layNb))
        z = np.zeros((lons[1]-lons[0], lats[1]-lats[0], self.layNb))
        e = np.zeros((lons[1]-lons[0], lats[1]-lats[0], self.layNb))
        h = np.zeros((lons[1]-lons[0], lats[1]-lats[0], self.layNb))
        t = np.zeros((lons[1]-lons[0], lats[1]-lats[0], self.layNb))

        zz = self.zi[lats[0]:lats[1],lons[0]:lons[1],-1]
        zz = gaussian_filter(zz, sigma)

        for k in range(self.layNb-1,-1,-1):
            th = gaussian_filter(self.thi[:,:,k], sigma)
            th[th<0] = 0.
            if k < self.layNb-1:
                thu = gaussian_filter(self.thi[:,:,k+1], sigma)
                thu[thu<0] = 0.
            for j in range(lats[1]-lats[0]):
                for i in range(lons[1]-lons[0]):
                    x[i,j,k] = lon[i+lons[0]]*hscale
                    y[i,j,k] = lat[j+lats[0]]*hscale
                    if k == self.layNb-1:
                        z[i,j,k] = zz[j,i]
                    else:
                        z[i,j,k] = z[i,j,k+1]-thu[j+lats[0],i+lons[0]]
                    e[i,j,k] = self.zi[j+lats[0],i+lons[0],k]
                    h[i,j,k] = th[j+lats[0],i+lons[0]]
                    t[i,j,k] = k

        gridToVTK(vtkfile, x, y, z, pointData = {"dep elev" : e, "th" :h, "layID" :t})

        return
