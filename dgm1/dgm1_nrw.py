# -- coding: ISO-8859-1 --
import os
import time
import glob
import gzip
import shutil
from abc import ABC

import numpy as np
import multiprocessing
import urllib.request
from timeit import default_timer
from osgeo import ogr, gdal, osr
from html.parser import HTMLParser


class DGM1NRW(object):
    """The class DGM1NRW downloads and processes dem rasters found in
    `https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_xyz/dgm1_xyz/`.

    :param dgm1_dir: directory to save downloaded 1 meter rasters as well as its shapefile and resampled rasters.
    :type dgm1_dir: str
    :param shp_region: shapefile with the region of interest. If defined, all processes refer to this shapefile.
    :type shp_region: str
    """

    nodata = -9999.0
    """nodata for the created rasters"""

    url_dgm1_xyz = 'https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_xyz/dgm1_xyz/'
    """URL of dem rasters with one meter resolution and 2000x2000 :math:`m^2` size."""

    compress_options = ['COMPRESS=LZW', 'PREDICTOR=2']
    """Compress options used in gdal driver.Create()."""

    creation_options = ["BIGTIFF=YES", "COMPRESS=LZW", "ZSTD_LEVEL=1", "TILED=YES", "PREDICTOR=2"]

    def __init__(self, dgm1_dir, shp_region=None):
        dgm1_dir = os.path.expanduser(dgm1_dir.strip())
        if shp_region:
            shp_region = os.path.expanduser(shp_region.strip())

        self.shp_region = shp_region
        """Shapefile with the region of interest. If defined, all processes refer to this shapefile. Otherwise all 
        rasters are downloaded and processed"""

        self.dgm1_dir = dgm1_dir
        """Directory to save the original 1 meter products as well as its shapefile and resampled rasters."""

        self.shp_filename_tiles = os.path.join(self.dgm1_dir, 'gis', 'dgm1_2_nw.shp')
        """Shapefile `dgm1_2_nw.shp` contaning all tiles found on the server as 2x2 :math:`km^2` squares. It is saved 
        in :attr:`dgm1_dir`/gis."""

        os.makedirs(os.path.dirname(self.shp_filename_tiles), exist_ok=True)

    def tif_dir(self, pixel_size):
        """Return the directory name of rasters with the given pixel_size. pixel_size = 1 corresponds to the
        downloaded rasters with 1 meter resolution.

        :param pixel_size: pixel size in meters. It must be a divisor of 2000 (2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80,
            100, 125, 200, 250, 400, 500, 1000, 2000).
        :type pixel_size: int
        :return: (str) full path of the directory.
        """
        d = os.path.join(self.dgm1_dir, 'dgm1_nrw_{:02d}m_tif'.format(pixel_size))
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def vrt_filename(pixel_size):
        """Return the basename of the vrt-file for the given pixel size.

        :param pixel_size: pixel size in meters. It must be a divisor of 2000 (2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80,
            100, 125, 200, 250, 400, 500, 1000, 2000).
        :type pixel_size: int
        :return: (str) basename of the vrt-file
        """
        return 'dgm1_nrw_{:02d}m.vrt'.format(pixel_size)

    def tif_filenames(self, pixel_size):
        """Return the names of the downloaded files with the given pixel files in tif-format.

        :param pixel_size: pixel size in meters. It must be a divisor of 2000 (2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80,
            100, 125, 200, 250, 400, 500, 1000, 2000).
        :type pixel_size: int
        :return: (list) full path file names.
        """
        return list(glob.glob('{}/dgm1_*_2_nw*.tif'.format(self.tif_dir(pixel_size))))

    @staticmethod
    def srs():
        """Return the compound coordinate reference system using `EPSG 25832` for the horizontal component and
        `EPSG 7837` for the vertical component. The compound srs is labeled as
        ``ETRS89 / UTM zone 32N + DHHN2016 height``.

        :return: (osr.SpatialReference) spatial reference system.
        """
        sr_25832 = osr.SpatialReference()
        sr_7837 = osr.SpatialReference()
        sr_25832.ImportFromEPSG(25832)
        sr_7837.ImportFromEPSG(7837)
        srs = osr.SpatialReference()
        srs.SetCompoundCS("ETRS89 / UTM zone 32N + DHHN2016 height", sr_25832, sr_7837)
        return srs

    def gz_filenames_intersecting_region(self):
        """Return a list of gzipped file names (dgm1_XXXXX_YYYY_2_nw.xyz.gz) found on the server, which
        intersect the region :attr:`shp_region`. Return an empty list if the region is not defined or it is outside the
        domain :attr:`shp_filename_tiles`.

        :return: (list) list of files with the suffix `xyz.gz`

        Example::

           >>> dgm1 = DGM1NRW('~/dgm1', '~/study_area/region.shp')
           >>> dgm1.gz_filenames_intersecting_region()
           ['dgm1_32350_5672_2_nw.xyz.gz', 'dgm1_32350_5674_2_nw.xyz.gz', ...
        """
        gz_filenames = []
        if not self.shp_region:
            return gz_filenames
        if not os.path.isfile(self.shp_filename_tiles):
            self.create_shapefile()

        shp_region = self.shp_region
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds_region = driver.Open(shp_region)
        ds_tiles = driver.Open(self.shp_filename_tiles)
        lr_region = ds_region.GetLayer(0)
        lr_tiles = ds_tiles.GetLayer(0)

        source = lr_region.GetSpatialRef()
        target = osr.SpatialReference()
        target.ImportFromEPSG(25832)
        transform = osr.CoordinateTransformation(source, target)

        for feat_region in lr_region:
            polygon = feat_region.GetGeometryRef()
            polygon.Transform(transform)
            lr_tiles.SetSpatialFilter(polygon)
            for feat_tiles in lr_tiles:
                gz_filenames.append(feat_tiles.GetField('Filename'))
        return gz_filenames

    def gz_filenames_intersecting_region_envelope(self):
        """Return a list of gzipped file names (dgm1_XXXXX_YYYY_2_nw.xyz.gz) found on the server, which
        intersect the envelopen of the region :attr:`shp_region`. Return an empty list if the region is not defined
        or the envelope is outside the domain :attr:`shp_filename_tiles`.

        :return: (list) list of files with the suffix `xyz.gz`

        Example::

           >>> dgm1 = DGM1NRW('~/dgm1', '~/study_area/region.shp')
           >>> dgm1.gz_filenames_intersecting_region()
           ['dgm1_32350_5672_2_nw.xyz.gz', 'dgm1_32350_5674_2_nw.xyz.gz', ...
        """
        gz_filenames = []
        if not self.shp_region:
            return gz_filenames
        if not os.path.isfile(self.shp_filename_tiles):
            self.create_shapefile()

        x_min, y_min, x_max, y_max = self.region_envelope()
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(x_min, y_min)
        ring.AddPoint(x_min, y_max)
        ring.AddPoint(x_max, y_max)
        ring.AddPoint(x_max, y_min)
        ring.AddPoint(x_min, y_min)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)

        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds_tiles = driver.Open(self.shp_filename_tiles)
        lr_tiles = ds_tiles.GetLayer(0)
        lr_tiles.SetSpatialFilter(polygon)
        for feat_tiles in lr_tiles:
            gz_filenames.append(feat_tiles.GetField('Filename'))
        ds_tiles = None
        return gz_filenames

    def create_shapefile(self):
        """Create a shapefile with squares of 2x2 km² using raster file names from :attr:`url_dgm1_xyz`.
        The shapefile :attr:`shp_filename_tiles` is saved in :attr:`dgm1_dir`. The shapefile attribute `Filename`
        indicates the gz-file name found on the server. The spatial coordinate system is defined in
        :meth:`create_spatial_reference`.

        """
        file_names_server = DGM1HTMLParser.get_filenames()
        driver = ogr.GetDriverByName("ESRI Shapefile")
        ds = driver.CreateDataSource(self.shp_filename_tiles)
        layer = ds.CreateLayer("dgm1_2_nw_xyz", self.srs(), ogr.wkbPolygon)
        field_name = ogr.FieldDefn("Filename", ogr.OFTString)
        field_name.SetWidth(27)
        layer.CreateField(field_name)

        for gz_filename in file_names_server:
            s = gz_filename.split('_')
            x_min = float(s[1][2:]) * 1000.0 - 0.5
            y_min = float(s[2]) * 1000.0 - 0.5
            x_max = x_min + 2000.0
            y_max = y_min + 2000.0
            feature = ogr.Feature(layer.GetLayerDefn())
            # Set the attributes using the values from the delimited text file
            feature.SetField("Filename", gz_filename)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x_min, y_min)
            ring.AddPoint(x_min, y_max)
            ring.AddPoint(x_max, y_max)
            ring.AddPoint(x_max, y_min)
            ring.AddPoint(x_min, y_min)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            feature.SetGeometry(poly)
            layer.CreateFeature(feature)
            feature = None
        ds = None

    @staticmethod
    def _download(n, dgm1_01m_tif, args):
        i, file_name_server = args[0], args[1]
        t0 = default_timer()
        file_name_tif = os.path.join(dgm1_01m_tif, file_name_server.replace('.xyz.gz', '.tif'))
        url = 'https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_xyz/dgm1_xyz/'
        with urllib.request.urlopen(os.path.join(url, file_name_server)) as f:
            arr = np.array([float(i) for v in gzip.decompress(f.read()).split(b'\n') for i in v.split()])
            try:
                arr = arr.reshape((4000000, 3))
            except ValueError as e:
                print('Filename: {}, array.shape = {}: {}'.format(file_name_server, str(arr.shape), e))
                return None
            arr[:, 0] -= 32000000.0
            x_min = arr[0, 0] - 0.5
            y_max = arr[-1, 1] + 0.5
            arr = arr.reshape((2000, 2000, 3))
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(file_name_tif, 2000, 2000, 1, gdal.GDT_Float32, DGM1NRW.compress_options)
            ds.SetGeoTransform((x_min, 1.0, 0, y_max, 0, -1.0))
            ds.SetProjection(DGM1NRW.srs().ExportToWkt())
            bn = ds.GetRasterBand(1)
            bn.SetNoDataValue(DGM1NRW.nodata)
            bn.WriteArray(np.rot90(arr[:, :, 2]))
            bn.FlushCache()
        print('{:04d}/{:04d} {} downloaded in {:0.2f} seconds.'.format(i, n, file_name_server, default_timer() - t0))
        return file_name_server

    def download(self, n_cores=1):
        r"""Download rasters from :attr:`url_dgm1_xyz` intersecting :attr:`shp_region`. Download all rasters if
        :attr:`region` is not defined.

        :param n_cores: number of cores for parallel downloading and transformation into tif-files. n_core is limited
            to :math:`n\_cores \leq n - 1`, where n ist the total number of cores found on the computer. n_cores can
            be intentionally be high in order to use :math:`n - 1` cores.
        """
        from functools import partial
        t0 = default_timer()

        def _files_to_download():
            if self.shp_region:
                file_names_server = self.gz_filenames_intersecting_region_envelope()
                if not file_names_server:
                    raise ValueError('No tile found in the region {}'.format(self.shp_region))
            else:
                file_names_server = DGM1HTMLParser.get_filenames()
            file_names_local = list(glob.glob('{}/dgm1_*_2_nw.tif'.format(self.tif_dir(1))))
            file_names_local = set([os.path.basename(f).replace('.tif', '.xyz.gz') for f in file_names_local])
            return sorted(list(set(file_names_server).difference(file_names_local)))

        files_to_download = _files_to_download()
        files_to_download = [(i + 1, f) for i, f in enumerate(files_to_download)]
        if not files_to_download:
            print('All files are already downloaded')
            return
        n = len(files_to_download)
        n_cores = max(1, min(n, n_cores, multiprocessing.cpu_count() - 1))
        print('Downloading {} files from {} using {} cores'.format(n, self.url_dgm1_xyz, n_cores))
        dgm1_01m_tif = self.tif_dir(1)
        pool = multiprocessing.Pool(n_cores)
        result = pool.map(partial(DGM1NRW._download, n, dgm1_01m_tif), files_to_download)
        result = [r for r in result if r]
        pool.close()
        pool.join()
        print('{} files downloaded in {:0.2f} seconds'.format(len(result), default_timer() - t0))

    def resample(self, pixel_size, force=False):
        """Resample dowloaded TIF files with 1 meter resolution into TIF files with a resolution of pixel_size.

        :param pixel_size: pixel size in meters. It must be a divisor of 2000 (2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80,
            100, 125, 200, 250, 400, 500, 1000, 2000)
        :type pixel_size: int
        :param force: if True, resample also existing files, otherwise skip them. Default force=False
        :type force: bool
        """
        if pixel_size <= 1:
            raise ValueError('Pixel size {} must be greater than 1 meter.'.format(pixel_size))
        t0 = default_timer()
        tif_1m_dir = self.tif_dir(1)
        tif_xm_dir = self.tif_dir(pixel_size)

        tif_filenames = {os.path.splitext(f)[0]: (
            os.path.join(tif_1m_dir,  f),
            os.path.join(tif_xm_dir,  f.replace('.tif', '_{:02d}m.tif'.format(pixel_size))))
            for f in os.listdir(tif_1m_dir) if f.endswith('_nw.tif') and f.startswith('dgm1_')}

        srs_wkt = self.srs().ExportToWkt()
        n = len(tif_filenames)
        for i, (tif_1m_filename, tif_xm_filename) in enumerate(sorted(tif_filenames.values())):
            if force or not os.path.isfile(tif_xm_filename):
                t00 = time.time()
                ds_1m = gdal.Open(tif_1m_filename, gdal.GA_ReadOnly)
                x_min, x_res, _, y_max, _, y_res = ds_1m.GetGeoTransform()
                y_res = -y_res
                x_len = ds_1m.RasterXSize * x_res
                y_len = ds_1m.RasterYSize * y_res
                if x_len % pixel_size != 0:
                    raise ValueError('Could not resample to {} m. '.format(pixel_size) +
                                     'Raster length/pixel size ({}/{}) is not integer.'.format(x_len, pixel_size))
                if y_len % pixel_size != 0:
                    raise ValueError('Could not resample to {} m. '.format(pixel_size) +
                                     'Raster height/pixel size ({}/{}) is not integer.'.format(y_len, pixel_size))
                n_x = int(x_len / pixel_size)
                n_y = int(y_len / pixel_size)
                driver = gdal.GetDriverByName('GTiff')
                ds_xm = driver.Create(tif_xm_filename, n_x, n_y, 1, gdal.GDT_Float32, DGM1NRW.compress_options)
                ds_xm.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
                ds_xm.SetProjection(srs_wkt)
                bn_xm = ds_xm.GetRasterBand(1)
                bn_xm.SetNoDataValue(DGM1NRW.nodata)
                gdal.ReprojectImage(ds_1m, ds_xm, srs_wkt, srs_wkt, gdal.GRA_Bilinear)
                ds_xm = None
                print('{}/{} {} resampled in {:0.2f} seconds.'.format(i, n, os.path.basename(tif_xm_filename), time.time() - t00))
        print('Resample finished in {:0.2f} seconds'.format(default_timer() - t0))

    def region_envelope(self):
        """Return [x_min, y_min, x_max, y_max] for :attr:`shp_region`

        :return: (list) envelope of :attr:`shp_region`
        """
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds_region = driver.Open(self.shp_region)
        lr_region = ds_region.GetLayer(0)
        srs_region = lr_region.GetSpatialRef()
        srs_dgm1 = self.srs()
        transform = osr.CoordinateTransformation(srs_region, srs_dgm1)

        env = list(lr_region.GetExtent())
        env[1], env[2] = env[2], env[1]
        for i, j in [[0, 1], [2, 3]]:
            p = ogr.Geometry(ogr.wkbPoint)
            p.AddPoint(env[i], env[j])
            p.Transform(transform)
            env[i], env[j] = p.GetX(), p.GetY()
        return env

    def create_vrt(self, vrt_filename, pixel_size=1, n_cores=1):
        """Create a raster named :attr:`vrt_filename` in vrt-format. A folder without the suffix `.vrt` will be created
        together with the and populated with rasters of the given :attr:`pixel_size` intersecting with the region.

        Example::

           >>> dgm1 = DGM1NRW('~/dgm1', '~/study_area/region.shp')
           >>> dgm1.create_vrt('~/study_area/dgm1/dgm1_region_02m.vrt', pixel_size=2)

        :param vrt_filename: full path vrt file name (.vrt)
        :type vrt_filename: str
        :param pixel_size: pixel size in meters. It must be a divisor of 2000 (2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80,
            100, 125, 200, 250, 400, 500, 1000, 2000)
        :type pixel_size:
        :param n_cores: number of cores for parallel downloading and transformation into TIF files. n_core is limited
            to :math:`n\_cores \leq n - 1`, where n ist the total number of cores found on the computer. n_cores can
            be intentionally be high in order to use :math:`n - 1` cores.
        :type n_cores: int
        """
        t0 = default_timer()
        vrt_filename = os.path.expanduser(vrt_filename.strip())
        print('Creating {}'.format(vrt_filename), end=' ')
        if not os.path.isfile(self.shp_region):
            raise ValueError('{} not found'.format(self.shp_region) if self.shp_region else 'Region not defined')
        output_dir = os.path.dirname(vrt_filename)
        os.makedirs(output_dir, exist_ok=True)
        tif_filenames = ['{}.tif'.format(f[:-7]) for f in self.gz_filenames_intersecting_region()]
        if not tif_filenames:
            raise ValueError('No TIF file in {} intersects the region {}'.format(self.tif_dir(1), self.shp_region))
        if not all([os.path.isfile(os.path.join(self.tif_dir(1), f)) for f in tif_filenames]):
            print()
            self.download(n_cores=n_cores)
            self.resample(pixel_size)
        tif_dir = self.tif_dir(pixel_size)
        if pixel_size > 1:
            filenames = [f.split('.')[0] for f in tif_filenames]
            n = len(filenames[0])
            filenames = set(filenames)
            tif_filenames = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f[:n] in filenames]
        else:
            tif_filenames = [os.path.join(tif_dir, f) for f in tif_filenames]
        tif_dir = os.path.join(os.path.dirname(vrt_filename),
                               os.path.splitext(os.path.basename(vrt_filename))[0])
        os.makedirs(tif_dir, exist_ok=True)
        tif_filenames_vrt = []
        for i, f_in in enumerate(tif_filenames):
            f_out = os.path.join(tif_dir, os.path.basename(f_in))
            tif_filenames_vrt.append(f_out)
            if not os.path.isfile(f_out):
                shutil.copy(f_in, tif_dir)

        gdal.BuildVRT(vrt_filename, tif_filenames_vrt, options=gdal.BuildVRTOptions(
            outputBounds=self.region_envelope(), resampleAlg='nearest', addAlpha=True))
        print('finished in {:0.2f} seconds'.format(default_timer() - t0))

    def mosaic(self, tif_filename, pixel_size, extent='region', n_cores=1, **kwargs):
        """Mosaic TIF files intersecting with the :attr:`shp_region`.

        :param tif_filename: output file name (.tif)
        :type tif_filename: str
        :param pixel_size: pixel size in meters. It must be a divisor of 2000 (2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80,
            100, 125, 200, 250, 400, 500, 1000, 2000)
        :type pixel_size: int
        :param extent: `clip` to clip :attr:`shp_region`; `region` for a raster covering the envelope of
            :attr:`shp_region`; `rasters` for a raster extended to full tiles. Default `region`. extent=None corresponds
            to extent=`rasters`
        :type extent: str
        :param n_cores: number of cores for parallel downloading and transformation into TIF files. n_core is limited
            to :math:`n\_cores \leq n - 1`, where n ist the total number of cores found on the computer. n_cores can
            be intentionally be high in order to use :math:`n - 1` cores.
        :type n_cores: int

        :Keyword Arguments:

           See https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
        """
        t0 = default_timer()

        if extent not in ['clip', 'region', 'rasters']:
            raise ValueError('extent = {} invalid. The parameter extent must be one of {}'.format(
                extent, ', '.join(['clip', 'region', 'rasters'])))

        tif_filename = os.path.expanduser(tif_filename.strip())
        print('Creating {}'.format(tif_filename), end=' ')
        if not os.path.isfile(self.shp_region):
            raise ValueError('{} not found'.format(self.shp_region) if self.shp_region else 'Region not defined')
        output_dir = os.path.dirname(tif_filename)
        os.makedirs(output_dir, exist_ok=True)

        tif_filenames = ['{}.tif'.format(f[:-7]) for f in self.gz_filenames_intersecting_region_envelope()]
        if not tif_filenames:
            raise ValueError('No TIF file in {} intersects the region {}'.format(self.tif_dir(1), self.shp_region))
        if not all([os.path.isfile(os.path.join(self.tif_dir(1), f)) for f in tif_filenames]):
            print()
            self.download(n_cores=n_cores)
            self.resample(pixel_size)

        tif_dir = self.tif_dir(pixel_size)
        if pixel_size > 1:
            filenames = [f.split('.')[0] for f in tif_filenames]
            n = len(filenames[0])
            filenames = set(filenames)
            tif_filenames = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f[:n] in filenames]
        else:
            tif_filenames = [os.path.join(tif_dir, f) for f in tif_filenames]

        n_cores = max(1, min(n_cores, multiprocessing.cpu_count() - 1))

        gdal.UseExceptions()

        co = DGM1NRW.creation_options if n_cores == 1 else DGM1NRW.creation_options + ["NUM_THREADS={}".format(n_cores)]
        if extent == 'clip':
            ds = gdal.Warp(tif_filename, tif_filenames, creationOptions=co, cutlineDSName=self.shp_region,
                           cropToCutline=True, **kwargs)
        elif extent == 'region':
            env = self.region_envelope()
            ds = gdal.Warp(tif_filename, tif_filenames, creationOptions=co, outputBounds=env, **kwargs)
        else:
            ds = gdal.Warp(tif_filename, tif_filenames, creationOptions=co, **kwargs)

        ds = None
        gdal.TranslateOptions()
        print('finished in {:0.2f} seconds'.format(default_timer() - t0))


class DGM1HTMLParser(HTMLParser, ABC):

    def __init__(self):
        super(DGM1HTMLParser, self).__init__()
        self.reset()
        self.filenames = list()

    def handle_starttag(self, tag, attrs):
        od = dict(attrs)
        if tag == 'file' and 'name' in od:
            filename = od['name']
            if filename.startswith('dgm1_') and filename.endswith('_nw.xyz.gz'):
                self.filenames.append(od['name'])

    @staticmethod
    def get_filenames():
        p = DGM1HTMLParser()
        with urllib.request.urlopen('https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_xyz/dgm1_xyz/') as f:
            p.feed(f.read().decode("utf8"))
            return p.filenames
