import os
import shutil
import unittest
from dgm1.dgm1_nrw import DGM1NRW


class TestGeom(unittest.TestCase):

    tif_filenames = ['dgm1_32380_5648_2_nw_01m.tif', 'dgm1_32380_5650_2_nw_01m.tif',
                     'dgm1_32382_5648_2_nw_01m.tif', 'dgm1_32382_5650_2_nw_01m.tif']

    shp_dir = './data/shp'
    dgm1_dir = './data/dgm1'

    def _remove_dir(self):
        d = os.path.join(self.dgm1_dir, 'dgm1_nrw_01m_tif')
        if os.path.isdir(d):
            shutil.rmtree(d)

    def _assert_files(self):
        d = os.path.join(self.dgm1_dir, 'dgm1_nrw_01m_tif')
        for f in self.tif_filenames:
            self.assertTrue(os.path.isfile(os.path.join(d, f)), 'File {} not downloaded'.format(f))

    def test_create_shapefile(self):
        d = os.path.join(self.dgm1_dir, 'gis')
        if os.path.isdir(d):
            shutil.rmtree(d)
        # create an instance of the class DGM1XYZ
        dgm1 = DGM1NRW(dgm1_dir=self.dgm1_dir)
        # create a shapefile with polygons (2x2 km²) representing dem-tiles
        dgm1.create_shapefile()

    def test_download_etrs89(self):
        print('Downloading using a shapefile in ETRS89')
        self._remove_dir()

        # create an instance of the class DGM1XYZ
        dgm1 = DGM1NRW(self.dgm1_dir, os.path.join(self.shp_dir, 'area_etrs89.shp'))
        # download intersecting files
        dgm1.download()

        self._assert_files()

    def test_download_wgs84(self):
        print('Downloading using a shapefile in WGS84')
        self._remove_dir()

        # create an instance of the class DGM1XYZ
        dgm1 = DGM1NRW(self.dgm1_dir, os.path.join(self.shp_dir, 'area.shp'))
        # download intersecting files
        dgm1.download()

        self._assert_files()
        self._remove_dir()


if __name__ == '__main__':
    unittest.main(verbosity=3)
