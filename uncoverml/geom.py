from __future__ import division

from affine import Affine
import numpy as np
import shapefile


def lonlat_pixel_centres(raster):

    # Get affine transform for pixel centres
    # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
    T1 = raster.affine * Affine.translation(0.5, 0.5)

    # No shearing or rotation allowed!!
    if not ((T1[1] == 0) and (T1[3] == 0)):
        raise RuntimeError("Transform to pixel coordinates has rotation "
                           "or shear")

    # compute the tiffs lat/lons
    f_lons = T1[2] + np.arange(raster.width) * T1[0]
    f_lats = T1[5] + np.arange(raster.height) * T1[4]
    return f_lons, f_lats


def bounding_box(raster):
    T1 = raster.affine

    # No shearing or rotation allowed!!
    if not ((T1[1] == 0) and (T1[3] == 0)):
        raise RuntimeError("Transform to pixel coordinates has rotation "
                           "or shear")

    # the +1 because we want pixel corner 1 beyond the last pixel
    lon_range = T1[2] + np.array([0, raster.width + 1]) * T1[0]
    lat_range = T1[5] + np.array([0, raster.height + 1]) * T1[4]

    lon_range = np.sort(lon_range)
    lat_range = np.sort(lat_range)
    return lon_range, lat_range


def points_from_shp(filename):
    # TODO check the shapefile only contains points
    coords = []
    sf = shapefile.Reader(filename)
    for shape in sf.iterShapes():
        coords.append(list(shape.__geo_interface__['coordinates']))
    label_coords = np.array(coords)
    return label_coords


def values_from_shp(filename, field):

    sf = shapefile.Reader(filename)
    fdict = {f[0]: i for i, f in enumerate(sf.fields[1:])}  # Skip DeletionFlag

    if field not in fdict:
        raise ValueError("Requested field is not in records!")

    vind = fdict[field]
    vals = [r[vind] for r in sf.records()]

    return np.array(vals)


class BoundingBox:
    def __init__(self, x_range, y_range):
        assert(len(x_range) == len(y_range))
        assert(x_range[0] < x_range[1])
        assert(y_range[0] < y_range[1])
        self.bbox = (tuple(x_range), tuple(y_range))

    @property
    def x_range(self):
        return self.bbox[0]

    @property
    def y_range(self):
        return self.bbox[1]

    @property
    def xmin(self):
        return self.bbox[0][0]

    @property
    def xmax(self):
        return self.bbox[0][1]

    @property
    def ymin(self):
        return self.bbox[1][0]

    @property
    def ymax(self):
        return self.bbox[1][1]

    def contains(self, otherbbox):
        """ Fully contains another BoundingBox object? """

        if (self.xmin <= otherbbox.xmin) \
                and (self.xmax >= otherbbox.xmax) \
                and (self.ymin <= otherbbox.ymin) \
                and (self.ymax >= otherbbox.ymax):
            return True

        return False

    def _to_json_dict(self):
        return {"bounding_box": {"min": [self.bbox[0][0], self.bbox[1][0]],
                                 "max": [self.bbox[0][1], self.bbox[1][1]]}}

    @classmethod
    def _from_json_dict(cls, json_dict):
        bbox = json_dict["bounding_box"]
        x_range = (bbox['min'][0], bbox['max'][0])
        y_range = (bbox['min'][1], bbox['max'][1])
        return cls(x_range, y_range)


class GridPointSpec(BoundingBox):
    def __init__(self, x_range, y_range, resolution):
        assert(len(resolution) == 2)
        assert((resolution[0] > 0) and (resolution[1] > 0))
        super(GridPointSpec, self).__init__(x_range, y_range)
        self.resolution = tuple(resolution)
        self.__Affine()

    @property
    def xres(self):
        return self.resolution[0]

    @property
    def yres(self):
        return self.resolution[1]

    @property
    def npoints(self):
        return self.resolution[0]*self.resolution[1]

    def lonlat2pix(self, lonlat):

        xy = np.floor([self.iA * ll for ll in lonlat]).astype(int)
        assert any(np.logical_and(xy[:, 0] >= 0,
                                  xy[:, 0] < self.resolution[0]))
        assert any(np.logical_and(xy[:, 1] >= 0,
                                  xy[:, 1] < self.resolution[1]))

        return xy

    def pix2latlon(self, xy):

        lonlat = np.array([self.A * pix for pix in xy])
        assert any(np.logical_and(lonlat[:, 0] >= self.xmin,
                                  lonlat[:, 0] < self.xmax))
        assert any(np.logical_and(lonlat[:, 1] >= self.ymin,
                                  lonlat[:, 1] < self.ymax))

        return lonlat

    def _to_json_dict(self):
        ds = super(GridPointSpec, self)._to_json_dict()
        full_dict = {"resolution": [self.resolution[0], self.resolution[1]]}
        full_dict.update(ds)
        return full_dict

    def __Affine(self):

        self.pixsize_x = (self.xmax - self.xmin) / (self.resolution[0] + 1)
        self.pixsize_y = (self.ymax - self.ymin) / (self.resolution[1] + 1)
        self.A = Affine(self.pixsize_x, 0, self.xmin,
                        0, -self.pixsize_y, self.ymax)
        self.A *= Affine.translation(0.5, 0.5)
        self.iA = _invert_affine(self.A)

    @classmethod
    def _from_json_dict(cls, json_dict):
        bbox = BoundingBox._from_json_dict(json_dict)
        resolution = json_dict["resolution"]
        return cls(bbox.x_range, bbox.y_range, resolution)


class ListPointSpec(BoundingBox):
    def __init__(self, coords, x_range=None, y_range=None):
        assert(coords.ndim == 2)
        assert(coords.shape[1] == 2)

        # compute the bounding box if I need to
        if x_range is None:
            x_range = (np.amin(coords[:, 0]), np.amax(coords[:, 0]))
        if y_range is None:
            y_range = (np.amin(coords[:, 1]), np.amax(coords[:, 1]))

        super(ListPointSpec, self).__init__(x_range, y_range)
        self.coords = coords

    @property
    def xcoords(self):
        return self.coords[:, 0]

    @property
    def ycoords(self):
        return self.coords[:, 1]

    @property
    def npoints(self):
        return self.coords.shape[0]

    def _to_json_dict(self):
        ds = super(ListPointSpec, self)._to_json_dict()
        full_dict = {"coordinates": self.coords.tolist()}
        full_dict.update(ds)
        return full_dict

    @classmethod
    def _from_json_dict(cls, json_dict):
        bbox = BoundingBox._from_json_dict(json_dict)
        coords = np.array(json_dict["coordinates"])
        return cls(coords, x_range=bbox.x_range, y_range=bbox.y_range)


def unserialise(json_dict):
    """
    returns a PointSpec object corresponding to the type of the json dict
    """

    if "coordinates" in json_dict:
        pspec = ListPointSpec._from_json_dict(json_dict)
    elif "resolution" in json_dict:
        pspec = GridPointSpec._from_json_dict(json_dict)
    else:
        raise RuntimeError("Invalid pointspec object input")

    return pspec


def _invert_affine(A):

    R = np.array([A[0:2], A[3:5]])
    T = np.array([[A[2], A[5]]]).T

    iR = np.linalg.pinv(R)
    iT = -iR.dot(T)
    iA = np.hstack((iR, iT))

    return Affine(*iA.flatten())
