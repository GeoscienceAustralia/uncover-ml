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


class BoundingBox:
    def __init__(self, x_range, y_range):
        assert(len(x_range) == len(y_range))
        assert(x_range[0] < x_range[1])
        assert(y_range[0] < y_range[1])
        self.bbox = (x_range, y_range)

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
        self.resolution = resolution

    @property
    def xres(self):
        return self.resolution[0]

    @property
    def yres(self):
        return self.resolution[1]

    def _to_json_dict(self):
        ds = super(GridPointSpec, self)._to_json_dict()
        full_dict = {"resolution": [self.resolution[0], self.resolution[1]]}
        full_dict.update(ds)
        return full_dict

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
