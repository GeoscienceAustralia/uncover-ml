# ga docker file

FROM terriajs/tie-ml-base
MAINTAINER Lachlan McCalman <lachlan.mccalman@nicta.com.au>

# so nlopt lib is found and click works (weird I know
RUN apt-get update && apt-get install -y \
    git \
    #gdal requirement
    libgeos-c1v5 \
    libgeos-dev \
    # Install gdal (yuk!)
  && cd /tmp \
  && wget http://download.osgeo.org/gdal/2.0.1/gdal-2.0.1.tar.gz \
  && tar -xvf gdal-2.0.1.tar.gz \
  && cd gdal-2.0.1 \
  && ./configure --prefix=/usr \
  && make -j8 \
  && make install \
  && export CPLUS_INCLUDE_PATH=/usr/include/gdal \
  && export C_INCLUDE_PATH=/usr/include/gdal \
  && pip3 -v install \ 
    #cmdline
    click \
    #geospatial
    GDAL \
    rasterio \
    pyshp \
    affine \
    # logging and celery worker support
    celery \
    redis \ 
    pytest \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install revrand
RUN pip3 install git+https://github.com/nicta/revrand.git@master

#ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.4/site-packages LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
ENV PYTHONPATH=$PYTHONPATH:/usr/src/python/statbadger:/usr/local/lib/python3.4/site-packages LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
#ENV PYTHONPATH=$PYTHONPATH:/usr/src/python/statbadger:/usr/local/lib/python3.4/site-packages LC_ALL=C.UTF-8 LANG=C.UTF-8 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

