Command Line Scripts
====================

.. currentmodule:: uncoverml.scripts

.. autosummary::
   :toctree: generated/

   maketargets
   extractfeats
   composefeats
   learnmodel
   predict
   validatemodel
   exportgeotiff
   tiff2kmz


Example Learning pipeline
-------------------------

 - maketargets - turns a shapefile of targets into an hdf5 file of targets and
   returns indices to split the targets into folds.
 - extractfeats - takes targets and geotiffs, extracts pixels/patches at the
   target locations and puts them into hdf5 chunk files (#chunks = #cpus), total
   number of files = #geotiffs * #chunks
 - composefeats - takes the chunked hdf5s created in the previous step and
   concatenates and transforms them (whitening, imputation etc), total number
   of files = #chunks, these are our X's
 - learnmodel - takes the output of composefeats (or extract feats if we want),
   and takes the target and crossval indices, and learns a model, saving a
   pickled python object. This only happens on one node (so ideally we'd use a
   node with many cpus here). Since we have to train on one machine, the
   learning dataset has to fit in memory of this one machine.
 - predictmodel - takes the pickled model, the composefeats outputs and
   predicts targets (for all X's)
 - validatemodel - takes the targets, crossval indices and predictions, and
   does the model validation

Example Prediction pipeline
---------------------------

 - extractfteats - takes the geotiffs and extracts all pixels patches and puts
   them into hdf5 chunks (again #chunks = #cpus), total number of files =
   #geotiffs * #chunks. This uses the (saved) settings from the learning
   pipeline.
 - composefeats - takes the chunked hdf5s created in the previous step and
   concatenates and transforms them (whitening, imputation etc), total number
   of files = #chunks, these are our X's. This uses (saved) settings from the
   learning pipeline.
 - predictmodel - takes the learned pickled model, the composefeats outputs and
   predicts targets (for all X's)
 - exportgeotiff - creates a geotiff from all of the predict chunks All of
   these operations are distributed, so there is no single memory bottleneck.
