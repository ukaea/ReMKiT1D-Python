=========
Examples and Tutorials
=========

This repository contains many examples which have been built up over various ReMKiT1D versions. Some of these will be using deprecated features, or more streamlined approaches might have been developed in newer versions. This page will be updated with links to the most relevant external resources:

#. The `code paper <https://www.sciencedirect.com/science/article/pii/S0010465524001188>`_
#. The `2024 workshop repository <https://github.com/ukaea/ReMKiT1D-Workshop-2024>`_(this is based on v1.1.0)

The two notebooks rendered here can be considered as tutorials covering everything from setting up the config file to analyzing the resulting output data. Note that both of these are based on ReMKiT1D v1.0.x, and more recent examples can be found in the examples directory and the above resource list.

.. note:: 

    Note that to reproduce the results presented here ReMKiT1D must be run separately with the config.json files produced. See the Fortran repository documentation for how to run ReMKiT1D.

.. note:: 
    
    Unfortunately, the interactive dashboard couldn't be rendered in this example. The user is referred to the notebooks themselves, including the ReMKiT1D_analysis notebook, which has multiple different examples of how to quickly visualize data using the dashboard and holoviews and xarray features.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   advection
   custom_fluid