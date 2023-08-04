.. DAML documentation master file, created by
   sphinx-quickstart on Fri Jul 21 18:20:16 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================
Data-Analysis Metrics Library
=============================

.. toctree::
   :hidden:

   ml_lifecycle
   usage
   api

----------
About DAML
----------

The Data-Analysis Metrics Library, or DAML, focuses on characterizing image data and its impact on model performance across classification and object-detection tasks.

----------

**Model-agnostic metrics that bound real-world performance**

* relevance/completeness/coverage
* metafeatures (data complexity)

----------

**Model-specific metrics that guide model selection and training**

* dataset sufficiency
* data/model complexity mismatch

----------

**Metrics for post-deployment monitoring of data with bounds on model performance to guide retraining**

* dataset-shift metrics
* model performance bounds under covariate shift
* guidance on sampling to assess model error and model retraining