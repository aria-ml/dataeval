.. DAML documentation master file, created by
   sphinx-quickstart on Fri Jul 21 18:20:16 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===============================
Welcome to DAML's Documentation
===============================

DAML, the *Data Analysis and Monitoring Library*, is an open-source toolkit that focuses on characterizing image data and its impact on model performance across classification and object-detection tasks.

---------------------------------------
:ref:`Installation<installation_guide>`
---------------------------------------

If DAML is not installed, follow this easy :ref:`step-by-step guide<installation_guide>`

--------------------------------------
:ref:`Quickstart Tutorials<tutorials>`
--------------------------------------

We are proud of our tools, so we highlighted common workflows with links so you can try them yourself!

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:doc:`Bayes Error Rate Tutorial<tutorials/notebooks/BayesErrorRateEstimationTutorial>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. We want to show visualizations of tutorials to peak the interest of a potential user
   Might be good to add a BER graph that a user would need (not necessarily from tutorial)
   i.e. A Graph with training accuracy curve, and a BER line (similar to sufficiency)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:doc:`Outlier Detection Tutorial<tutorials/notebooks/OutlierDetectionTutorial>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. We want to show visualizations of tutorials to peak the interest of a potential user
   We could show 3 images from a training set class next to 1 that is an Outlier but classified the same
   Could even make a few rows (multiple classes). 

DAML is a powerful toolkit for any data analysis workflow, so be sure to check out the **Quickstart Tutorials** page for a more comprehensive list of all the tools we offer.

-----------------------
:ref:`How To's<how-to>`
-----------------------

For the more experienced user, or if you are just curious, these guides show different ways that DAML's features can be used that might fit operational use more closely

---------------------------
:ref:`Reference<reference>`
---------------------------

Looking for a specific function or class? This reference guide has all the technical details needed to understand the DAML Ecosystem

-------------------
:ref:`About<about>`
-------------------

For more information about why DAML exists, this page gives an overview to DAML's purpose in ML

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   tutorials/index
   how_to/index   
   reference/index
   about
