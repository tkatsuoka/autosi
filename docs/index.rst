:layout: landing

AutoSI
======

**Automatic selective inference for algorithms written as ordinary array
code.**

Write your selection algorithm with AutoSI's NumPy-like tracked
operations, run it once on the observed data, and call
:func:`autosi.inference` — you get an exactly valid selective *p*-value.
The selection event is derived automatically by recording, for every
comparison and selection the algorithm performs, the range of data on
which its outcome is unchanged. No hand-derivation is required, and
selection events of any polynomial degree (any algorithm expressible
through rational functions of the data) are supported — including a lasso
tuned by cross-validated R², which is beyond every existing exact method.

.. container:: buttons

   :doc:`Get Started <quickstart>`
   :doc:`Examples <examples>`
   :doc:`API Reference <api>`

.. raw:: html

   <div id="autosi-bg" aria-hidden="true"></div>

.. toctree::
   :hidden:

   quickstart
   examples
   api
