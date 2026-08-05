API Reference
=============

.. currentmodule:: autosi

Inference
---------

.. autosummary::
   :toctree: generated/

   inference
   inference_chi
   NoHypothesisError

Tracked array
-------------

.. autosummary::
   :toctree: generated/

   siarray
   array

Selection operations (tracked)
------------------------------

These record the comparisons that fix their outcome as constraints on the
truncation region.

.. autosummary::
   :toctree: generated/

   abs
   max
   min
   argmax
   argmin
   sort
   argsort

Arithmetic reductions and layout (untracked)
--------------------------------------------

.. autosummary::
   :toctree: generated/

   sum
   mean
   var
   prod
   flatten
   stack
   concatenate

Advanced
--------

.. autosummary::
   :toctree: generated/

   set_memoization
   reset_memoization
   IntervalTracker
