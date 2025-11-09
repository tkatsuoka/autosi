import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import autosi as si

a = si.siarray(np.array([1, 2, 3]))

b = si.RationalFunction(
    numerator=[np.array([1, 2, 3])],
    denominator=[np.array([1, 2, 3])],
)

si.IntervalTracker().get_instance().set_z(1.0)

c = a @ b

print(c.data)

a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
c = a @ b
print(c)
