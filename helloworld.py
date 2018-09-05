import sympy as sp
sp.init_printing(use_latex='mathjax')

x, y, z = sp.symbols('x y z')
f = sp.sin(x * y) + sp.cos(y * z)
sp.integrate(f, x)
from IPython.display import Image
Image('http://jakevdp.github.com/figures/xkcd_version.png')
import numpy as np
import pandas as pd


df = pd.DataFrame({'A': 1.,
                   'B': pd.Timestamp('20130102'),
                   'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                   'D': np.array([3] * 4, dtype='int32'),
                   'E': pd.Categorical(["test", "train", "test", "train"]),
                   'F': 'foo'})

df
import matplotlib.pyplot as plt
import numpy as np
plt.plot[t, np.sin(y)]
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
t = np.linspace(0, 20, 500)

plt.plot(t, np.sin(t))
plt.show()
