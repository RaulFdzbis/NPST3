import numpy as np

## A noisy sine wave as query
idx = np.linspace(0,6.28,num=50)
query = np.sin(idx) + np.random.uniform(size=50)/10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formula
from dtw import *
alignment = dtw(query, template, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(query, template, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()

## And much more!

# How to align signals
list_index1 = alignment.index1.tolist()
list_index2 = alignment.index2.tolist()

# Plot aligned signals
dtw(query[list_index1], template[list_index2], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)
    
# Plot the resulting warped signal
plt.plot(alignment.index2, query[alignment.index1]) #Query se warpea para igualar los indices de la otra señal. Lo único malo es que no es una función valida. Hay puntos con más de un valor (están warpeados)
plt.plot(template)
plt.show()




