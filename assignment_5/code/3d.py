import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,1.5,0.001)

plt.ylim(0,10.5)
plt.xlim(0,1.05)
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,11,1))
plt.vlines(0.5,0,2,color='red',linestyle='dotted')
plt.hlines(1,0,0.6,color='red',linestyle='dotted')
plt.xlabel("Probability of error")
plt.ylabel("Weight multiplier")
plt.plot(x,[(1-i)/i for i in x])
plt.show()
