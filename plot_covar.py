import numpy as np
import matplotlib.pyplot as plt

d1=np.loadtxt('covar_SCC_on.dat')
d2=np.loadtxt('covar_SCC_0.dat')

ax=plt.subplot(131)


x=ax.matshow(d1)
plt.colorbar(x)

ax=plt.subplot(132)



x=ax.matshow(d2)
plt.colorbar(x)


ax=plt.subplot(133)

x=ax.matshow((d2-d1)/d1)
#plt.colorbar(x)

b=d2/d1
print np.min(b), np.max(b), np.std(b)

plt.show()