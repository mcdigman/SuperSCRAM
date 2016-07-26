import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)

ax.set_xlim(-2,2)
ax.set_ylim(-2,2)

fig.patch.set_visible(False)
ax.axis('off')

eps=.1
m=.7
x=np.linspace(-1,1,10)
y=x*m

ax.plot(x,y,lw=2,color='black')
ax.plot(x[0],y[0],'o', lw=4, color='black')
ax.plot(x[-1],y[-1],'o', lw=4, color='black')
ax.plot(np.array([0]),np.array([0]),'o', lw=4, color='black')

ax.annotate(r'$\mathcal{O}_I(-\epsilon)$', xy=(x[0]-eps,y[0]-2*eps), size=20)
ax.annotate(r'$\mathcal{O}_I(+\epsilon)$', xy=(x[-1],y[-1]+eps), size=20)
ax.annotate(r'$\mathcal{O}_I(0)$', xy=(0,0-2*eps), size=20)

ax.plot(x,np.zeros_like(x),'--', color='black')

ax.annotate(r'$\Theta^i(-\epsilon)$', xy=(x[0]-eps,0+eps), size=20)
ax.annotate(r'$\Theta^i(+\epsilon)$', xy=(x[-1],0+eps), size=20)


plt.show()
fig.savefig('finite_diff_O.pdf')