import numpy as np
import matplotlib.pyplot as plt


x=np.linspace(0,10,500)
L=10
l1=.1
l2=1

d1=1.3*np.cos(2*np.pi*l1*x-np.pi/3.)
d2=4*np.cos(2*np.pi*l2*x-.9*np.pi)

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)

ax.set_xlim(-1,12)
ax.set_ylim(-10,10)

fig.patch.set_visible(False)
ax.axis('off')

ax.plot(x,d1, color='black')
ax.plot(x,d2, color='black', alpha=.5)

x=[1,5,5,1,1]
y=[-5,-5,5,5,-5]

ax.plot(x,y, lw=4, color='black')
ax.annotate('survey window', xy=(1.8,5.5), size=20 ) 

ax.annotate(r'$\delta_l$', xy=(x[0]-1.8,d1[0]), size=30 ) 
ax.annotate(r'$\delta_s$', xy=(x[0]-1.8,d2[0]), size=30 ) 


plt.show()
fig.savefig('split_delta.pdf')