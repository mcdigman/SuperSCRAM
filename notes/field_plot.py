import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)

ax.set_xlim(-1.3,1.3)
ax.set_ylim(-1.3,1.3)

fig.patch.set_visible(False)
ax.axis('off')

theta=np.linspace(0,2*np.pi,100)
x=np.cos(theta); y=np.sin(theta) 

ax.plot(x,y, lw=3, color='black')
ax.annotate(r'$R_{\rm max}$', xy=(-.01,1.1), size=20)

theta=np.linspace(np.pi/4., np.pi/2.,50)
A=.8
x=A*np.cos(theta); y=A*np.sin(theta)
ax.plot(x,y, lw=3, color='black', alpha=.5)
ax.plot([0,x[0]], [0,y[0]], lw=3, color='black', alpha=.5)
ax.plot([0,x[-1]], [0,y[-1]], lw=3, color='black', alpha=.5)
ax.annotate('survey 1', xy=(x[-1]-.02,y[-1]+.02), size=20)
ax.annotate(r'$z_4$', xy=(x[0]+.02,y[0]), size=20)

A=.5
x=A*np.cos(theta); y=A*np.sin(theta)
ax.plot(x,y, lw=3, color='black', alpha=.5)
ax.annotate(r'$z_3$', xy=(x[0]+.08,y[0]+.02), size=20)

A=.2
x=A*np.cos(theta); y=A*np.sin(theta)
ax.plot(x,y, lw=3, color='black', alpha=.5)
ax.annotate(r'$z_1$', xy=(x[0]+.08,y[0]+.02), size=20)

A=.35
x=A*np.cos(theta); y=A*np.sin(theta)
ax.plot(x,y, lw=3, color='black', alpha=.5)
ax.annotate(r'$z_2$', xy=(x[0]+.08,y[0]+.02), size=20)


theta=np.linspace(np.pi, 3/(2.6)*np.pi,50)
A=.8
x=A*np.cos(theta); y=A*np.sin(theta)
ax.plot(x,y, lw=3, color='black', alpha=.5)
ax.plot([0,x[0]], [0,y[0]], lw=3, color='black', alpha=.5)
ax.plot([0,x[-1]], [0,y[-1]], lw=3, color='black', alpha=.5)
ax.annotate('survey 2', xy=(x[-1]-.02,y[-1]-.08), size=20)
ax.annotate(r'$z_4$', xy=(x[0],y[0]+.05), size=20)

A=.5
x=A*np.cos(theta); y=A*np.sin(theta)
ax.plot(x,y, lw=3, color='black', alpha=.5)
ax.annotate(r'$z_3$', xy=(x[0],y[0]+.05), size=20)

A=.2
x=A*np.cos(theta); y=A*np.sin(theta)
ax.plot(x,y, lw=3, color='black', alpha=.5)
ax.annotate(r'$z_1$', xy=(x[0],y[0]+.05), size=20)

A=.35
x=A*np.cos(theta); y=A*np.sin(theta)
ax.plot(x,y, lw=3, color='black', alpha=.5)
ax.annotate(r'$z_2$', xy=(x[0],y[0]+.05), size=20)



plt.axes().set_aspect('equal')
plt.show()
fig.savefig('field.pdf')