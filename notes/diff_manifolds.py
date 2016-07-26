from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

x1=np.arange(-100,-10,1)
y1=x1


X1,Y1=np.meshgrid(x1,y1)
Z1=X1*0 + Y1*0 

x2=np.arange(10,110,1,dtype=float)
y2=x2

sigma=20; mid=55; A=2
X2,Y2=np.meshgrid(x2,y2)
xx=X2-mid
yy=Y2-mid


Z2=A*np.exp(-(xx*xx + yy*yy)/(2*sigma**2))

fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
ax.set_zlim(-1,8)
fig.patch.set_visible(False)
ax.axis('off')

ax.plot_wireframe(X1, Y1, Z1, rstride=10, cstride=10 ,color='black')
ax.plot_wireframe(X2, Y2, Z2, rstride=10, cstride=10, color='black')
ax.text(10,-80,.5,r'$\delta(\mathbf{x}),\;\{\mathbf{x}, t\}$', size=20)
ax.text(80,-40,0,r'$\delta_{\rm{L}}(\mathbf{x}_{\rm{L}}),\;\{\mathbf{x}_{\rm{L}}, t_{\rm{L}}\}$', size=20)

plt.show()
