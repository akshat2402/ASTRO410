import random as r
import math as m
from pylab import *
import matplotlib.pyplot as plt

imax = 100
x = 0.
y = 0.

# Start at origin
#graph1 = gdisplay(width=500, height=500, title='Random Walk', xtitle='x',ytitle='y')
#pts = gcurve(color = color.blue)
fig,ax = plt.subplots(1,1,figsize=(7,7))
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.plot([x],[y], marker='o', color='r')

for i in range(0, imax + 1):
    theta = 2.*m.pi*r.random()              # 0 =< angle =< 2 pi
    x1 = x + cos(theta)                                 # -1 =< x =< 1
    y1 = y + sin(theta)                                 # -1 =< y =< 1
    ax.plot([x,x1],[y,y1], marker='o', linestyle='-', color='k', markerfacecolor='r')
    x = x1
    y = y1    
    title("This walk's distance R = %8.4f"%m.sqrt(x*x + y*y))

    pause(0.1)

fig.savefig("randomWalk.jpg", dpi=300, bbox_inches="tight")
