import ImarisLib

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

vImarisLib = ImarisLib.ImarisLib()
vImarisApplication = vImarisLib.GetApplication(0)
vFactory = vImarisApplication.GetFactory()
vFilaments = vFactory.ToFilaments(vImarisApplication.GetSurpassSelection())

vPos = vFilaments.GetPositionsXYZ(0) # Gets all points on the filament as a list of lists
vEdges = vFilaments.GetEdges(0)

# Create animated gif showing plot
fig = plt.figure()
ax = Axes3D(fig)

def init():
    xx = []
    yy = []
    zz = []
    for row in vPos:
        xx.append(row[0])
        yy.append(row[1])
        zz.append(row[2])
    ax.scatter(xx, yy, zz, marker='o', s=20, c="goldenrod", alpha=0.6)

    for row in vEdges:
        xxx = [vPos[row[0]][0],vPos[row[1]][0]]
        yyy = [vPos[row[0]][1],vPos[row[1]][1]]
        zzz = [vPos[row[0]][2],vPos[row[1]][2]]
        ax.plot(xxx,yyy,zzz,'red')
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i*3)
    return fig,

writer = PillowWriter(fps=5)
anim = animation.FuncAnimation(fig, animate, init_func=init,frames=120, interval=20, blit=True)
anim.save('basic_animation2.gif', writer=writer)