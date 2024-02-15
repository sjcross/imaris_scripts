import ImarisLib
import math
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

filaments_name = 'Filaments 5'
spots_name = 'Spots 5'

def get_selection_by_name(imaris_application, name):
    surpass_scene = imaris_application.GetSurpassScene()

    for i in range(0,surpass_scene.GetNumberOfChildren()):
        child = surpass_scene.GetChild(i)
        if child.GetName() == name:
            return child

    return None

def find_closest_filament_position(filaments_pos,spot_pos):
    min_dist = float("inf")
    closest_pos = None
    closest_pos_idx = None

    for i in range(0,len(filaments_pos)):
        filament_pos = filaments_pos[i]
        dist = get_dist(filament_pos,spot_pos)
        
        if dist < min_dist:
            min_dist = dist
            closest_pos = filament_pos
            closest_pos_idx = i

    return (closest_pos_idx,closest_pos,min_dist)

def build_matrix(edges, filaments_pos):
    row = []
    col = []
    distances = []
    for edge in edges:
        row.append(edge[0])
        col.append(edge[1])

        p1 = filaments_pos[edge[0]]
        p2 = filaments_pos[edge[1]]
        distances.append(get_dist(p1,p2))

    sz = len(filaments_pos)
    return csr_matrix((distances, (row, col)),shape=(sz,sz))

def get_dist(p1,p2):
    xx = p1[0]-p2[0]
    yy = p1[1]-p2[1]
    zz = p1[2]-p2[2]

    return math.sqrt((xx*xx)+(yy*yy)+(zz*zz))

def get_shortest_path(predecessors, start_pt, end_pt):
    predecessor = start_pt
    path = [predecessor]
    
    while predecessor != -9999:
        predecessor = predecessors[end_pt,predecessor]
        path.append(predecessor)

    return path

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Get Imaris instance
imaris_lib = ImarisLib.ImarisLib()
imaris_application = imaris_lib.GetApplication(0)
if imaris_application is None:
    print('Cant find Imaris')
    exit

factory = imaris_application.GetFactory()

# Get filament edges and positions
filament_selection = get_selection_by_name(imaris_application=imaris_application,name=filaments_name)
imaris_application.SetSurpassSelection(filament_selection)
filaments = factory.ToFilaments(imaris_application.GetSurpassSelection())
filaments_pos = filaments.GetPositionsXYZ(0)
filaments_edges = filaments.GetEdges(0)

# Get spot positions
spot_selection = get_selection_by_name(imaris_application=imaris_application,name=spots_name)
imaris_application.SetSurpassSelection(spot_selection)
spots = factory.ToSpots(imaris_application.GetSurpassSelection())
all_spots_pos = spots.GetPositionsXYZ()

# Finding closest filament position to each spot
max_dist = 0.01
closest_pos_idxs = []
spots_pos = []
for spot_pos in all_spots_pos:
    (closest_pos_idx,closest_pos,min_dist) = find_closest_filament_position(filaments_pos=filaments_pos,spot_pos=spot_pos)
    if min_dist <= max_dist:
        spots_pos.append(spot_pos)
        closest_pos_idxs.append(closest_pos_idx)
    
# Calculate distance matrix and predecessors for the spot positions
graph = build_matrix(edges=filaments_edges,filaments_pos=filaments_pos)
dist_mat, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True, indices=closest_pos_idxs)
spot_dist_mat = dist_mat[:,closest_pos_idxs]

# Removing self-referential values
for i in range(0,len(closest_pos_idxs)):
    spot_dist_mat[i,i] = float("inf")

# Extract shortest path between two points
for idx in range(0,len(closest_pos_idxs)):
    start_pt = closest_pos_idxs[idx]
    loc = np.argmin(spot_dist_mat[idx,:])
    end_pt = closest_pos_idxs[loc]
    path = get_shortest_path(predecessors=predecessors, start_pt=start_pt, end_pt=loc)
    
    fig = plt.figure()
    ax = Axes3D(fig)

    def init():
        for filament_edge in filaments_edges:
            xx = [filaments_pos[filament_edge[0]][0],filaments_pos[filament_edge[1]][0]]
            yy = [filaments_pos[filament_edge[0]][1],filaments_pos[filament_edge[1]][1]]
            zz = [filaments_pos[filament_edge[0]][2],filaments_pos[filament_edge[1]][2]]
            ax.plot(xx,yy,zz,'red')

        calc_path_length = 0
        for i in range(0,len(path)-2):
            p1 = filaments_pos[path[i]]
            p2 = filaments_pos[path[i+1]]
            calc_path_length = calc_path_length + get_dist(p1=p1,p2=p2)
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'green')

        # Showing straight line distance
        xx = [filaments_pos[start_pt][0],filaments_pos[end_pt][0]]
        yy = [filaments_pos[start_pt][1],filaments_pos[end_pt][1]]
        zz = [filaments_pos[start_pt][2],filaments_pos[end_pt][2]]
        ax.plot(xx,yy,zz,'blue')

        # Showing spots, their closest positions and the link between them
        for i in range(0,len(spots_pos)):
            # Plotting the spot position
            spot_pos = spots_pos[i]
            ax.scatter(spot_pos[0],spot_pos[1],spot_pos[2],marker='o',c='black')

            # Plotting the closest filament position
            closest_pos_idx = closest_pos_idxs[i]
            filament_pos = filaments_pos[closest_pos_idx]
            ax.scatter(filament_pos[0],filament_pos[1],filament_pos[2],marker='o',c='cyan')
            
            # Plotting the link between the two
            ax.plot([spot_pos[0],filament_pos[0]],[spot_pos[1],filament_pos[1]],[spot_pos[2],filament_pos[2]],'gray')

        ax.set_title('Dijkstra distance = '+str(dist_mat[loc,start_pt])+'\r\n'+'Calculated distance = '+str(calc_path_length))
        set_axes_equal(ax)

        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i*3)
        
        return fig,

    writer = PillowWriter(fps=5)
    anim = animation.FuncAnimation(fig, animate, init_func=init,frames=120, interval=20, blit=True)
    anim.save('ex'+str(idx)+'.gif', writer=writer)