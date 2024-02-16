# Nearest neighbour analysis for spots associated with filaments
#
#Written by Stephen Cross
#
#Feb 2024 - Imaris 7.6.5

# <CustomTools>
#     <Menu>
#         <Submenu name="Spots Functions">
#             <Item name="NN distance along filaments" icon="Python">
#                 <Command>PythonXT::nn_distance(%i)</Command>
#             </Item>
#         </Submenu>
#     </Menu>
# </CustomTools>
#
#Description:


import ImarisLib
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

from matplotlib import animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

def nn_distance(imaris_id):
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

    print("Initialising analysis")

    # Get Imaris instance
    imaris_lib = ImarisLib.ImarisLib()
    imaris_application = imaris_lib.GetApplication(imaris_id)
    if imaris_application is None:
        print('Imaris could not be found')
        quit

    factory = imaris_application.GetFactory()

    # Get selections
    if sys.version_info < (3, 0):
        filaments_name = raw_input("Enter filaments name: ")
        spots_name = raw_input("Enter spots name: ")
    else:
        filaments_name = input("Enter filaments name: ")
        spots_name = input("Enter spots name: ")

    print("Collecting objects")

    # Get filaments
    filament_selection = get_selection_by_name(imaris_application=imaris_application,name=filaments_name)
    imaris_application.SetSurpassSelection(filament_selection)
    filaments = factory.ToFilaments(imaris_application.GetSurpassSelection())
    num_filaments = filaments.GetNumberOfFilaments()

    # Get all spots and their positions
    spot_selection = get_selection_by_name(imaris_application=imaris_application,name=spots_name)
    imaris_application.SetSurpassSelection(spot_selection)
    all_spots = factory.ToSpots(imaris_application.GetSurpassSelection())
    all_spots_pos = all_spots.GetPositionsXYZ()

    # Gathering all the filament positions into a single list
    all_filaments_pos = []
    for filament_idx in range(0,num_filaments):
        for curr_filaments_pos in filaments.GetPositionsXYZ(filament_idx):
            all_filaments_pos.append(curr_filaments_pos)

    print("Linking spots to filaments")

    # Determining the closest each spot is to any filament.  
    # Later, we will only use a spot if the distance to the current filament is equal 
    # to its minimum distance (i.e. this is the filament it was closest to).
    all_min_dists = []
    for spot_pos in tqdm (all_spots_pos, desc="Linking..."):
        (closest_pos_idx,closest_pos,min_dist) = find_closest_filament_position(filaments_pos=all_filaments_pos,spot_pos=spot_pos)
        all_min_dists.append(min_dist)

    colours = ['orange','blue','gray']

    # Initialising output statistics
    out_mNames = [None]*len(all_spots_pos)
    out_mValues = [None]*len(all_spots_pos)
    out_mUnits = [None]*len(all_spots_pos)
    out_mFactorNames = ['Category']
    out_mFactors = [[None]*len(all_spots_pos)]
    out_mIds = [None]*len(all_spots_pos)

    # Iterating over all filaments
    for filament_idx in tqdm (range(0,num_filaments), desc="Processing..."):
        filaments_pos = filaments.GetPositionsXYZ(filament_idx)
        filaments_edges = filaments.GetEdges(filament_idx)

        # Finding the spots closest to this filament and identifying their closest filament position 
        closest_pos_idxs = []
        spots_pos = []
        spots_idx = []
        for spot_idx in range(0,len(all_spots_pos)):
            spot_pos = all_spots_pos[spot_idx]
            (closest_pos_idx,closest_pos,min_dist) = find_closest_filament_position(filaments_pos=filaments_pos,spot_pos=spot_pos)

            if min_dist == all_min_dists[spot_idx]:
                spots_pos.append(spot_pos)
                closest_pos_idxs.append(closest_pos_idx)
                spots_idx.append(spot_idx)

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
            end_idx = np.argmin(spot_dist_mat[idx,:])
            end_pt = closest_pos_idxs[end_idx]
            path = get_shortest_path(predecessors=predecessors, start_pt=start_pt, end_pt=end_idx)

            out_mNames[spots_idx[idx]] = 'NN_distance'
            out_mValues[spots_idx[idx]] = dist_mat[end_idx,start_pt]
            out_mUnits[spots_idx[idx]] = 'um'
            out_mFactors[0][spots_idx[idx]] = 'Spot'
            out_mIds[spots_idx[idx]] = spots_idx[idx]

            # fig = plt.figure()
            # ax = Axes3D(fig)

            # def init():
            #     # Displaying all filaments
            #     for plot_filament_idx in range(0,num_filaments):
            #         plot_filaments_pos = filaments.GetPositionsXYZ(plot_filament_idx)
            #         plot_filaments_edges = filaments.GetEdges(plot_filament_idx)
            #         for filament_edge in plot_filaments_edges:
            #             xx = [plot_filaments_pos[filament_edge[0]][0],plot_filaments_pos[filament_edge[1]][0]]
            #             yy = [plot_filaments_pos[filament_edge[0]][1],plot_filaments_pos[filament_edge[1]][1]]
            #             zz = [plot_filaments_pos[filament_edge[0]][2],plot_filaments_pos[filament_edge[1]][2]]
            #             ax.plot(xx,yy,zz,colours[plot_filament_idx])

            #     calc_path_length = 0
            #     for i in range(0,len(path)-2):
            #         p1 = filaments_pos[path[i]]
            #         p2 = filaments_pos[path[i+1]]
            #         calc_path_length = calc_path_length + get_dist(p1=p1,p2=p2)
            #         ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'green')

            #     # Showing spots, their closest positions and the link between them
            #     for i in range(0,len(spots_pos)):
            #         # Plotting the spot position
            #         spot_pos = spots_pos[i]
            #         ax.scatter(spot_pos[0],spot_pos[1],spot_pos[2],marker='o',c='black')

            #         # Plotting the closest filament position
            #         closest_pos_idx = closest_pos_idxs[i]
            #         filament_pos = filaments_pos[closest_pos_idx]
            #         ax.scatter(filament_pos[0],filament_pos[1],filament_pos[2],marker='o',c='cyan')
                    
            #         # Plotting the link between the two
            #         ax.plot([spot_pos[0],filament_pos[0]],[spot_pos[1],filament_pos[1]],[spot_pos[2],filament_pos[2]],'gray')

            #     ax.scatter(filaments_pos[start_pt][0],filaments_pos[start_pt][1],filaments_pos[start_pt][2],marker='o',c='green')
            #     ax.scatter(filaments_pos[end_pt][0],filaments_pos[end_pt][1],filaments_pos[end_pt][2],marker='o',c='red')
                
            #     ax.set_title('Dijkstra distance = '+str(dist_mat[end_idx,start_pt])+'\r\n'+'Calculated distance = '+str(calc_path_length))
            #     set_axes_equal(ax)

            #     return fig,

            # def animate(i):
            #     ax.view_init(elev=20., azim=i*6)
                
            #     return fig,

            # writer = PillowWriter(fps=5)
            # anim = animation.FuncAnimation(fig, animate, init_func=init,frames=30, interval=20, blit=True)
            # anim.save(filaments_name+' ('+str(filament_idx)+')_'+spots_name+' ('+str(idx)+').gif', writer=writer)

    all_spots.AddStatistics(out_mNames, out_mValues, out_mUnits, out_mFactors, out_mFactorNames, out_mIds)

nn_distance(0)