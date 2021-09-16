import numpy as np 
import networkx as nx


def labeling (int_mat):
    
    all_groups = []
    grp_type = np.unique(int_mat)
    grp_type = grp_type[grp_type>0]

    for j in grp_type :

        A = int_mat

        temp_mat = (A==j).astype(float)
        G = nx.from_numpy_matrix(temp_mat)
        groups = list(list(G.subgraph(c).nodes) for c in nx.connected_components(G))
        # print(groups)

        sub_groups = [j for j in groups if len(j)>1]
        # print('sub_groups: ', sub_groups)
        if len(sub_groups)>=1:

            all_groups.append(sub_groups[0])

    return all_groups


def dataset(grp_labels, pos_vect):

    labeling = grp_labels
    database = []

    for j in labeling:
        database.append(np.array(pos_vect[j, :, :], dtype=object))

    return database

def convert_gps(Box_yard,pos_vect):
    max_lon = Box_yard[1][0]
    min_lon = Box_yard[1][1]

    min_lat = Box_yard[0][0]
    max_lat = Box_yard[0][1]

    x_min = pos_vect[:,0,:].min()
    x_max = pos_vect[:,0,:].max()

    y_min = pos_vect[:,1,:].min()
    y_max = pos_vect[:,1,:].max()

    x = pos_vect[:,0,:]
    y = pos_vect[:,1,:]
    

    # A_new = (A_old-min(A))/(max(A)-min(A))*(max(new_range)-min(new_range))+min(new_range)
    new_x = ((x-x_min)/(x_max-x_min))*(max_lon-min_lon)+min_lon
    new_y = ((y-y_min)/(y_max-y_min))*(max_lat-min_lat)+min_lat


    new_pos_vect = np.zeros((pos_vect.shape[2], pos_vect.shape[0], pos_vect.shape[1]))

    new_pos_vect[:,:,0] = np.transpose(new_x, (1,0))
    new_pos_vect[:,:,1] = np.transpose(new_y, (1,0))

    return new_pos_vect

    
