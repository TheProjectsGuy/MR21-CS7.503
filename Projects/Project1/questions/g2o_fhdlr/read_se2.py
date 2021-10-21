"""
    Python functions that help interact with a `.g2o` file that is
    written for SE(2) points: (x, y, theta) format. This file contains
    functions to read from such a file
"""

# %% Import everything
import numpy as np

# %% Read vertices
def read_vertex(file_name, sort_i=False):
    """
    Reads vertices described in the `.g2o` file passed. Note that the
    vertex is defined in the format

    VERTEX_SE2 i X Y Theta

    Where 'i' is the index and 'X', 'Y' and 'Theta' are SE(2) pose.

    Parameters:
    - file_name: str
        Full file name (with '.g2o' extension) with relative path
    - sort_i: bool      default: False
        Sort the indices. By default, the order of 'i' is as they
        appear in file (ideally should be continuously ascending).
        This argument, if set to 'True', sorts the final vector in 
        ascending order of 'i' before returning 'vtxs'.
    
    Returns:
    - vtxs: np.ndarray      shape: (n, 4)
        All vertices. 'n' is the number of vertices in the file. Each
        vertex is (i, x, y, th) as described in the format above. The
        order is decided through 'sort_i'
    """
    # Open file
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    # Get vertices
    vtxs = []
    for line in lines:
        lc = line.split()
        if lc[0].upper() == "VERTEX_SE2":   # Vertex
            i = int(lc[1])
            x, y, th = map(float, lc[2:])
            vtxs.append([i, x, y, th])
    vtxs = np.array(vtxs)
    if sort_i:
        vint = vtxs[:,0].astype(int)
        vtxs = vtxs[np.argsort(vint)]    # Sort by 'i'
    return vtxs

# %% Read edges (odometry and loop closure)
def read_edges_olc(file_name, rmat=False):
    """
    Reads edges described in the `.g2o` file passed. Note that the
    edge is described in the format

    EDGE_SE2 i j x y th s11 s12 s13 s22 s23 s33

    Where 'i' and 'j' are the vertices. If 'j' = 'i' + 1, the given
    edge is `odometry`, else it is `loop closure`. The frame 
    transformation ('j' in 'i') is given by ('x', 'y', 'th'). The
    values of 's' are 's[row][col]' in the information matrix (which
    is the inverse of covariance matrix b/w variables 'x', 'y' and 
    'th').

    Parameters:
    - file_name: str
        Full file name (with '.g2o' extension) with relative path
    - rmat: bool        default: False
        Return transformation and covariance inverse as numpy arrays.
        The transformation is converted into an SE2 homogeneous tf
        matrix, the covariance inverse (information matrix) is 
        converted into a 3x3 matrix.

    Returns:
    - odom_edges: list[[i, j], tf, ci]: N1
    - lc_edges: list[[i, j], tf, ci]: N2
        N1 is number of odometry edges, N2 is the number of loop
        closure constraints. 'tf' and 'ci' are either direct values
        (when 'rmat' is False) or numpy arrays (when 'rmat' is True).
        When 'rmat' is True, 'tf' is 3x3 ndarray and 'ci' is 3x3
        covariance inverse.
    """
    # Open file
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    # Get edges
    odom_edges = []
    lc_edges = []
    for line in lines:
        lc = line.split()
        if lc[0].upper() == "EDGE_SE2": # Edge
            i, j = map(int, lc[1:3])
            x, y, th = map(float, lc[3:6])  # tf_i_j
            s11, s12, s13, s22, s23, s33 = map(float, lc[6:12])
            # All that's to be saved
            pts = [i, j]
            if not rmat:
                tf = [x, y, th]
                ci = [s11, s12, s13, s22, s23, s33]
            else:
                tf = np.array([
                    [np.cos(th), -np.sin(th), x],
                    [np.sin(th), np.cos(th), y],
                    [0, 0, 1]
                ])
                ci = np.array([
                    [s11, s12, s13],
                    [s12, s22, s23],
                    [s13, s23, s33]
                ])
            record = [pts, tf, ci]  # Record that's saved
            if j == i + 1:  # Odometry edge
                odom_edges.append(record)
            else:           # Loop closure
                lc_edges.append(record)
    # Return the arrays
    return odom_edges, lc_edges

# %%
