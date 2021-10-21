"""
    Python functions that help interact with a `.g2o` file that is
    written for SE(2) points: (x, y, theta) format. This file contains
    functions to write to such a file
"""

# %% Import everything
import numpy as np

# %% Write vertices
def write_vertex(file_name, verts):
    """
    Writes the vertices (states) from variable 'verts' to the file
    in 'file_name' (full name, with path and extension). Note that the
    vertex is defined in the format

    VERTEX_SE2 i X Y Theta

    Where 'i' is the index and 'X', 'Y' and 'Theta' are SE(2) pose.
    The value for 'i' could be passed embedded in 'verts' or is
    implicitly assigned.

    Parameters:
    - file_name: str
        Full file name (with '.g2o' extension) with relative path
    - 
    """
    # Lines to write
    lines = []
    for i, vert in enumerate(verts):
        ls = ["VERTEX_SE2"]
        if vert.shape[0] == 3:  # No index given
            ls += [str(i)]
            ls += list(map(str, vert))
        elif vert.shape[0] == 4:    # First value is index
            ls += [str(int(vert[0]))]
            ls += list(map(str, vert[1:]))
        line = " ".join(ls)
        lines.append(line)
    # Write lines into the file
    with open(file_name, 'w') as fhdlr:
        for line in lines:
            fhdlr.write(line)
            fhdlr.write("\n")

# %%
