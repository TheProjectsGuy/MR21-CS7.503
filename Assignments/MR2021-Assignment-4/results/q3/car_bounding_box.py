# %% Import everything
import sympy as sp
import numpy as np

# %% Symbols
# -- Known parameters (as floats) --
vl_val, vw_val, vh_val = 4.10, 1.51, 1.38   # L, W, H in m
vth_val = np.deg2rad(5) # Angle (in rad)
ch_val = 1.65   # Cam height in m
vcx_val, vcy_val = 858, 240 # Camera pixel of vehicle center
K_val = [   # Camera intrinsic parameter matrix
    [7.2153e+02,0,6.0955e+02],
    [0,7.2153e+02,1.7285e+02],
    [0,0,1]]
K_np = np.array(K_val, float)   # As numpy
# - The above will only be used in the end -

# -- Known Parameters (as symbols) --
# Vehicle properties
vl, vw, vh = sp.symbols(r"V_l, V_w, V_h")   # dimensions (L, W, H)
vth = sp.symbols(r"V_\theta")   # Z rotation for vehicle (in rad)
# Camera properties
ch = sp.symbols(r"C_h") # Camera height (from ground)
# Camera projection matrix
K_11, K_12, K_13 = sp.symbols(r"k_{11}, k_{12}, k_{13}")
K_22, K_23, K_33 = sp.symbols(r"k_{22}, k_{23}, k_{33}")
K_sp = sp.Matrix([[K_11, K_12, K_13], [0, K_22, K_23], [0, 0, K_33]])
vcx, vcy = sp.symbols(r"V'_{c_x}, V'_{c_y}")    # Pixel of car center

# -- Unknown parameters --
# Vehicle parameters
vx, vy = sp.symbols(r"V_x, V_y")    # Vehicle X and Y from {world}

# %% Prior to main work
# Image point (homogeneous coordinates)
vimg = sp.Matrix([vcx, vcy, 1])
# -- Homogeneous Transformations --
# - TF {vehicle} in {world} -
# Rotation for {vehicle} in {world}
R_w_v = sp.Matrix([ # Rot(Z, vth)
    [sp.cos(vth), -sp.sin(vth), 0],
    [sp.sin(vth), sp.cos(vth), 0],
    [0, 0, 1]])
# Vehicle origin (in {world} - homogeneous coordinates)
vorg_w = sp.Matrix([vx, vy, 0, 1])
# Homogeneous Transformation matrix ({vehicle} in {world})
tf_w_v = sp.Matrix.hstack(  # Stacking R_w_v and vorg_w
    sp.Matrix.vstack(R_w_v, sp.Matrix([[0, 0, 0]])), vorg_w)
# - TF {camera} in {world} -
# Rotation from world to camera
R_w_c = sp.Matrix([ # Z out of cam, Y down, X to right
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]])
# Camera origin (in {world} - homogeneous coordinates)
corg_w = sp.Matrix([0, 0, ch, 1])
# Homogeneous Transformation matrix ({camera} in {world})
tf_w_c = sp.Matrix.hstack(  # Stacking R_w_v and vorg_w
    sp.Matrix.vstack(R_w_c, sp.Matrix([[0, 0, 0]])), corg_w)
# - TF {world} in {camera} -
tf_c_w = sp.Matrix.hstack(
    sp.Matrix.vstack(R_w_c.T, sp.Matrix([[0, 0, 0]])), 
    sp.Matrix.vstack(
        -R_w_c.T * sp.Matrix(corg_w[0:3]), sp.Matrix([[1]]))
    )   # Invert the transformation matrix

# %% Equation for resolving points
# Vehicle center in {vehicle}
vc_v = sp.Matrix([vl/2, vw/2, vh/2, 1])
# Vehicle center in {world}
vc_w = tf_w_v * vc_v
# Vehicle center in {camera}
vc_c = tf_c_w * vc_w

# %% Camera projection equations
lhs_eq = K_sp.inv() * vimg  # Image projected to the world
rhs_eq = sp.Matrix([    # Vehicle center in camera frame [X;Y;Z]
    [vc_c[0]/vc_c[3]],
    [vc_c[1]/vc_c[3]],
    [vc_c[2]/vc_c[3]]])
# The last value of LHS is 1 (projection), set the same of for RHS
rhs_eqn = rhs_eq / rhs_eq[2] # Last value corresponds
lhs_eqn = lhs_eq / lhs_eq[2] # Last value corresponds
eq_s = sp.Eq(lhs_eqn, rhs_eqn)    # Equality to solve
sols = sp.solvers.solve(eq_s, [vx, vy]) # Solutions to the equality
vx_sol = sols[vx]
vy_sol = sols[vy]

# %% Solution for vehicle positions
# Substitution values
val_subs = {
    vl: vl_val,
    vw: vw_val,
    vh: vh_val,
    ch: ch_val,
    vth: vth_val,
    vcx: vcx_val,
    vcy: vcy_val,
    K_11: K_np[0, 0], K_12: K_np[0, 1], K_13: K_np[0, 2],
    K_22: K_np[1, 1], K_23: K_np[1, 2], K_33: K_np[2, 2]
}
vx_res = float(vx_sol.subs(val_subs))
vy_res = float(vy_sol.subs(val_subs))
print(f"Vehicle BRD at (X, Y): {vx_res:.4f}, {vy_res:.4f}")

# %%
