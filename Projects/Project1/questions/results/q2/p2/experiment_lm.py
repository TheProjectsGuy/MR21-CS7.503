# %% Function to insert after declaring everything
def run_lm_experiment(x_mea, cert_o, cert_lc, cert_zero):
    # Parameters
    lm_lb = 0.01    # Damping parameter (lambda) for Levenberg algo
    a, b = 2.5, 1.05    # Parameters for damping factor
    num_iter = 50   # Number of iterations
    zero_pos = x_mea[0]     # Zero position
    # Variables for optimization
    xc = np.copy(x_mea)  # Optimization vector
    err_hist = []
    lmlb_hist = []
    error_start = error_value(xc, odom_c, lc_c, cert_o, cert_lc, 
        cert_zero, zero_pos)
    err_hist.append(error_start)
    # Levenberg algorithm
    i = 0
    j, fs = 0, num_iter * 3 # Failsafe (to avoid infinite loop)
    while i < num_iter:
        lmlb_hist.append(lm_lb) # Value of lambda
        # Failsafe
        j += 1
        if j == fs-1:
            print("Broke failsafe, check 'lm_lb'")
            break
        # Jacobian value
        J = jacobian_mat(xc, odom_c, lc_c)
        omega_mat = certainty_matrix(odom_c, lc_c, cert_o, cert_lc, 
            cert_zero)
        jtoj = J.T @ omega_mat @ J
        im_d = np.diag(np.diagonal(jtoj))
        jtoj_inv = np.linalg.inv(jtoj + lm_lb * im_d)
        rv = residual_vector(xc, odom_c, lc_c, zero_pos)
        dx_vect = -jtoj_inv @ J.T @ omega_mat.T @ rv
        xc_hold = xc + dx_vect.reshape(-1, 3)    # Don't apply update
        # Error value
        err = error_value(xc_hold, odom_c, lc_c, cert_o, cert_lc, 
            cert_zero, zero_pos)
        if err_hist[-1] - err > 10: # Successful update
            xc = xc_hold    # Apply the update
            lm_lb *= (1/a)
        else:   # Increase damping
            lm_lb *= b
            continue    # Retry update with new damping
        err_hist.append(err)
        # Increment i
        i += 1
    err_hist = np.array(err_hist)
    lmlb_hist = np.array(lmlb_hist)
    return err_hist, lmlb_hist, xc

# %% Actual test code
# Analysis to log
run_on = [
    [800, 100, 1000],
    [600, 600, 1000], 
    [500, 700, 1000], 
    [500, 700, 2000], 
    [400, 800, 2000],
    [200, 1000, 2000],
    [50, 1000, 2000],
    [20, 1100, 2000]
]
data_analysis = []  # err_h, lm_h, xv
for i, (co, clc, cz) in enumerate(run_on):
    print(f"Running: {i}")
    err_h, lm_h, xv = run_lm_experiment(x_mea, co, clc, cz)
    print(f"Error goes from: {err_h[0]} to {err_h[-1]}")
    data_analysis.append([err_h, lm_h, xv])

# %% Analysis
plt.figure(1, (10, 8))
draw(data_analysis[6][2], "50")
draw(data_analysis[7][2], "<40")
draw(x_gt, "Ground Truth")
plt.legend()
plt.show()
