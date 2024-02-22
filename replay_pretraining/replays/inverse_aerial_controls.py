import numpy as np
from numba import njit

T_r = -36.07956616966136  # torque coefficient for roll
T_p = -12.14599781908070  # torque coefficient for pitch
T_y = 8.91962804287785  # torque coefficient for yaw
D_r = -4.47166302201591  # drag coefficient for roll
D_p = -2.798194258050845  # drag coefficient for pitch
D_y = -1.886491900437232  # drag coefficient for yaw


# From RLUtilities
@njit
def quat_to_rot_mtx(quat: np.ndarray) -> np.ndarray:
    w = -quat[0]
    x = -quat[1]
    y = -quat[2]
    z = -quat[3]

    theta = np.zeros((3, 3), dtype=np.float32)

    norm = (quat * quat).sum()
    if norm != 0:
        s = 1.0 / norm

        # front direction
        theta[0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[1, 0] = 2.0 * s * (x * y + z * w)
        theta[2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[0, 1] = 2.0 * s * (x * y - z * w)
        theta[1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[0, 2] = 2.0 * s * (x * z + y * w)
        theta[1, 2] = 2.0 * s * (y * z - x * w)
        theta[2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

    return theta


@njit
def aerial_inputs(omega_start, omega_end, theta_start, dt):
    tau = (omega_end - omega_start) / dt  # net torque in world coordinates
    tst = np.transpose(theta_start)
    # tau1 = np.dot(tst, tau)  # net torque in local coordinates
    tau = np.array([
        tst[0, 0] * tau[0] + tst[0, 1] * tau[1] + tst[0, 2] * tau[2],
        tst[1, 0] * tau[0] + tst[1, 1] * tau[1] + tst[1, 2] * tau[2],
        tst[2, 0] * tau[0] + tst[2, 1] * tau[1] + tst[2, 2] * tau[2]
    ])
    # omega_local1 = np.dot(tst, omega_start)  # beginning-step angular velocity in local coordinates
    omega_local = np.array([
        tst[0, 0] * omega_start[0] + tst[0, 1] * omega_start[1] + tst[0, 2] * omega_start[2],
        tst[1, 0] * omega_start[0] + tst[1, 1] * omega_start[1] + tst[1, 2] * omega_start[2],
        tst[2, 0] * omega_start[0] + tst[2, 1] * omega_start[1] + tst[2, 2] * omega_start[2]
    ])

    # assert np.allclose(tau1, tau, equal_nan=True)
    # assert np.allclose(omega_local1, omega_local, equal_nan=True)

    rhs = np.array([
        tau[0] - D_r * omega_local[0],
        tau[1] - D_p * omega_local[1],
        tau[2] - D_y * omega_local[2]
    ])

    u = np.array([
        rhs[0] / T_r,  # roll
        rhs[1] / (T_p + np.sign(rhs[1]) * omega_local[1] * D_p),  # pitch
        rhs[2] / (T_y - np.sign(rhs[2]) * omega_local[2] * D_y)  # yaw
    ])

    # ensure values are between -1 and +1
    u = np.clip(u, -1, +1)

    return u[1], u[2], u[0]  # pitch, yaw, roll


def pyr_from_dataframe(deltas, player_df):
    is_repeated = (player_df[["pos_x", "pos_y", "pos_z",
                              "vel_x", "vel_y", "vel_z",
                              "ang_vel_x", "ang_vel_y", "ang_vel_z",
                              "quat_w", "quat_x", "quat_y", "quat_z"]].diff() == 0).all(axis=1).values
    ang_vels = player_df[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].values
    quats = player_df[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values
    pyrs = np.zeros((len(deltas), 3))
    deltas = deltas.values
    i = 0
    while i < len(deltas) - 1:
        omega_start = ang_vels[i]
        theta_start = quat_to_rot_mtx(quats[i])
        j = i
        delta = np.array(0, dtype=np.float32)
        while True:
            j += 1
            delta += deltas[j]
            if not is_repeated[j] or j >= len(deltas) - 1:
                break
        if delta <= 0:
            pyrs[i] = np.nan
        else:
            omega_end = ang_vels[j]
            pyrs[i:j] = aerial_inputs(omega_start, omega_end, theta_start, delta)
        i = j
    return pyrs
