import numpy as np
import sys
import os
import numpy as np


import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe


import scipy.interpolate as interp
import matplotlib.patches as mpatches
from scipy.linalg import cholesky, sqrtm
from scipy.stats import multivariate_normal

# controller coding part
# Define the necessary functions
def spf_initialization(mean, nsig, cov):
    nx = len(mean)  # number of states
    nsp = 2 * nx + 1  # number of sigma points
    ensp = np.ones(nsp)  # vector of all ones

    # Generate weighting matrices
    Wi = 0.5 / nsig**2
    W0M = (nsig**2 - nx) / nsig**2
    W0C = (nsig**2 - nx) / nsig**2 + 3 - nsig**2 / nx
    WM = np.concatenate(([W0M], np.ones(2 * nx) * Wi))
    WC = np.concatenate(([W0C], np.ones(2 * nx) * Wi))

    # Initialize
    Psqrtm = nsig * cholesky(cov).T
    SigmaPts = np.column_stack((np.zeros((nx, 1)), -Psqrtm, Psqrtm))
    SigmaPts = SigmaPts + mean[:, np.newaxis] * ensp

    return SigmaPts, WM, WC, ensp

def get_crosstrack_error(x, y, path_ref):
    position = [x, y]
    A = [path_ref[0][0], path_ref[0][1]]
    B = [path_ref[1][0], path_ref[1][1]]
    crosstrack_error = ((B[0]-A[0])*(A[1]-position[1]) - (A[0]-position[0])*(B[1]-A[1])) / \
                        ((B[0]-A[0])**2 + (B[1]-A[1])**2)**0.5
    return crosstrack_error


def stanley(delta, Xk, path_ref, control_gains):
    L = 2.8  # length of car
    Kc = control_gains['Kc']
    Ksoft = control_gains['Ksoft']
    # Kvel = control_gains['Kvel']  # Uncomment if needed

    # Compute crosstrack error
    crosstrack_error = get_crosstrack_error(Xk[0], Xk[1], path_ref)

    # Calculate theta correction
    theta_correction = np.arctan2(Kc * crosstrack_error, Ksoft + Xk[2])

    # Compute steer angle
    steer_angle = delta + theta_correction

    # Normalize steer angle to the range [-pi, pi]
    steer_angle = (steer_angle + np.pi) % (2 * np.pi) - np.pi

    return steer_angle, crosstrack_error


def predict_states_spf(Xk, Uinput, dt):
    L = 2.8  # length of car
    steer = Uinput[0]
    accel = Uinput[1]

    steer_clipped = np.clip(steer, -50 * np.pi / 180, 50 * np.pi / 180)

    new_yaw = Xk[3] + (Xk[2] / L) * np.sin(steer_clipped) * dt
    new_x = Xk[0] + Xk[2] * np.cos(new_yaw + steer_clipped) * dt
    new_y = Xk[1] + Xk[2] * np.sin(new_yaw + steer_clipped) * dt
    new_vel = Xk[2] + dt * accel

    Xk1 = np.array([new_x, new_y, new_vel, new_yaw])
    return Xk1, steer_clipped

def calc_a_priori(pred_pts, WM, WC, Qx, ensp):
    # Calculate Mean (a priori)
    xPred = np.dot(pred_pts, WM)
    xhat_next = xPred

    # Calculate Covariance (a priori)
    exSigmaPts = pred_pts - np.outer(xPred, ensp)
    PxxPred = np.dot(exSigmaPts, np.diag(WC).dot(exSigmaPts.T)) + Qx
    Phat_next = PxxPred

    return xhat_next, Phat_next


def obtain_traj_samples(traj_ref,cov1):
    #npath=len(tmp_x)
    #V0 = np.linalg.norm(np.array([-226.22554494,498.52536952])-np.array([-226.22396974,498.51941811]))/0.1 #predicted velocity

    #traj_ref = np.column_stack((tmp_x1, tmp_y1))

    #V0=(v + a * self.dt)[0]
    #cov1=cov1.squeeze(0).numpy()[0]
    #print(cov1)
    cov1=[[cov1[0][0][0],cov1[0][0][1],0,0],
        [cov1[0][1][0],cov1[0][1][1],0,0],
        [0,0,0.4,0],
        [0,0,0,0.4]]
    traj_ref=traj_ref[0][0]
    V0 = np.linalg.norm(traj_ref[1, :] - traj_ref[0, :]) / 0.1
    len_traj = 20
    V_ref = np.ones(len_traj) * V0
    ttx = traj_ref[-1][0] - traj_ref[0][0]
    tty = traj_ref[-1][1] - traj_ref[0][1]
    T0 = np.arctan2(tty, ttx)
    T_ref = np.ones(len_traj) * T0
    X0 = np.array([traj_ref[0][0], traj_ref[0][1], V0, T0])
    Xtot = np.array([X0])
    deltatot = [0]
    steer_ideal = deltatot
    n_ref, _ = Xtot.shape
    '''
    # Prep for plotting
    Xmax, Xmin = np.max(Xtot[:, 0]), np.min(Xtot[:, 0])
    Ymax, Ymin = np.max(Xtot[:, 1]), np.min(Xtot[:, 1])
    '''
    #%% Controller gains (Stanley, Velocity)
    Kc_samples = np.arange(1, 31, 2)
    Ksoft_samples = np.arange(0.2, 3.2, 0.2)
    Kspeed_samples = np.arange(0.1, 4.1, 0.1)

    C, S, Sp = np.meshgrid(Kc_samples, Ksoft_samples, Kspeed_samples)
    combinations = np.vstack([C.ravel(), S.ravel(), Sp.ravel()]).T

    #%% Simulation setup

    dpath = np.sqrt(np.diff(traj_ref[:, 0])**2 + np.diff(traj_ref[:, 1])**2)
    tfinal = 2
    #print(tfinal)
    num_runs = 30

    # Set up time vector for simulation
    dt = 0.1
    tsim = np.arange(0, tfinal, dt)
    nsim = len(tsim)
    #print(nsim)
    dt_ref = dpath / V_ref[:-1]
    t_ref = np.concatenate(([0], np.cumsum(dt_ref)))

    nsig = 2
    ns = 4
    P0 = np.array(cov1)
    #print(P0.shape)
    xhat = np.zeros((ns, nsim))
    Phat = np.zeros((ns, ns, nsim))

    xhat[:, 0] = X0
    Phat[:, :, 0] = P0
    Q = np.diag([0.1**2, (2 * 50 * np.pi / 180)**2])

    # Number of elements in w
    nw = 2

    # Generate random samples
    w = np.dot(np.linalg.cholesky(Q), np.random.randn(nw, nsim))

    # Define matrix G
    G = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

    # Compute Qx
    Qx = np.dot(G, np.dot(Q, G.T))

    #%% Simulation run
    steer_hat = np.zeros((1, 2*ns+1))
    accel_hat = np.zeros((1, 2*ns+1))
    steer_clipped_sample = np.zeros((1, 2*ns+1))
    xPredSigmaPts = np.zeros((ns, 2*ns+1))
    crosstrack_sample = np.zeros((1, 2*ns+1))
    crosstrack_all = np.zeros((9, nsim))
    steer_clipped_all = np.zeros((9, nsim))

    steer_mean = np.zeros((1, nsim))
    steer_cov = np.zeros((1, nsim))
    accel_mean = np.zeros((1, nsim))
    accel_cov = np.zeros((1, nsim))
    flag=False
    control_gains = {'Kc': 6, 'Ksoft': 0.3}
    for s in range(1):
        control_gains['Kspeed'] = Kspeed_samples[s]

        for k in range(nsim - 1):
            # Placeholder for sigma points initialization function
            if np.isnan(xhat[:,k]).any() or np.isnan(Phat[:, :, k]).any():
                flag=True
                break
            xSigmaPts, WM, WC, ensp = spf_initialization(xhat[:, k], nsig, Phat[:, :, k])

            steer_hat, accel_hat = [], []
            for j in range(len(xSigmaPts.T)):
                distances = np.sqrt((traj_ref[:, 0] - xSigmaPts[0, j])**2 + (traj_ref[:, 1] - xSigmaPts[1, j])**2)
                nearest_index = np.argmin(distances)
                if nearest_index == 0:
                    path_ref = traj_ref[nearest_index:nearest_index+2, :]
                else:
                    path_ref = traj_ref[nearest_index-1:nearest_index+1, :]

                # Calculate angle difference between car and path
                angleCar = xSigmaPts[3, j]
                path_dy = path_ref[1, 1] - path_ref[0, 1]
                path_dx = path_ref[1, 0] - path_ref[0, 0]
                anglePath = np.arctan2(path_dy, path_dx)
                delta = anglePath - angleCar


                # Placeholder for Stanley controller function
                steer_angle, crosstrack_error = stanley(delta, xSigmaPts[:, j], path_ref, control_gains)
                steer_hat.append(steer_angle)

                Vref = V_ref[nearest_index]
                accel = control_gains['Kspeed'] * (Vref - xSigmaPts[3, j])
                accel_hat.append(accel)

                Uinput = [steer_angle, accel]
                # Placeholder for state prediction function
                Xk1, steer_clipped = predict_states_spf(xSigmaPts[:, j], Uinput, dt)

                xPredSigmaPts[:, j] = Xk1


            steer_hat = np.array(steer_hat)
            accel_hat = np.array(accel_hat)

            # Placeholder for a priori calculation function
            xhat_next, Phat_next = calc_a_priori(xPredSigmaPts, WM, WC, Qx, ensp)
            xhat[:, k + 1] = xhat_next
            Phat[:, :, k + 1] = Phat_next

            steer_mean_next, steer_cov_next = calc_a_priori(steer_hat, WM, WC, 0, ensp)
            steer_mean[:, k] = steer_mean_next
            steer_cov[:, k] = steer_cov_next

            accel_mean_next, accel_cov_next = calc_a_priori(accel_hat, WM, WC, 0, ensp)
            accel_mean[:, k] = accel_mean_next
            accel_cov[:, k] = accel_cov_next

    # Define the point

    # Define k
    if flag==True:
        traj_ref=traj_ref.tolist()
        return np.array([[traj_ref]]),0,0,None
    nll_res=[]
    for k in range(len(xhat[0,:])):
        #k = len(xhat[0,:])-1
        x1=xhat[0, :][k]
        y1=xhat[1, :][k]
        point=np.array([x1,y1])
        mean = xhat[0:2, k]
        covariance = Phat[0:2, 0:2, k]
        #print((2 * np.pi) ** 2 * np.linalg.det(covariance))
        #print(covariance)
        # Calculate the probability density function value
        #pdf_value = multivariate_normal.pdf(point, mean=mexan, cov=covariance)
        x_minus_mu=(point-mean)
        quad_form=x_minus_mu.T @ np.linalg.inv(covariance) @ x_minus_mu
        #print((x_minus_mu.T @ np.linalg.inv(covariance) @ x_minus_mu))
        #print((2 * np.pi) ** 2 * np.linalg.det(covariance))

        #print((x_minus_mu.T @ np.linalg.inv(covariance) @ x_minus_mu))
        # Calculate the negative log-likelihood
        nll_value = 0.5 * (np.log(2 * np.pi) + np.log(np.linalg.det(covariance)+1.0) +quad_form )
        nll_res.append(nll_value)
    traj_clc = np.column_stack((xhat[0,:], xhat[1,:])).tolist()
    traj_clc=np.array([[traj_clc]])
    return traj_clc,V0,nll_res,Phat

def prediction_output_to_trajectories(prediction_output_dict,
                                      dt,
                                      max_h,
                                      ph,
                                      sigma_matrix,
                                      map=None,
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()
    future_v=dict()
    ct=0
    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        #future_v[t]=dict()
        prediction_nodes = prediction_output_dict[t].keys()

        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
            position_state = {'position': ['x', 'y']}
            #v_state={'velocity': ['norm']}
            #v_statey={'velocity', 'y'): vy}
            #v_norm= {('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)}
            history = node.get(np.array([t - max_h, t]), position_state)  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]

            future = node.get(np.array([t + 1, t + ph]), position_state)
            future = future[~np.isnan(future.sum(axis=1))]

            if prune_ph_to_future:
                predictions_output = predictions_output[:, :, :future.shape[0]]
                if predictions_output.shape[2] == 0:
                    continue

            trajectory = predictions_output
            #future_v1 = node.get(np.array([t + 1, t + ph]), v_state)

            if map is None:
                histories_dict[t][node] = history
                #print(ct)
                sigma_matrix1=sigma_matrix[t][node]
                #size_s=sigma_matrix1.size()[1]
                #print(sigma_matrix.size())
                cov1=sigma_matrix1
                #print(cov1)
                res,_,_,_=obtain_traj_samples(trajectory,cov1)
                output_dict[t][node] = res

                futures_dict[t][node] = future

                #future_v[t][node]=future_v1
            else:
                histories_dict[t][node] = map.to_map_points(history)
                output_dict[t][node] = map.to_map_points(trajectory)
                futures_dict[t][node] = map.to_map_points(future)
            ct=ct+1
    return output_dict, histories_dict, futures_dict
