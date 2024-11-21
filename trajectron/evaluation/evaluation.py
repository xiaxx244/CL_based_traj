import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from utils import prediction_output_to_trajectories
import visualization
from matplotlib import pyplot as plt

def nll_calculation(predicted_trajs,sigma_matrix,gt_trajs):
    #print(predicted_trajs)
    #print(gt_trajs)
    #print(sigma_matrix)
    #print(" ")
    #sigma_matrix=sigma_matrix[0]
    #print(gt_trajs.shape)
    #print(predicted_trajs[0].shape)
    nll_CL = np.zeros((len(gt_trajs), 1))
    for k in range(len(gt_trajs)):
        truth = gt_trajs[k]
        mu = predicted_trajs[0][0][k]
        Cov = sigma_matrix[k]
        #print(Cov)
        #print(" ")
        ''' 
        print(truth)
        print(mu)
        print(Cov)
        print(" ")
        '''
        quad_form = np.dot(np.dot((truth - mu).T, np.linalg.inv(Cov)), (truth - mu))
        nll_CL[k, 0] = 0.5 * (np.log(2 * np.pi) + np.log(np.linalg.det(Cov)) + quad_form)
    return sum(nll_CL[:,0])

def compute_ade1(predicted_trajs, ph,gt_traj):
    error=np.linalg.norm(np.reshape(predicted_trajs,(ph,2))[:gt_traj.shape[0]] - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    #print(ade.flatten())
    return ade.flatten()


def compute_fde1(predicted_trajs, gt_traj):
    '''
    if  np.reshape(predicted_trajs,(20,2)).shape!=gt_traj.shape:
        final_error = np.linalg.norm(predicted_trajs[:, :, -1] - np.append(gt_traj,[gt_traj[-1]],axis=0)[-1], axis=-1)
    else:
    '''
    #print(predicted_trajs[-1])
    #print(gt_traj[-1])
    #print(" ")
    final_error = np.linalg.norm(predicted_trajs[0][0][-1] - gt_traj[-1])
    return np.asarray([final_error])

def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


def compute_batch_statistics(prediction_output_dict,
                             dt,
                             max_hl,
                             ph,
                             sigma_matrix,
                             node_type_enum,
                             kde=True,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       sigma_matrix,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(),'nll':list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            '''
            print(prediction_dict[t][node])
            print(futures_dict[t][node])
            print(" ")
            '''
            #print(sigma_matrix.shape)
            ade_errors = compute_ade1(prediction_dict[t][node],ph, futures_dict[t][node])
            fde_errors = compute_fde1(prediction_dict[t][node], futures_dict[t][node])
            nll_errors = nll_calculation(prediction_dict[t][node], sigma_matrix[t][node],futures_dict[t][node])
            #print(fde_errors)
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].extend(list(ade_errors))
            batch_error_dict[node.type]['fde'].extend(list(fde_errors))
            batch_error_dict[node.type]['nll'].extend(list([nll_errors]))
            batch_error_dict[node.type]['kde'].extend([kde_ll])
            batch_error_dict[node.type]['obs_viols'].extend([obs_viols])

    return batch_error_dict


def log_batch_errors(batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                log_writer.add_histogram(f"{node_type.name}/{namespace}/{metric}", metric_batch_error, curr_iter)
                log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error), curr_iter)
                log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error), curr_iter)

                if metric in bar_plot:
                    pd = {'dataset': [namespace] * len(metric_batch_error),
                                  metric: metric_batch_error}
                    kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_barplots(ax, pd, 'dataset', metric)
                    log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_bar_plot", kde_barplot_fig, curr_iter)

                if metric in box_plot:
                    mse_fde_pd = {'dataset': [namespace] * len(metric_batch_error),
                                  metric: metric_batch_error}
                    fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', metric)
                    log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_box_plot", fig, curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error))
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error))
