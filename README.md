# Close loop control with trajectory forecasting #

## Installation ##

### Cloning ###
When cloning this repository, make sure you clone the submodules as well, with the following command:
```
git clone --recurse-submodules <repository cloning URL>
```
Alternatively, you can clone the repository as normal and then load submodules later with:
```
git submodule init # Initializing our local configuration file
git submodule update # Fetching all of the data from the submodules at the specified commits
```

**NOTE:** If you would like to use the code as it was for ECCV 2020, please check out the `eccv2020` branch. The `master` branch will differ as new updates are made to the codebase (including potential non-interoperabilities between the two versions).

### Environment Setup ###
First, we'll create a conda environment to hold the dependencies.
```
conda create --name trajectron++ python=3.6 -y
source activate trajectron++
pip install -r requirements.txt
```

Then, since this project uses IPython notebooks, we'll install this conda environment as a kernel.
```
python -m ipykernel install --user --name trajectronpp --display-name "Python 3.6 (Trajectron++)"
```

### Data Setup ###
#### Ithaca365 Datasets ####
We've already included preprocessed data splits for the Ithaca365 datasets in the Graphite, you can see them in `/home/yx454/original_traj/ablation_study/Trajectron-plus-plus/experiments/processed_ithaca_new_track/`. 

#### nuScenes Dataset ####
Download the nuScenes dataset (this requires signing up on [their website](https://www.nuscenes.org/)). Note that the full dataset is very large, so if you only wish to test out the codebase and model then you can just download the nuScenes "mini" dataset which only requires around 4 GB of space. Extract the downloaded zip file's contents and place them in the `experiments/nuScenes` directory. Then, download the map expansion pack (v1.1) and copy the contents of the extracted `maps` folder into the `experiments/nuScenes/v1.0-mini/maps` folder. Finally, process them into a data format that our model can work with.
```
cd experiments/nuScenes

# For the mini nuScenes dataset, use the following
python process_data.py --data=./v1.0-mini --version="v1.0-mini" --output_path=../processed

# For the full nuScenes dataset, use the following
python process_data.py --data=./v1.0 --version="v1.0-trainval" --output_path=../processed
```
In case you also want a validation set generated (by default this will just produce the training and test sets), replace line 406 in `process_data.py` with:
```
    val_scene_names = val_scenes
```

## Model Training ##
### Ithaca365 Dataset ###
To train a model on the Ithaca365 datasets, you can execute a version of the following command from within the `trajectron/` directory.
```
python train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScenes/models/int_ee/config.json --train_data_dict ithaca365_train_youya.pkl --eval_data_dict ithaca365_test_youya.pkl --offline_scene_graph yes --preprocess_workers 10 --batch_size 64 --log_dir <directory to log the trained model>  --train_epochs 20 --data_dir /home/yx454/original_traj/ablation_study/Trajectron-plus-plus/experiments/processed_ithaca_new_track --node_freq_mult_train --log_tag _int_ee --augment

```


### nuScenes Dataset ###
To train a model on the nuScenes dataset, you can execute one of the following commands from within the `trajectron/` directory, depending on the model version you desire.

| Model                                     | Command                                                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Base                                      | `python train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScenes/models/vel_ee/config.json --train_data_dict nuScenes_train_full.pkl --eval_data_dict nuScenes_val_full.pkl --offline_scene_graph yes --preprocess_workers 10 --batch_size 256 --log_dir ../experiments/nuScenes/models --train_epochs 20 --node_freq_mult_train --log_tag _vel_ee --augment`                      |
| +Dynamics Integration                     | `python train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScenes/models/int_ee/config.json --train_data_dict nuScenes_train_full.pkl --eval_data_dict nuScenes_val_full.pkl --offline_scene_graph yes --preprocess_workers 10 --batch_size 256 --log_dir ../experiments/nuScenes/models --train_epochs 20 --node_freq_mult_train --log_tag _int_ee --augment`                      |



### CPU Training ###
By default, our training script assumes access to a GPU. If you want to train on a CPU, comment out line 38 in `train.py` and add `--device cpu` to the training command.

## Model Evaluation ##
### Ithaca365 Datasets ###
To evaluate a trained model, you can execute a version of the following command:

```
python evaluate1.py --model <directory that saved your model> --checkpoint=<checkpoints you want to evaluate> --data /home/yx454/original_traj/ablation_study/Trajectron-plus-plus/experiments/processed_ithaca_new_track/ithaca365_test_youya.pkl --output_path results_dis3 --output_tag int_ee --node_type VEHICLE --prediction_horizon 20
```
### nuScenes Dataset ###

To evaluate a trained model's performance on forecasting vehicles, you can execute a one of the following commands from within the `experiments/nuScenes` directory.

| Model                                     | Command                                                                                                                                                                                          |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Base                                      | `python evaluate.py --model models/vel_ee --checkpoint=12 --data ../processed/nuScenes_test_full.pkl --output_path results --output_tag vel_ee --node_type VEHICLE --prediction_horizon 6`       |
| +Dynamics Integration                     | `python evaluate.py --model models/int_ee --checkpoint=12 --data ../processed/nuScenes_test_full.pkl --output_path results --output_tag int_ee --node_type VEHICLE --prediction_horizon 6`       |


## Notes ##

### change covariance ###
If you want to change initial covariance, please change the line 192 & 193 to modify the index (3,3) and (4,4) for the initial covariance and then change line 1271 for index (1,1) and (2,2) of t he initial covariance:

diagonal_values = torch.tensor([(1,1) value, (2,2) value])

cov1 = torch.diag(diagonal_values)

### Training ###
modify the line 61 in `trajectron/model/components/gmm2d1.py` such that it will be `dx = value.cpu() - self.mus.cpu()` 
such that the two variables will be on the same cpu to train.

### nuScenes Dataset ###
If you only want to evaluate models (e.g., produce trajectories and plot them), then the nuScenes mini dataset should be fine. If you want to train a model, then the full nuScenes dataset is required. In either case, you can find them on the [dataset website](https://www.nuscenes.org/).
