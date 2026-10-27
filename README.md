# SplatSim
SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting

[Project Page](https://splatsim.github.io) | [Arxiv](https://arxiv.org/abs/2409.10161)



This repository contains the code for the paper "SplatSim". 

> **About this fork:** this fork of [qureshinomaan/SplatSim](https://github.com/qureshinomaan/SplatSim) adapts SplatSim to a **uFactory xArm5 Lite** for robotic tomato harvesting. It adds xArm5 scenes (`xarm5_lite_plant_v*`), a simulation server with base/wrist splat cameras and trajectory recording, a LeRobot diffusion-policy agent and an automated policy evaluation script. See [xArm5 Lite tomato harvesting](#xarm5-lite-tomato-harvesting). The original UR5 instructions below still apply.

## Installation

### Clone this repository
```bash
cd ~/code
git clone --recursive https://github.com/BenjaminUCN/SplatSim.git
```

If you accidentally forgot the `--recursive`, run this command to download the files for the `submodules` folder: `git submodule update --init --recursive`

### Submodule forks

This repo depends on modified versions of two submodules, hosted as forks on the `splatsim-xarm5` branch. They are already set in `.gitmodules`, so a recursive clone fetches them automatically:

| Submodule | Fork | Upstream | Changes |
|---|---|---|---|
| `submodules/gello_software` | [BenjaminUCN/gello_software](https://github.com/BenjaminUCN/gello_software/tree/splatsim-xarm5) | [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software) | Adds the robot `state` to the observations and skips the RealSense hardware reset on startup |
| `submodules/pybullet-URDF-models` | [BenjaminUCN/pybullet-URDF-models](https://github.com/BenjaminUCN/pybullet-URDF-models/tree/splatsim-xarm5) | [GPT-RL/pybullet-URDF-models](https://github.com/GPT-RL/pybullet-URDF-models) | Replaces the `plastic_apple` model with a tomato mesh and textures (the name `plastic_apple` is kept) |

If you cloned before the submodules pointed to these forks, update them with:
```bash
git submodule sync
git submodule update --init --recursive
```

### Create conda env

```bash
conda create -n splatsim python=3.12
conda activate splatsim
```

### Install pytorch


Go to https://pytorch.org/get-started/locally/ to find the right commands for your system. For example, this repo worked with this configuration:
  - torch==2.7.1
  - torchaudio=2.7.1+cu126
  - torchvision=0.22.1+cu126

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Install submodules

Note: If nothing is inside the `submodules/` folder, run `git submodule update --init --recursive`

```bash
CC=gcc-11 CXX=g++-11 pip install --no-build-isolation submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/diff-gaussian-rasterization
pip install submodules/pybullet-URDF-models 
pip install submodules/pybullet-playground-wrapper/ submodules/ghalton

# This needs editable mode for some reason
pip install -e submodules/gaussian-splatting-wrapper

# Seems like -e pip installs must be installed separately
pip install -e submodules/gello_software

# Install dependencies from gello_software
pip install -r submodules/gello_software/requirements.txt
pip install -e submodules/gello_software/third_party/DynamixelSDK/python

# Allow the install to use your version of pytorch; this takes a few mins for some reason
pip install submodules/simple-knn/ --no-build-isolation
```

QOL to clean up git status
```bash
echo '*.egg-info' >> .git/modules/submodules/ghalton/info/exclude
echo '*.egg-info' >> .git/modules/submodules/gello_software/modules/third_party/DynamixelSDK/info/exclude
echo '*.egg-info' >> .git/modules/submodules/pybullet-URDF-models/info/exclude
echo 'build/*' >> .git/modules/submodules/pybullet-URDF-models/info/exclude
```

### Install other dependencies

Install requirements:
```bash
cd ~/code/SplatSim
pip install -r requirements.txt
```

Note: It's important to pip install the `ghalton` submodule in the previous step before installing requirements.txt because for some reason ghalton must be installed from source

### Install this repo as a package

Install in editable mode:

```bash
cd ~/code/SplatSim
pip install -e .
```

## Running the rendering code 
### 1. Download the colmap and gaussian-splatting models from the below links:
- [colmap (test_data)](https://drive.google.com/file/d/14D3fFtaPX4GBe9dSJLKAIvUYlgK7fUxS/view?usp=sharing)
- [gaussian-splats (output)](https://drive.google.com/file/d/1rAUkf7l2ZZqG1Bm3ih6cAO5HCd9dSTO-/view?usp=sharing)
- [trajectories (bc_data/gello)](https://drive.google.com/file/d/1NhSBNYMi51hETAspk6vN7F-Ih1134_lt/view?usp=sharing)

`test_data` is the folder name of the output of colmap. `output` is the folder name of the output of gaussian splat generation. `bc_data/gello` is the folder name of the demo trajectories recorded by one of the scripts in this repo.

Assume below that these files are stored under:

- test_data: /home/yourusername/data/test_data
- output: /home/yourusername/data/output
- bc_data/gello: /home/yourusername/data/bc_data/gello

### 2. Configure the configs to match your folder directory structure

#### Open `configs/object_configs/objects.yaml`. The data you downloaded in step 1 is for the robot `robot_iphone`.

Modify `robot_iphone` as below:
- source_path: /home/yourusername/data/test_data/robot_iphone # Path to a folder you downloaded
- model_path: /home/yourusername/data/output/robot_iphone # Path to a folder you downloaded

Modify all `ply_path` attributes to point to `/home/yourusername/data/output/...`, for example for `plastic_apple`

#### Open `configs/folder_configs.yaml`

Modify as follows:
- traj_folder: /home/yourusername/data/bc_data/gello

### 3. Run the rendering script:

Launch the robot server which includes an apple and a plate:
```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name robot_iphone
```

In another terminal tab, launch a node that will send the recorded trajectories in `/home/yourusername/data/bc_data/gello` to the server so that it will be rendered:
```bash
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

A window should pop up that is a rendering of the robot in the pybullet simulation. If you drag the end effector of the robot around in pybullet, it should be reflected in the render.

The rendered images for trajectory 0 are saved in `{traj_folder}/0/images_1`, for example `/home/yourusername/data/bc_data/gello/0/images_1`.

Congrats! Your static splat is now being simulated! 🚀

## xArm5 Lite tomato harvesting

### Paths to configure

Several paths are still hardcoded to the original author's machine (`/home/magister/...`). Update them to your own:

| File | What to change |
|---|---|
| `configs/object_configs/objects.yaml` | For the `xarm5_lite_plant_v*` entries: `urdf_path` (xArm5 URDF), `source_path` (colmap folder) and `model_path` (gaussian splat output) |
| `configs/folder_configs.yaml` | `traj_folder`: where recorded trajectories are saved |
| `splatsim/agents/lerobot_diffusion_agent.py` | `checkpoint` (trained policy), `dataset_id` / `dataset_root` (LeRobot dataset used for normalization stats) and `urdf_path` (xArm5 URDF used for IK) |

Each scene also needs `data/labels_path/<robot_name>_labels.npy` and `data/pcds_path/<robot_name>_pcd.ply`, which are included for `xarm5_lite_plant_v0`–`v3`.

### Run the simulation server

```bash
python scripts/launch_nodes.py --robot sim_xarm5_splat_camera --robot_name xarm5_lite_plant_v3 --hz 10
```

- `--robot_name`: the scene from `objects.yaml` (`xarm5_lite_plant_v3` is the latest).
- `--gui`: `viser` (default) opens a web control panel at http://localhost:8080 (`--panel_port`); `pybullet` keeps the old PyBullet sliders and buttons instead.
- `--move_apple`: only with `--gui pybullet`, adds the tomato sliders and buttons (the viser panel always has them).
- `--hz`: the rate at which the client sends commands. **It must match `--hz` of the client** (`run_env_sim.py`, default 30). Each command advances the simulation by exactly `1/hz` seconds, so recorded frames are evenly spaced in simulated time.

The server renders `base_rgb`, `base2_rgb` and `wrist_rgb` from the gaussian splat; pick a subset with `--cameras base_rgb wrist_rgb` (every rendered camera is recorded). Camera FoV, recording length and other constants live in `splatsim/robots/xarm/config.py`.

**Control panel (viser).** Open http://localhost:8080 in a browser. It shows the world frame, the end effector, the base camera frustums, the tomato and the saved grid positions in 3D, and has these folders:
- *Robot*: Reset Robot Joints.
- *Policy*: pick a model from `configs/registry.yaml` (by `name`, or `suggested_name`), see its description, cameras and dataset, and Run / Stop it. Run starts `run_env_sim.py --agent lerobot_diffusion --checkpoint <model>` in another process with the server's `--hz` and port (optionally resetting the robot first); its output goes to the server terminal. Reload Registry picks up a regenerated registry.
- *Recording*: Start / Stop (same as `r`/`s` + Enter in the terminal) and the frame count.
- *Base cameras*: one tab per camera with its xyz / rpy; drag the gizmo in the 3D view or type the values. Save Camera Poses.
- *Tomato*: xyz, gizmo, Random Position, Save Position to Grid, and a list to load a saved grid position (or click its orange sphere in the 3D view).
- *Camera previews*: one checkbox per camera; each one you turn on is rendered about twice per second (all off by default because each render costs time).

If `configs/apple_grid.yaml` has positions, the tomato starts at the first one.

**Base cameras.** `base_rgb` and `base2_rgb` are fixed cameras with independent poses, each with its own controls (panel tab, or X/Y/Z/Roll/Pitch/Yaw sliders `base`, `base2` with `--gui pybullet`). "Save Camera Poses" writes both poses to `configs/camera_poses.yaml`, which the server loads on startup and which overrides the defaults in `BASE_CAMS` (`config.py`). Fix the camera poses before recording a dataset, and record with all cameras so you can later train on 1, 2 or 3 of them from the same demonstrations.

### Record demonstrations

With the server running and a client sending commands (e.g. `run_env_sim.py --agent gello` or a policy), type in the **server** terminal:

- `r` + Enter: start recording
- `s` + Enter: stop recording (it also stops automatically after `MAX_RECORD_SECONDS` of simulated time, 2 s by default)

One frame is saved per command received, in `traj_folder`:

```
<traj_folder>/000/00001.pkl, 00002.pkl, ...        # observations: joints, ee state, object poses, sim_time
<traj_folder>/000/image_1/base_rgb_00001.png, ...  # one image per camera and frame
```

Check that recordings have a constant time step:
```bash
python scripts/check_recording_timing.py [traj_folder]
```

Convert recordings to a LeRobot dataset (it detects the image numbering of each episode, both the current per-frame one and the interleaved one of older recordings; note that it pushes the dataset to the Hugging Face Hub at the end):
```bash
python splatsim/utils/lerobot_transform_v2.py --input_dir <traj_folder> --output_dir <dataset_dir> --fps 10 [--cameras base_rgb base2_rgb wrist_rgb]
```

### Train the diffusion policy

Parameters (dataset, cameras, `n_obs_steps`, `horizon`, `n_action_steps`, steps, batch size, Hub upload…) live in `configs/train/diffusion.yaml`. Copy it per experiment, or override any field from the CLI:
```bash
python scripts/train_diffusion.py --config configs/train/diffusion.yaml
python scripts/train_diffusion.py --config configs/train/diffusion.yaml --horizon 32 --cameras base_rgb --no-push-to-hub
```
The output folder `<output_root>/<model_name>/` gets the final policy, checkpoints every `save_freq` steps and the resolved `train_config.yaml`.

### Registry of recordings, datasets and models

`configs/registry.yaml` links each model to the dataset it was trained on, and each dataset to the GELLO recordings it was converted from. Regenerate it after recording, converting or training:
```bash
python scripts/build_registry.py   # defaults: ~/data/bc_data, ~/lerobot/splatsim_test_data, ~/lerobot/examples/tutorial/diffusion/outputs
```
Models are matched to datasets exactly through the normalization statistics stored in the model, and datasets to recordings through their sequence of episode lengths. Fill in `name`, `description` and `notes` by hand (or set `name` / `description` in the training config): regenerating keeps them. Each model also gets a `suggested_name` built from its metadata.

### Run the diffusion policy

Start the server as above, then run the policy with the same `--hz` it was trained at (10 for the current datasets):
```bash
python scripts/run_env_sim.py --agent lerobot_diffusion --hz 10 [--checkpoint <model folder>]
```
Without `--checkpoint` it uses `DEFAULT_CHECKPOINT` in `splatsim/agents/lerobot_diffusion_agent.py`. The dataset the model was trained on is looked up in `configs/registry.yaml` (override with `--dataset_root`). `run_policy_trials.py` takes the same two options. You can also run a model from the panel's *Policy* folder.

### Automated policy trials

`scripts/run_policy_trials.py` evaluates the policy over a grid of tomato positions without manual timing. It loads the policy once, then for each position and trial it resets the arm, places the tomato, resets the policy, records the rollout and stops after a fixed simulated duration.

1. Define the tomato positions `(x, y, z)` to test in `configs/apple_grid.yaml`. The easiest way is to place the tomato from the control panel (*Tomato* folder) and press **Save Position to Grid** for each position you want; you can also edit the file by hand.
2. Start the server:
   ```bash
   python scripts/launch_nodes.py --robot sim_xarm5_splat_camera --robot_name xarm5_lite_plant_v3 --hz 10
   ```
3. Run the trials:
   ```bash
   python scripts/run_policy_trials.py --trials-per-position 3 --trial-seconds 15
   python scripts/run_policy_trials.py --positions 0 4 8 --trials-per-position 1   # only some positions
   ```

Results are saved in `~/data/policy_trials/<date>/` (change with `--output-dir`):
- `000/`, `001/`, ...: one recording per trial, in the same format as demonstrations, plus a `meta.json` with the tomato position and trial index.
- `trials.csv`: one row per trial with the number of frames and the final and minimum distance between the end effector and the tomato.

Press Ctrl+C to stop early; the current recording is closed and the arm is reset.

# Optional

## Adding a new robot

What if you want to simulate a different robot than the one downloaded above? Or with a new background?

### Create gaussian splat

First, create a gaussian splat of your robot within the scene. Record the joint angles of the robot in your static scene
<details>
<summary>Tips for creating a good gaussian splat of the robot</summary>

- Set the pose of your robot to be in an easy-to-describe state. For example, all exactly 90 degree angles. This is for easier calibration later

- Don't put any objects within the robot's rectangular bounding box. The pipeline currently uses a simple segmentation technique with a rectangular bounding box

- 1-2 minute landscape video on a smartphone or similar

- Capture the scene from all angles

- Make sure to capture areas of finer detail, for example the robot gripper, by sometimes moving the camera closer to it

- Make sure the background is textured; use posters to cover plain walls. This helps when recovering camera poses (structure from motion / colmap)

- Diffuse lighting is best (ex: white paper over a strong light), or else the result will have glares, which will not be updated by the simulation when robot joints are moved
</details>

Train the gaussian splat with the [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) repo, which is in `submodules/gausisan-splatting-wrapper/gaussian_splatting` in this repo. Software like Polycam unfortunately does not give you the structure from motion outputs that are used for generating camera views in this repo.
<details>
<summary> Summary of how to train splat </summary>

- Create a folder structure for example at `~/data/your_robot_name/input`. Put all images into the `input` folder. For example, you can convert your video to images with ffmpeg (ex: `ffmpeg -i my_video.MOV -qscale:v 1 output_%04d.jpeg`).

- Recover camera poses with structure from motion / colmap. First, install colmap then run the script. This will populate `~/data/your_robot_name` with camera info
```bash
conda install colmap
python submodules/gaussian-splatting-wrapper/gaussian_splatting/convert.py -s ~/data/your_robot_name
```

- Train gaussian splat. This will create a folder that looks like `./output/258f657d-c` from where you run this command, and it contains the `output/258f657-c/point_cloud/iteration_30000/point_cloud.ply` file, which is the gaussian splat.
```bash
python submodules/gaussian-splatting-wrapper/gaussian_splatting/train.py -s ~/data/your_robot_name
```
</details>

### Align the simulator and the splat

#### Configs

Add `your_robot_name` to `configs/object_configs/objects.yaml`. First, copy-paste the attributes from `robot_iphone`.

- Set `model_path` to the folder output of the gaussian splat training (ex: `~/.../.../output/258f657d-c`)

- Set `source_path` to the folder with the image data and colmap outputs (ex: `~/data/your_robot_name/input`)

- Set `joint_states` (radians) to the joint angles that the robot had when the gaussian splat data was collected. There might be an extra 0 preceding the base joint (ex: [0, 0, 1.57, ...])

- If you have a different URDF, change `urdf_path`. Note that `robot_iphone` is a UR5 robot.

#### Convert URDF to point cloud

Run
```bash
python scripts/articulated_robot_pipeline.py --robot_name your_robot_name
```

Verify that the first point cloud visualization has the same joint poses as your robot had in the splat. If not, adjust `joint_states`. Ignore the second visualization for now.

The point cloud is outputted in `data/pcds_path/your_robot_name_pcd.ply`.

#### Align robot coordinate frames in sim and in splat

Download CloudCompare, which visualizes point clouds. 

Open both the URDF point cloud `data/pcds_path/your_robot_name_pcd.ply` and the gaussian splat `output/.../point_cloud/point_cloud/iteration_30000/point_cloud.ply`. The goal is to apply transformations (rotation/translation/scale) *to your splat* such that the robot arm matches between the sim and splat, then you can copy that transformation to a config file. Don't apply transformations to the simulated robot arm.

<details>
<summary> Tips and tricks with CloudCompare </summary>

- To see rgb colors on your trained splat, download `3dgsconverter` and run `3dgsconverter -i point_cloud.ply -o output_cloudcompare.ply -f cc --rgb`. After importing `output_cloudcompare.ply` to CloudCompare, select it in the top left sidebar, then in the bottom left sidebar, set `Properties > Colors` to `RGB`

- Use the Segment tool (scissor in top bar) to crop out the table and other objects, leaving only the robot. Also crop out any wires (which wouldn't be present in the simulated robot). Note: select the point cloud you want to segment on the left toolbar before you click Segment, or else it will try to segment the wrong point cloud, thus making no changes. The points you segmented out re-appear after you save the segmentation because they are now in another group (in the left toolbar). You can deselect it to stop visualizing it. First select create the selection polygon with left clicks, right click to finish your polygon, click either Segment In or Segment Out, then if you want to keep on iterating on this, find a new angle then press the unpause button to start segmenting again. When you're done, press the green checkmark.

- The fastest way to align the robots seems to doing ICP w/o scale adjustment, manual adjustment, then ICP with scale adjustment. To start ICP, ctrl+click both point clouds, then `Tools > Registration > Fine Registration (ICP)`. Set the simulated robot to be the reference and the splat to be the aligned. Uncheck `adjust scale` for this first pass of ICP. Then, manually fix errors with `Translate/Rotate` (click on point cloud in left sidebar then click on the button at the top toolbar). Then run ICP again but with `adjust scaling` checked. If the URDF is fundamentally different from your real robot, prioritize aligning the robot's end effector to mitigate cascading error when doing forward kinematics in the sim.

- You can double-check alignment by setting the floor as visible and seeing if the floor planes are aligned, or by looking at all orthographic views (left toolbar)
</details>

The splat-to-simulator transformation is in `Transformation History` (scroll to the bottom of Properties in the left sidebar). Copy-paste it to `configs/object_configs/objects.yaml` under your_robot_name > transformation > matrix, while fitting the yaml format

#### Double check calibration

Run this script below again. The last visualization that compares the two point clouds should have colors and coordinate frames lined up.

```bash
python scripts/articulated_robot_pipeline.py --robot_name your_robot_name
```

<details>
<summary>Note if using WSL</summary>

Run this before running the script:
```bash
export DISPLAY=:0
export WAYLAND_DISPLAY=
export XDG_SESSION_TYPE=x11

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
```
</details>

Note: `urdf_bbox_adjustment` can handle cases where your physical robot has an additional attachment compared to the URDF. You can check its effects in the same final visualization

#### Your custom robot can now follow the same recorded joint state trajectories!

Launch the simulation server
```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name your_robot_name
```

Set the robot to follow the recorded trajectories.
```bash
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

## Generating new trajectories

The trajectories are stored at the folder specified in `configs/trajectory_configs.yaml` (`trajectory_folder`). New trajectories are added as larger folder id numbers, and then the replay_trajectory agent starts playing trajectories from folder 0 and up.

For new trajectories, clear out the trajectory folder. Either change `trajectory_folder` or move the previously generated trajectories to another location.

The provided demos are for placing an apple on a plate. If instead you wanted to generate demos for placing a banana on a plate, run

```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_banana --robot_name your_robot_name
```

This populates the `trajectory_folder`. Then, to replay these new trajectories, do the same visualization setup but with this banana on plate environment:

Launch the simulation server
```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_banana_interactive --robot_name your_robot_name
```

Set the robot to follow the recorded trajectories.
```bash
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

You can use `splatsim/robots/sim_robot_pybullet_object_on_plate.py` as a template for configuring custom environments.

## GELLO integration

[GELLO](https://github.com/wuphilipp/gello_software) can be used as a teleoperation system for collecting demos. Please refer to the GELLO repo for hardware setup and general connectivity. Connect the GELLO via USB.

Start the simulation server in interactive mode:
```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name your_robot_name
```

Set the robot to follow GELLO commands and show the save interface GUI:
```bash
python scripts/run_env_sim.py --agent gello --use-save-interface
```

If you move your GELLO, it the simulated robot should move, as well.

In the gray save interface window, you can start and stop recording a new demonstration. Click on the window and press `s` to start recording (the window will turn green). When  you're done recording, click on the window and press `q` (the window will turn red). You can do multiple `s` and `q` recordings.

Play back your recordings with the same command as before (note that new recordings are played last; you can delete or move files out of `trajectory_folder` to view your newly recorded trajectories)
```bash
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

## Visualizing a trained policy in the Splat Sim

For the xArm5 LeRobot diffusion policy, see [Run the diffusion policy](#run-the-diffusion-policy) and [Automated policy trials](#automated-policy-trials).

## A list of TODOs

- [x] ~~Proper instructions for installation. Current instructions might not work.~~
    - [x] Installation instructions should work now.
- [x] ~~Add links to pretrain gaussian-splats and trajectories, so that people can run rendering script.~~
    - [x] Links to pretrain gaussian-splats, colmap and trajectories are added.
- [x] ~~Create a new file for rendering robot and objects, without hardcoding the segmentation and shifting everything to KNN based segmentation.~~
- [x] Clean up the splat folder for only keeping necessary files and easy creation of KNN based segmentation for robots.
- [x] Documentation for the codebase.
- [x] Adding new robots (in sim or any other environment).
- [x] Instructions to generate a trajectory and render it. 
    - [ ] Trajectory format should be specified properly.
- [x] Clean up the gello folder and only keep files that are necessary. 
 