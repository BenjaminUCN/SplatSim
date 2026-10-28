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
- `--control_mode`: how joint targets (from the policy or GELLO) are followed: `position` (default), `interp` or `velocity`; see [Control modes](#control-modes).
- `--move_apple`: only with `--gui pybullet`, adds the tomato sliders and buttons (the viser panel always has them).
- `--hz`: the rate at which the client sends commands. **It must match `--hz` of the client** (`run_env_sim.py`, default 30). Each command advances the simulation by exactly `1/hz` seconds, so recorded frames are evenly spaced in simulated time.
- `--sim_clock`: `fixed` (default) is the behavior above, right for the policy and trials: the dataset step and the policy step are the same simulated time no matter how slow rendering is. If the client loop can't keep up with `--hz`, the simulation runs in slow motion. For **GELLO teleoperation** use `realtime` (see [Teleoperate and record with GELLO](#teleoperate-and-record-with-gello)): physics runs continuously at wall-clock speed and commands only change the target. `run_env_sim.py` prints the actual loop rate next to `--hz`.

The server renders `base_rgb`, `base_left_rgb`, `base_right_rgb` and `wrist_rgb` from the gaussian splat; render only an arrangement with `--arrangement 2cam`, or any subset with `--cameras base_rgb wrist_rgb` (every rendered camera is recorded). Camera FoV, recording length and other constants live in `splatsim/robots/xarm/config.py`.

**Control panel (viser).** Open http://localhost:8080 in a browser. The 3D view shows the world frame, the end effector, the base camera frustums, the tomato and the saved grid positions. The controls are split into a main panel (docked right) with two tabs and a separate *Cameras* window (docked left); both can be moved, docked or minimized.

*Run* tab:
- *Robot*: Reset Robot Joints, the control mode (`position` / `interp` / `velocity`, see [Control modes](#control-modes)), which can be changed while a policy runs and resumes from the current pose, and the sim clock (`fixed` / `realtime`, see `--sim_clock`).
- *Policy*: pick a model from `configs/registry.yaml` (by `name`, or `suggested_name`), see its description, cameras and dataset, and Run / Stop it. Run starts `run_env_sim.py --agent lerobot_diffusion --checkpoint <model>` in another process with the server's `--hz` and port (optionally resetting the robot first); its output goes to the server terminal. Reload Registry picks up a regenerated registry. Its *Trials* subfolder runs `run_policy_trials.py` for the selected model (trials per position, seconds per trial, grid positions such as `0,2,5-7` or empty for all) and shows the progress, the last trial's distances and the results folder. Run and Run Trials share the Stop button: only one client drives the robot at a time. With **Record as trial** ticked, Run instead runs one trial at the tomato's current position without moving it (`run_policy_trials.py --here`): recording starts when the policy starts acting and lasts *Trial seconds*, and the result goes to the model's trials folder like any other trial (its `pos_idx` is the grid index if the tomato is on a grid position, else -1).
- *Recording*: Start / Stop (same as `r`/`s` + Enter in the terminal) and the frame count.

*Scene* tab:
- *Tomato*: xyz, gizmo, Random Position, Save Position to Grid, and a list to load a saved grid position (or click its orange sphere in the 3D view).
- *Base cameras*: one tab per camera with its xyz / rpy; drag the gizmo in the 3D view or type the values. Save Camera Poses.

*Cameras* window: one checkbox per camera (`base_rgb` and `wrist_rgb` on by default), refreshed about twice per second. While a client (e.g. a running policy) is requesting observations, the window reuses those images; otherwise each active camera costs one render.

If `configs/apple_grid.yaml` has positions, the tomato starts at the first one.

**Cameras and arrangements.** `configs/camera_poses.yaml` defines the fixed base cameras (`base_rgb` in the center, `base_left_rgb` and `base_right_rgb` to the sides) and the camera arrangements, which are the subsets used to train and evaluate:

| Arrangement | Cameras |
|---|---|
| `1cam` | `base_rgb` |
| `2cam` | `base_rgb`, `wrist_rgb` |
| `3cam` | `base_left_rgb`, `base_right_rgb`, `wrist_rgb` |

Record with all cameras (the default) so every arrangement is trained on the same demonstrations; then pick the arrangement with `arrangement:` in the training config. Adjust the poses in the panel (*Scene* tab, one tab per base camera, or `--gui pybullet` sliders) and press Save Camera Poses, which rewrites the poses and keeps the arrangements; edit the arrangements by hand. In the *Cameras* window, the Arrangement list turns on the previews of an arrangement.

### Control modes

The policy and GELLO always send joint positions; `--control_mode` only changes how the low-level loop (30 Hz) follows them. The same law, in `splatsim/robots/joint_controller.py`, runs in the simulator and in the real xArm5 driver (`--robot xarm5_lite`), so both behave alike:

| Mode | Real xArm | Behavior |
|---|---|---|
| `position` | mode 1, `set_servo_angle_j` | Step toward the latest target, at most `max_delta` per tick (0.05 rad on the real arm, `SIM_MAX_DELTA` = 0.15 rad in sim). Reaches each target and waits for the next one (stop-and-go). |
| `interp` | mode 1, `set_servo_angle_j` | Minimum-jerk interpolation from the previous command to the new target over one command period. Continuous motion, one period of delay. |
| `velocity` | mode 4, `vc_set_joint_velocity` | `v = feedforward + kp·error`, with velocity and acceleration limits. The feedforward is the velocity implied by consecutive targets, so the arm keeps moving between them. Each command carries a `duration` watchdog: if the control thread stalls, the arm stops. |

Start the real robot with the same `--hz` the client uses (e.g. `--hz 10` for the policy), since `interp` and `velocity` use it as the expected time between targets. Gains (`kp`, `v_max`, `a_max`, ...) are in `ControllerGains`. Each observation includes `control_mode`. Try `velocity` on the real arm with low `v_max` first and the emergency stop at hand.

### Real xArm5 station

The real arm runs behind the same ZMQ protocol, recorder and control panel as the simulator (`splatsim/robots/station/`), so `run_env_sim.py` (GELLO or the policy), `run_policy_trials.py` and the panel work against both. Settings are in `configs/xarm5_real.yaml`: IP, `home_joints` (the pose **Reset Robot** moves to, slowly; `null` = Reset Robot does nothing), `reset_speed`, the cameras and the recordings folder.
```bash
python scripts/launch_camera_nodes.py --config configs/xarm5_real.yaml   # one ZMQ node per RealSense
python scripts/launch_nodes.py --robot xarm5_lite --hz 10 [--control_mode interp] [--gui none]
```
- Cameras: each entry of `cameras` maps the name the policy sees (`base_rgb`, `wrist_rgb`, as in the sim) to a RealSense serial and a port. With `serial: null` the nodes take the detected cameras in order, which is not stable across reboots; set the serials (`launch_camera_nodes.py` prints them) once the cameras are mounted. Images are 640×480 RGB uint8, the sim's size.
- The panel (http://localhost:8080) has the *Run* tab: Reset Robot, control mode, Policy (Run / Stop), Recording and the camera views. There is no Scene tab: the tomato is placed by hand.
- Recording (`r`/`s` + Enter or the panel) saves one frame per command with the camera images, in `recordings`; frames carry the server clock in `time` (there is no simulated time, so the step is whatever the client loop achieves).
- If a camera node is not running, observations fail with a message naming its port, and the server keeps serving.
- **Trials on the real arm** use the same panel buttons as in the sim (*Run Trials*, or *Run* with *Record as trial*). Since there is no tomato position, `run_policy_trials.py` asks the operator through an *Operator* box in the Policy folder: before each trial, after Reset Robot, "place the tomato at grid position #k" (**Start** / **Skip** / **Stop trials**; with *Record as trial*, wherever you want it), and after it "result?" (**Success** / **Fail** / **Discard**). The answer goes to the `manual_success` column of `trials.csv` (Discard writes no row, the recording stays), the trial is paced at the server's `--hz` on the wall clock, and `compare_trials.py` counts the operator's mark as the success of those trials. The grid positions are in sim world coordinates: mark where each one falls on the real plant before running trials.
- `--robot_ip` overrides the IP in the file. Test new settings with low speeds and the emergency stop at hand.

### Teleoperate and record with GELLO

Rendering one observation takes ~0.25 s (mostly deforming the robot splat to the current pose), so a loop that renders every step can't follow a hand in real time, and recording a frame per rendered step gives an irregular, mislabeled time step. With `--sim_clock realtime` teleoperation and rendering are decoupled:
```bash
python scripts/launch_nodes.py --robot sim_xarm5_splat_camera --robot_name xarm5_lite_plant_v3 --sim_clock realtime
python scripts/run_env_sim.py --agent gello --gello_port /dev/ttyACM0 --start_joints -3.1415 0.4712 -1.8151 0.4188 1.5708 0
```
- The client sends commands at `--hz` (30) without asking for observations; the server's physics follows the clock and the arm follows the leader smoothly (in sim the joint step limit is `SIM_MAX_DELTA`, 0.15 rad per 30 Hz tick, looser than the real arm's 0.05).
- `wrist_rgb` and `base_rgb` (only `base_rgb` with `--no_wrist_camera`) keep showing what the policy will see: a server thread renders them continuously, outside the physics lock, at whatever rate rendering allows (~3-4 Hz), and the client shows them in its windows. The status line shows the loop and camera rates.
- Recording (`r`/`s` + Enter in the server, or the panel) saves **state only**, one frame per command (~30 Hz), with `sim_time` and every PyBullet joint position (`sim_joint_positions`), up to `TELEOP_MAX_RECORD_SECONDS` (120 s).

Then render the images at a fixed rate:
```bash
python scripts/render_recordings.py ~/data/bc_data/<folder> --fps 10 [--arrangement 2cam | --cameras ...]   # → <folder>_10hz
```
Each episode is resampled by `sim_time` to exactly `1/fps` (joints interpolated to each instant, state recomputed from that pose, gripper command from the nearest frame) and rendered in the same format as below, so the reviewer and the converter work unchanged. Convert it with the same `--fps`, and run the policy with `--sim_clock fixed --hz <fps>`: the dataset step and the policy step are then the same simulated time. The same recordings can be rendered again with other cameras or rates. Review and label the rendered folder (stop labels are per frame); `selection.yaml` is copied over. The output folder gets a `render_info.yaml` (source folder, fps, cameras); when the reviewer opens it, *Show source recording* overlays the original teleoperation trail (orange, ~30 Hz samples) on the resampled frames (blue dots), and the *Resampling* folder shows the source and resampled rates and how far the resampled poses are from the original path, plus a figure of x/y/z and gripper over time.

`render_recordings.py` replaces `--agent replay_trajectory_and_save` for these recordings. That agent replays the joints as commands through the server, so the rendered pose is wherever the controller got to in each step (and the step is the server's `--hz`), and it writes the images into the source episode. `render_recordings.py` restores the exact recorded pose (all joints, gripper included), resamples to an exact rate, writes to a separate folder and needs no running server.

### Record demonstrations

With `--sim_clock fixed` (the default), a server running and a client sending commands (e.g. a policy), type in the **server** terminal:

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

Review recordings before converting them:
```bash
python scripts/review_recordings.py   # http://localhost:8081, folders of ~/data/bc_data (--root, --port)
```
It shows the end-effector trail of the selected episode with a color gradient from start (violet) to end (yellow), the tomato, the other trails of the folder (gray = kept, red = discarded), frame-by-frame playback with the recorded camera images, and the frames, duration and end-effector–tomato distance of each episode. Tick **Discard episode** to exclude it: the selection is saved in `<folder>/selection.yaml` and the converter skips those episodes (`--ignore_selection` to include them). The first time a folder is opened its trails are read in the background (the `.pkl` files include the images, ~1-2 s per episode); after that they come from a cache in `~/.cache/splatsim/trails/`.

**Stop labels for the visual classifier.** The classifier decides when the arm is in grasp position, so the policy can stop. In the reviewer's *Stop labels* folder each episode gets a frame `valid_from`: from there to the end the arm is in position (positive), frames before it are negatives, and the few frames right before it are left out as ambiguous. For sim recordings it is proposed automatically from the grasp error (see [Grasp point](#grasp-point-and-success)): **Accept auto**, **Valid from this frame** (at the playback frame), **No valid frames**, **Exclude** or **Reset**. Real recordings have no tomato position and are labeled by hand. The plot shows the grasp error over the episode, *Trail color → stop label* paints the trail by label (green in position, blue not yet, gray unused), and *Show → stop: unlabeled* walks through the episodes still to review. Labels are saved in `<folder>/stop_labels.yaml` (`splatsim/utils/stop_labels.py`).

Convert recordings to a LeRobot dataset (it detects the image numbering of each episode, both the current per-frame one and the interleaved one of older recordings; note that it pushes the dataset to the Hugging Face Hub at the end):
```bash
python splatsim/utils/lerobot_transform_v2.py --input_dir <traj_folder> --output_dir <dataset_dir> --fps 10 [--cameras base_rgb base_left_rgb base_right_rgb wrist_rgb]
```

### Train the diffusion policy

Parameters (dataset, cameras, `n_obs_steps`, `horizon`, `n_action_steps`, steps, batch size, Hub upload…) live in `configs/train/diffusion.yaml`. Copy it per experiment, or override any field from the CLI:
```bash
python scripts/train_diffusion.py --config configs/train/diffusion.yaml
python scripts/train_diffusion.py --config configs/train/diffusion.yaml --horizon 32 --cameras base_rgb --no-push-to-hub
```
The output folder `<output_root>/<model_name>/` gets the final policy, checkpoints every `save_freq` steps and the resolved `train_config.yaml`.

### Training queue

Train several models one after another, e.g. to compare camera arrangements or horizons:
```bash
python scripts/train_queue.py configs/train/queue_example.yaml --dry_run   # list the runs
python scripts/train_queue.py configs/train/queue_example.yaml             # train them
```
A queue file has a `base` config, `common` overrides, explicit `runs` (each with `name`, `description` and any training field) and an optional `sweep` (one run per combination of values, named like `h32-a16-2cam`). Each run's final config, log and status go to `~/data/train_queues/<queue>/`; relaunching skips finished runs (`--rerun` to repeat them, `--only <name>` to pick some). When the queue ends, `configs/registry.yaml` is regenerated so the new models show up in the panel with their name and description.

### Training queue panel

To keep adding trainings while one is running (e.g. queue the next one before leaving an 8-hour run overnight):
```bash
python scripts/train_server.py   # http://localhost:8082
```
- *New job*: name, description, base config, dataset (from the registry), camera arrangement or cameras, horizons, steps, batch size and any other training field as YAML. It checks that `horizon` is a multiple of 8, that `n_action_steps` fits in the horizon and that the dataset has the arrangement's cameras.
- *Queue*: the current job's step, loss, speed, ETA and log tail; the job list (reorder, retry, duplicate, remove); **Pause queue** (finish the current job, don't start the next); **Stop current job** (stops it and pauses the queue); and *Import queue file* to add the runs of a `train_queue.py` file.

The queue is saved in `~/data/train_queues/live/jobs.yaml` with each job's config and log. Trainings run in their own session: closing the panel does not stop them, and reopening it picks them up again. After each finished job the registry is regenerated, so the model appears in the simulator panel.

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

Results are saved in `~/data/policy_trials/<model name>/<date>/` (change the base with `--output-dir` and the model folder with `--run-name`; by default it is the model's `name` or `suggested_name` in `configs/registry.yaml`):
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

### Grasp point and success

`obs["state"]` is the pose of the arm's last link, not the gripper tip, so the raw end-effector–tomato distance never drops below ~13 cm, even in a perfect demo. What is consistent is where the tomato sits **in the end-effector frame** at the end of the demos (≈ 17 cm along the tool axis):
```bash
python scripts/estimate_grasp_point.py --recordings ~/data/bc_data/gello_3plants   # writes configs/grasp_point.yaml
```
The **grasp error** of a frame is the distance between the tomato (in the end-effector frame) and that point (`splatsim/utils/grasp_point.py`). In the demos' last frames it is 1.9 cm (p50) and 3.4 cm (p90). `run_policy_trials.py` records the final and minimum grasp error of each trial, and a trial succeeds when it gets within 4 cm.

### Comparing models

After running trials for several models (panel *Trials* or `run_policy_trials.py`), compare them:
```bash
python scripts/compare_trials.py ~/data/policy_trials/2cam-h64 ~/data/policy_trials/1cam-h64 ~/data/policy_trials/3cam-h64
```
Each argument is a model folder (all its runs are pooled) or a single run, optionally as `label=path`. Success is a minimum grasp error ≤ 4 cm by default (`--metric final_grasp_err`, or the raw `min_dist` / `final_dist`; `--success_dist` changes the threshold); trials recorded before the grasp error existed get it computed from their saved frames. It reports each model's success rate with a 95% Wilson interval, a Fisher exact test (2 models) or a chi-square test plus Holm-corrected pairwise Fisher tests (3 or more), and writes `summary.md`, CSV tables and three figures (success rate, success per tomato position, distance per trial) to `~/data/policy_trials/comparisons/<date>/`. With 9 positions, plan on ~10 trials per position per model for these tests to detect moderate differences.

## Credits, licenses and citations

This fork builds on the following projects. If you use it in a publication, please cite the ones your work relies on.

| Project | Used for | License |
|---|---|---|
| [SplatSim](https://github.com/qureshinomaan/SplatSim) | Base simulator (PyBullet + gaussian splat rendering) | No license file in the upstream repository; ask the authors before redistributing |
| [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (submodule) | Splat training and rendering | Inria/MPII license: **research and evaluation use only, non-commercial** |
| [GELLO](https://github.com/wuphilipp/gello_software) (submodule, [fork](https://github.com/BenjaminUCN/gello_software)) | Teleoperation, ZMQ robot client/server | MIT |
| [LeRobot](https://github.com/huggingface/lerobot) | Datasets, diffusion policy training and inference | Apache 2.0 |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | Policy architecture (LeRobot implementation) | — (method) |
| [viser](https://github.com/nerfstudio-project/viser) | Web control panel | Apache 2.0 (LICENSE file; the PyPI metadata says MIT) |
| [PyBullet](https://github.com/bulletphysics/bullet3) | Physics simulation | zlib |
| [pybullet-URDF-models](https://github.com/GPT-RL/pybullet-URDF-models) (submodule, [fork](https://github.com/BenjaminUCN/pybullet-URDF-models)) | Object models | MIT |

All of them allow academic use and publication. When you redistribute code (for example, publishing this repository with the paper), keep each project's license and copyright notice, and note that the gaussian-splatting license does not allow commercial use.

```bibtex
@article{qureshi2024splatsim,
  title={SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting},
  author={Qureshi, Mohammad Nomaan and Garg, Sparsh and Yandun, Francisco and Held, David and Kantor, George and Silwal, Abhisesh},
  journal={arXiv preprint arXiv:2409.10161},
  year={2024}
}

@article{kerbl3Dgaussians,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023}
}

@misc{wu2023gello,
  title={GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators},
  author={Philipp Wu and Yide Shentu and Zhongke Yi and Xingyu Lin and Pieter Abbeel},
  year={2023}
}

@misc{cadene2024lerobot,
  author={Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
  title={LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  howpublished={\url{https://github.com/huggingface/lerobot}},
  year={2024}
}

@inproceedings{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2023}
}

@article{yi2025viser,
  title={Viser: Imperative, web-based 3d visualization in python},
  author={Yi, Brent and Kim, Chung Min and Kerr, Justin and Wu, Gina and Feng, Rebecca and Zhang, Anthony and Kulhanek, Jonas and Choi, Hongsuk and Ma, Yi and Tancik, Matthew and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2507.22885},
  year={2025}
}
```

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
 