# xArm5 Lite tomato harvesting

How this fork uses SplatSim with a **uFactory xArm5 Lite** for robotic tomato harvesting: the simulation and real-robot servers, the control panel, GELLO teleoperation and recording, rendering, training, the registry, and policy trials. For installation and the original SplatSim workflow see the [main README](../README.md). Commands and paths are relative to the repository root.

- [Paths to configure](#paths-to-configure)
- [Run the simulation server](#run-the-simulation-server)
- [Control modes](#control-modes)
- [Real xArm5 station](#real-xarm5-station)
- [Teleoperate and record with GELLO](#teleoperate-and-record-with-gello)
- [Record demonstrations](#record-demonstrations)
- [Train the diffusion policy](#train-the-diffusion-policy)
- [Training queue](#training-queue)
- [Training queue panel](#training-queue-panel)
- [Registry of recordings, datasets and models](#registry-of-recordings-datasets-and-models)
- [Run the diffusion policy](#run-the-diffusion-policy)
- [Automated policy trials](#automated-policy-trials)
- [Grasp point and success](#grasp-point-and-success)
- [Comparing models](#comparing-models)

The scripts added by this fork are in `scripts/xarm5/` (recording review and rendering, training and its queue, the registry, policy trials and their comparison, dataset tools). `scripts/` keeps SplatSim's own scripts; three of them are also the entry points of the xArm5 workflow and were extended for it: `launch_nodes.py` (simulation or real-robot server), `run_env_sim.py` (GELLO or policy client) and `launch_camera_nodes.py` (RealSense cameras). Run everything from the repository root.

## Paths to configure

Several paths are still hardcoded to the original author's machine (`/home/magister/...`). Update them to your own:

| File | What to change |
|---|---|
| `configs/object_configs/objects.yaml` | For the `xarm5_lite_plant_v*` entries: `urdf_path` (xArm5 URDF), `source_path` (colmap folder) and `model_path` (gaussian splat output) |
| `configs/folder_configs.yaml` | `traj_folder`: where recorded trajectories are saved |
| `splatsim/agents/lerobot_diffusion_agent.py` | `checkpoint` (trained policy), `dataset_id` / `dataset_root` (LeRobot dataset used for normalization stats) and `urdf_path` (xArm5 URDF used for IK) |

Each scene also needs `data/labels_path/<robot_name>_labels.npy` and `data/pcds_path/<robot_name>_pcd.ply`, which are included for `xarm5_lite_plant_v0`–`v3`.

## Run the simulation server

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
- *Robot*: Reset Robot Joints, the control mode (`position` / `interp` / `velocity`, see [Control modes](#control-modes)), which can be changed while a policy runs and resumes from the current pose, and the sim clock (`fixed` / `realtime`, see `--sim_clock`). **Control Hz** changes the server's `--hz` without restarting it: Run and the trials use the new rate, and the model info warns when it differs from the fps of the model's dataset. In sim it must divide the 240 Hz physics rate (5, 10, 15, 20, 30...) so each command is a whole number of steps; on the real arm it is 1-30 Hz. It can't change while a policy started from the panel is running or a recording is active.
- *Policy*: pick a model from `configs/registry.yaml` (by `name`, or `suggested_name`), see its description, cameras and dataset, and Run / Stop it. Run starts `run_env_sim.py --agent lerobot_diffusion --checkpoint <model>` in another process with the server's `--hz` and port (optionally resetting the robot first); its output goes to the server terminal. Reload Registry picks up a regenerated registry. Its *Trials* subfolder runs `run_policy_trials.py` for the selected model (trials per position, seconds per trial, grid positions such as `0,2,5-7` or empty for all) and shows the progress, the last trial's distances and the results folder. Run and Run Trials share the Stop button: only one client drives the robot at a time. *Load model* adds a model by path, for machines where `configs/registry.yaml` doesn't apply (e.g. a laptop next to the real arm): the model's final folder (`config.json` + `model.safetensors`) and the LeRobot dataset it was trained on (empty = the one in the registry). It checks that the dataset has the model's cameras, selects it, and passes `--dataset_root` to Run and the trials; loaded models are kept in `configs/local_models.yaml` (not in git) and listed first, marked `· local`. With **Record as trial** ticked, Run instead runs one trial at the tomato's current position without moving it (`run_policy_trials.py --here`): recording starts when the policy starts acting and lasts *Trial seconds*, and the result goes to the model's trials folder like any other trial (its `pos_idx` is the grid index if the tomato is on a grid position, else -1).
- *GELLO*: Start GELLO / Stop launches `run_env_sim.py --agent gello --gello_port <port> --start_joints <joints>` as the client (it shares the one-client slot with the policy). *Port* lists the connected ports that have a calibration in `PORT_CONFIG_MAP` (`splatsim/agents/gello_agent.py`; the agent requires that exact name, e.g. `/dev/ttyACM0`) and warns if you lack permission (`sudo usermod -aG dialout $USER`); *Start joints* comes from `configs/gello.yaml`. Untick *Camera windows* to start it without the cv2 windows of `base_rgb` / `wrist_rgb` (`--no_camera_windows`; with `realtime` it then stops asking the server for those views too); the panel's *Cameras* window keeps showing them, and with `realtime` it gets them from the same render thread, outside the physics lock. The client runs at 30 Hz (`realtime_hz`) with `--sim_clock realtime`, otherwise at the server's Control Hz.
- *Recording*: Start / Stop (same as `r`/`s` + Enter in the terminal) and the frame count. *Folder* picks where new recordings go, among the folders next to the default one (e.g. `~/data/bc_data/*`), and *Create Folder* makes a new one; it shows how many episodes the folder has and warns when its recordings are of the other kind (with images vs state only) than what the server records now. The folder can't change during a recording.

*Scene* tab:
- *Tomato*: xyz, gizmo, Random Position, Save Position to Grid, and a list to load a saved grid position (or click its orange sphere in the 3D view).
- *Base cameras*: one tab per camera with its xyz / rpy; drag the gizmo in the 3D view or type the values. Save Camera Poses.

*Tools* tab: **Start** / **Stop** the recordings reviewer (`review_recordings.py`, port 8081) and the training queue (`train_server.py`, port 8082) without another terminal, each with a link that opens it in a new browser tab (using the host this panel was opened with, so it also works from another computer). They run in their own session, so stopping the robot server does not stop them (nor does stopping the queue stop a training in progress); their output goes to `~/.cache/splatsim/tools/<script>.log`. If one is already running on its port (started from a terminal), the panel shows its link instead of starting another. Each panel has its own browser tab title: *xArm5 sim*, *xArm5 real*, *Recordings review*, *Training queue*. Under WSL, *USB (WSL)* shows whether the GELLO (CH343) and the RealSense cameras are attached and attaches them with `usbipd attach --wsl` (what the Windows `.bat` did); a device that is not shared yet needs `usbipd bind --busid <id>` once from an admin terminal. On native Linux the section is hidden.

*Cameras* window: one checkbox per camera (`base_rgb` and `wrist_rgb` on by default), refreshed about twice per second. While a client (e.g. a running policy) is requesting observations, the window reuses those images; otherwise each active camera costs one render.

If `configs/apple_grid.yaml` has positions, the tomato starts at the first one.

**Cameras and arrangements.** `configs/camera_poses.yaml` defines the fixed base cameras (`base_rgb` in the center, `base_left_rgb` and `base_right_rgb` to the sides) and the camera arrangements, which are the subsets used to train and evaluate:

| Arrangement | Cameras |
|---|---|
| `1cam` | `base_rgb` |
| `2cam` | `base_rgb`, `wrist_rgb` |
| `3cam` | `base_left_rgb`, `base_right_rgb`, `wrist_rgb` |

Record with all cameras (the default) so every arrangement is trained on the same demonstrations; then pick the arrangement with `arrangement:` in the training config. Adjust the poses in the panel (*Scene* tab, one tab per base camera, or `--gui pybullet` sliders) and press Save Camera Poses, which rewrites the poses and keeps the arrangements; edit the arrangements by hand. In the *Cameras* window, the Arrangement list turns on the previews of an arrangement.

## Control modes

The policy and GELLO always send joint positions; `--control_mode` only changes how the low-level loop (30 Hz) follows them. The same law, in `splatsim/robots/joint_controller.py`, runs in the simulator and in the real xArm5 driver (`--robot xarm5_lite`), so both behave alike:

| Mode | Real xArm | Behavior |
|---|---|---|
| `position` | mode 1, `set_servo_angle_j` | Step toward the latest target, at most `max_delta` per tick (0.05 rad on the real arm, `SIM_MAX_DELTA` = 0.15 rad in sim). Reaches each target and waits for the next one (stop-and-go). |
| `interp` | mode 1, `set_servo_angle_j` | Minimum-jerk interpolation from the previous command to the new target over one command period. Continuous motion, one period of delay. |
| `velocity` | mode 4, `vc_set_joint_velocity` | `v = feedforward + kp·error`, with velocity and acceleration limits. The feedforward is the velocity implied by consecutive targets, so the arm keeps moving between them. Each command carries a `duration` watchdog: if the control thread stalls, the arm stops. |

Start the real robot with the same `--hz` the client uses (e.g. `--hz 10` for the policy), since `interp` and `velocity` use it as the expected time between targets. Gains (`kp`, `v_max`, `a_max`, ...) are in `ControllerGains`. Each observation includes `control_mode`. Try `velocity` on the real arm with low `v_max` first and the emergency stop at hand.

## Real xArm5 station

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

## Teleoperate and record with GELLO

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
python scripts/xarm5/render_recordings.py ~/data/bc_data/<folder> --fps 10 [--arrangement 2cam | --cameras ...]   # → <folder>_10hz
```
Each episode is resampled by `sim_time` to exactly `1/fps` (joints interpolated to each instant, state recomputed from that pose, gripper command from the nearest frame) and rendered in the same format as below, so the reviewer and the converter work unchanged. Convert it with the same `--fps`, and run the policy with `--sim_clock fixed --hz <fps>`: the dataset step and the policy step are then the same simulated time. The same recordings can be rendered again with other cameras or rates. Review and label the rendered folder (stop labels are per frame); `selection.yaml` is copied over. The output folder gets a `render_info.yaml` (source folder, fps, cameras); when the reviewer opens it, *Show source recording* overlays the original teleoperation trail (orange, ~30 Hz samples) on the resampled frames (blue dots), and the *Resampling* folder shows the source and resampled rates and how far the resampled poses are from the original path, plus a figure of x/y/z and gripper over time.

`render_recordings.py` replaces `--agent replay_trajectory_and_save` for these recordings. That agent replays the joints as commands through the server, so the rendered pose is wherever the controller got to in each step (and the step is the server's `--hz`), and it writes the images into the source episode. `render_recordings.py` restores the exact recorded pose (all joints, gripper included), resamples to an exact rate, writes to a separate folder and needs no running server.

## Record demonstrations

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
python scripts/xarm5/check_recording_timing.py [traj_folder]
```

Review recordings before converting them:
```bash
python scripts/xarm5/review_recordings.py   # http://localhost:8081, folders of ~/data/bc_data (--root, --port)
```
It shows the end-effector trail of the selected episode with a color gradient from start (violet) to end (yellow), the tomato, the other trails of the folder (gray = kept, red = discarded), frame-by-frame playback with the recorded camera images, and the frames, duration and end-effector–tomato distance of each episode. Tick **Discard episode** to exclude it: the selection is saved in `<folder>/selection.yaml` and the converter skips those episodes (`--ignore_selection` to include them). The first time a folder is opened its trails are read in the background (the `.pkl` files include the images, ~1-2 s per episode); after that they come from a cache in `~/.cache/splatsim/trails/`.

**Stop labels for the visual classifier.** The classifier decides when the arm is in grasp position, so the policy can stop. In the reviewer's *Stop labels* folder each episode gets a frame `valid_from`: from there to the end the arm is in position (positive), frames before it are negatives, and the few frames right before it are left out as ambiguous. For sim recordings it is proposed automatically from the grasp error (see [Grasp point](#grasp-point-and-success)): **Accept auto**, **Valid from this frame** (at the playback frame), **No valid frames**, **Exclude** or **Reset**. Real recordings have no tomato position and are labeled by hand. The plot shows the grasp error over the episode, *Trail color → stop label* paints the trail by label (green in position, blue not yet, gray unused), and *Show → stop: unlabeled* walks through the episodes still to review. Labels are saved in `<folder>/stop_labels.yaml` (`splatsim/utils/stop_labels.py`).

Convert recordings to a LeRobot dataset (it detects the image numbering of each episode, both the current per-frame one and the interleaved one of older recordings; by default it pushes the dataset to the Hugging Face Hub at the end):
```bash
python splatsim/utils/lerobot_conversion.py --input_dir <traj_folder> --output_dir <dataset_dir> --fps 10 [--cameras base_rgb base_left_rgb base_right_rgb wrist_rgb] [--no_push]
```
`--input_dir` takes several folders to merge them into one dataset; `--no_push` keeps the dataset local instead of pushing it to the Hugging Face Hub.

## Train the diffusion policy

Parameters (dataset, cameras, `n_obs_steps`, `horizon`, `n_action_steps`, steps, batch size, Hub upload…) live in `configs/train/diffusion.yaml`. Copy it per experiment, or override any field from the CLI:
```bash
python scripts/xarm5/train_diffusion.py --config configs/train/diffusion.yaml
python scripts/xarm5/train_diffusion.py --config configs/train/diffusion.yaml --horizon 32 --cameras base_rgb --no-push-to-hub
```
The output folder `<output_root>/<model_name>/` gets the final policy, checkpoints every `save_freq` steps and the resolved `train_config.yaml`.

## Training queue

Train several models one after another, e.g. to compare camera arrangements or horizons:
```bash
python scripts/xarm5/train_queue.py configs/train/queue_example.yaml --dry_run   # list the runs
python scripts/xarm5/train_queue.py configs/train/queue_example.yaml             # train them
```
A queue file has a `base` config, `common` overrides, explicit `runs` (each with `name`, `description` and any training field) and an optional `sweep` (one run per combination of values, named like `h32-a16-2cam`). Each run's final config, log and status go to `~/data/train_queues/<queue>/`; relaunching skips finished runs (`--rerun` to repeat them, `--only <name>` to pick some). When the queue ends, `configs/registry.yaml` is regenerated so the new models show up in the panel with their name and description.

## Training queue panel

To keep adding trainings while one is running (e.g. queue the next one before leaving an 8-hour run overnight):
```bash
python scripts/xarm5/train_server.py   # http://localhost:8082
```
- *New job*: name, description, base config, dataset (from the registry), camera arrangement or cameras, horizons, steps, batch size and any other training field as YAML. It checks that `horizon` is a multiple of 8, that `n_action_steps` fits in the horizon and that the dataset has the arrangement's cameras.
- *Queue*: the current job's step, loss, speed, ETA and log tail; the job list (reorder, retry, duplicate, remove); **Pause queue** (finish the current job, don't start the next); **Stop current job** (stops it and pauses the queue); and *Import queue file* to add the runs of a `train_queue.py` file.

The queue is saved in `~/data/train_queues/live/jobs.yaml` with each job's config and log. Trainings run in their own session: closing the panel does not stop them, and reopening it picks them up again. After each finished job the registry is regenerated, so the model appears in the simulator panel.

## Registry of recordings, datasets and models

`configs/registry.yaml` links each model to the dataset it was trained on, and each dataset to the GELLO recordings it was converted from. Regenerate it after recording, converting or training:
```bash
python scripts/xarm5/build_registry.py   # defaults: ~/data/bc_data, ~/lerobot/splatsim_test_data, ~/lerobot/examples/tutorial/diffusion/outputs
```
Models are matched to datasets exactly through the normalization statistics stored in the model, and datasets to recordings through their sequence of episode lengths. Fill in `name`, `description` and `notes` by hand (or set `name` / `description` in the training config): regenerating keeps them. Each model also gets a `suggested_name` built from its metadata.

## Run the diffusion policy

Start the server as above, then run the policy with the same `--hz` it was trained at (10 for the current datasets):
```bash
python scripts/run_env_sim.py --agent lerobot_diffusion --hz 10 [--checkpoint <model folder>]
```
Without `--checkpoint` it uses `DEFAULT_CHECKPOINT` in `splatsim/agents/lerobot_diffusion_agent.py`. The dataset the model was trained on is looked up in `configs/registry.yaml` (override with `--dataset_root`). `run_policy_trials.py` takes the same two options. You can also run a model from the panel's *Policy* folder.

## Automated policy trials

`scripts/xarm5/run_policy_trials.py` evaluates the policy over a grid of tomato positions without manual timing. It loads the policy once, then for each position and trial it resets the arm, places the tomato, resets the policy, records the rollout and stops after a fixed simulated duration.

1. Define the tomato positions `(x, y, z)` to test in `configs/apple_grid.yaml`. The easiest way is to place the tomato from the control panel (*Tomato* folder) and press **Save Position to Grid** for each position you want; you can also edit the file by hand.
2. Start the server:
   ```bash
   python scripts/launch_nodes.py --robot sim_xarm5_splat_camera --robot_name xarm5_lite_plant_v3 --hz 10
   ```
3. Run the trials:
   ```bash
   python scripts/xarm5/run_policy_trials.py --trials-per-position 3 --trial-seconds 15
   python scripts/xarm5/run_policy_trials.py --positions 0 4 8 --trials-per-position 1   # only some positions
   ```

Results are saved in `~/data/policy_trials/<model name>/<date>/` (change the base with `--output-dir` and the model folder with `--run-name`; by default it is the model's `name` or `suggested_name` in `configs/registry.yaml`):
- `000/`, `001/`, ...: one recording per trial, in the same format as demonstrations, plus a `meta.json` with the tomato position and trial index.
- `trials.csv`: one row per trial with the number of frames and the final and minimum distance between the end effector and the tomato.

Press Ctrl+C to stop early; the current recording is closed and the arm is reset.

## Grasp point and success

`obs["state"]` is the pose of the arm's last link, not the gripper tip, so the raw end-effector–tomato distance never drops below ~13 cm, even in a perfect demo. What is consistent is where the tomato sits **in the end-effector frame** at the end of the demos (≈ 17 cm along the tool axis):
```bash
python scripts/xarm5/estimate_grasp_point.py --recordings ~/data/bc_data/gello_3plants   # writes configs/grasp_point.yaml
```
The **grasp error** of a frame is the distance between the tomato (in the end-effector frame) and that point (`splatsim/utils/grasp_point.py`). In the demos' last frames it is 1.9 cm (p50) and 3.4 cm (p90). `run_policy_trials.py` records the final and minimum grasp error of each trial, and a trial succeeds when it gets within 4 cm.

## Comparing models

After running trials for several models (panel *Trials* or `run_policy_trials.py`), compare them:
```bash
python scripts/xarm5/compare_trials.py ~/data/policy_trials/2cam-h64 ~/data/policy_trials/1cam-h64 ~/data/policy_trials/3cam-h64
```
Each argument is a model folder (all its runs are pooled) or a single run, optionally as `label=path`. Success is a minimum grasp error ≤ 4 cm by default (`--metric final_grasp_err`, or the raw `min_dist` / `final_dist`; `--success_dist` changes the threshold); trials recorded before the grasp error existed get it computed from their saved frames. It reports each model's success rate with a 95% Wilson interval, a Fisher exact test (2 models) or a chi-square test plus Holm-corrected pairwise Fisher tests (3 or more), and writes `summary.md`, CSV tables and three figures (success rate, success per tomato position, distance per trial) to `~/data/policy_trials/comparisons/<date>/`. With 9 positions, plan on ~10 trials per position per model for these tests to detect moderate differences.
