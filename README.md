# Camera calibration
Collection of OpenCV 5 scripts related to camer calibration  


## Features

#### Targets
- Checkerboard
- ChArUco

#### 

#### Report file generation

## Installation
This project uses `uv` to manage python packages:  
```bash
uv sync
```

## Usage
Help
```bash
uv run camera_calibration.py

```
tbd

## Sample commands
ChArUco (legacy) board with grid of 18x12 using DICT_4X4_250 and report creation
```bash
uv run camera_calibration.py --images_dir captures --board_type charuco --board_cols 18 --board_rows 12 --square_size 0.02 --aruco_dict DICT_4X4_250 --create_report
```
