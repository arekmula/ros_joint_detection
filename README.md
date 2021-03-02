# ros_joint_segmentation


## Dependencies
- ROS Noetic
- Anaconda

## Installation
- Create conda environment from environment.yml file `conda env create -f environment.yml`
- Activate environment `conda activate ros_joint_seg`
- Create catkin workspace with Python executable set from conda:

```
source /opt/ros/noetic/setup.bash
mkdir -p caktin_ws/src
cd catkin_ws
catkin_make -DPYTHON_EXECUTABLE=~/anaconda3/envs/ros_joint_seg/bin/python3.8
```
- Clone the repository
```
source devel/setup.bash
cd src
git clone https://github.com/arekmula/ros_joint_segmentation
cd ~/catkin_ws
catkin_make
```

## Run with
From activated conda environment run following commands (**remember to source ROS base and devel environment**):
- Setup ROS parameters:
```
rosparam set rgb_image_topic "image/topic"
rosparam set joint_prediction_topic "topic/to/publish/prediction"
rosparam set visualize_joint_prediction True/False
rosparam set joint_seg_model "path/to/model/model.h5"
```

- Run with
```
rosrun ros_joint_segmentation joint_segmentation.py 
```