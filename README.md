# ros_joint_segmentation


## Dependencies
- ROS Noetic
- Tensorflow
- Python 3.8

## Installation
- Create catkin workspace with Python executable set from conda:
```
source /opt/ros/noetic/setup.bash
mkdir -p caktin_ws/src
cd catkin_ws
catkin_make 
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

- Setup ROS parameters:
```
rosparam set rgb_image_topic "image/topic"
rosparam set joint_prediction_topic "topic/to/publish/prediction"
rosparam set visualize_joint_prediction True/False
```

- Run with
```
rosrun ros_joint_segmentation joint_segmentation.py 
```