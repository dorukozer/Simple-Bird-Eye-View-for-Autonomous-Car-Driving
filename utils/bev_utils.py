LSS_CAMERA_CONF = [
    {
        "type": "sensor.camera.rgb",
        "x": 1.6,
        "y": -0.3,
        "z": 1.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "width": 512,
        "height": 512,
        "fov": 100,
        "id": "front_left_cam"
    },
    {
        "type": "sensor.camera.rgb",
        "x": 1.6,
        "y": 0.3,
        "z": 1.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "width": 512,
        "height": 512,
        "fov": 100,
        "id": "front_right_cam"
    },
    {
        "type": "sensor.camera.rgb",
        "x": 1.3,
        "y": -0.4,
        "z": 1.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": -75.0,
        "width": 512,
        "height": 512,
        "fov": 100,
        "id": "side_left_cam"
    },
    {
        "type": "sensor.camera.rgb",
        "x": 1.3,
        "y": 0.4,
        "z": 1.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 75.0,
        "width": 512,
        "height": 512,
        "fov": 100,
        "id": "side_right_cam"
    }
]


import numpy as np

def get_intrins(sensor):
    image_w = sensor["width"]
    image_h = sensor["height"]
    fov = sensor["fov"]

    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0
    return K



import math

def get_transformation_and_rotation_from_carla(x, y, z, roll, pitch, yaw):
    """
    Convert carla sensor position and rotation to transformation matrix in
    right hand coordinate frame
    """
    trans = np.asarray([x, -y, z])

    roll = math.radians(roll)
    pitch = math.radians(-pitch)  # due to the left hand coord frame in CARLA
    yaw = math.radians(yaw)

    cos = math.cos
    sin = math.sin

    rot = np.zeros((3, 3))
    rot[0, 0] = cos(yaw) * cos(pitch)
    rot[0, 1] = cos(yaw) * sin(pitch) * sin(roll) - sin(yaw)*cos(roll)
    rot[0, 2] = cos(yaw) * sin(pitch) * cos(roll) + sin(yaw)*sin(roll)
    rot[1, 0] = sin(yaw) * cos(pitch)
    rot[1, 1] = sin(yaw) * sin(pitch) * sin(roll) + cos(yaw)*cos(roll)
    rot[1, 2] = sin(yaw) * sin(pitch) * cos(roll) - cos(yaw)*sin(roll)
    rot[2, 0] = -sin(pitch)
    rot[2, 1] = cos(pitch) * sin(roll)
    rot[2, 2] = cos(pitch) * cos(roll)

    return trans, rot


import torch 

def get_trans_and_rot_from_sensor_list(sensor_list):
    rots = []
    trans = []
    intrins = []
    cams = []
    for sensor in sensor_list:
        #print('sensor r ', sensor)
        cams.append(sensor_list[sensor]["id"])
        tran, rot = get_transformation_and_rotation_from_carla(
            sensor_list[sensor]["x"], sensor_list[sensor]["y"], sensor_list[sensor]["z"],
            sensor_list[sensor]["roll"], sensor_list[sensor]["pitch"], sensor_list[sensor]["yaw"])
        trans.append(torch.Tensor(tran))
        rots.append(torch.Tensor(rot))
        intrins.append(torch.Tensor(get_intrins(sensor_list[sensor])))
    return cams, trans, rots, intrins
