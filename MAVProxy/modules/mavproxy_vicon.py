"""
use vicon data to provide VISION_POSITION_ESTIMATE and GPS_INPUT data
"""

import math
import threading
import time
import numpy as np
import json

from MAVProxy.modules.lib import mp_module
from MAVProxy.modules.lib import mp_settings
from MAVProxy.modules.lib import LowPassFilter2p
from MAVProxy.modules.lib import mp_util
from pymavlink.rotmat import Vector3
from pymavlink.quaternion import Quaternion
from pymavlink import mavutil
from pymavlink import mavextra

# from pyvicon import pyvicon
import motioncapture

import numpy as np


def matrix_to_vicon_format(pose_matrix):
    """
    Converts a 4x4 Transformation Matrix to Vicon format:
    Translation: [X, Y, Z]
    Rotation:    [qX, qY, qZ, qW]
    """
    # 1. Extract Translation (Column 3, Rows 0-2)
    # Vicon usually streams in Millimeters by default.
    # If your tracker logic (previous code) was in Meters,
    # you might need to multiply by 1000 if you want exact Vicon scaling.
    translation = pose_matrix[:3, 3]

    # 2. Extract Rotation Matrix (3x3 top-left)
    rot_matrix = pose_matrix[:3, :3]

    # 3. Convert to Quaternion
    # Scipy's as_quat() returns [x, y, z, w] (Scalar Last)
    # This matches the standard Vicon DataStream SDK convention.
    r = sciR.from_matrix(rot_matrix)
    quat = r.as_quat()

    return translation, quat


# Fast Nearest Neighbor Search (Numba Optimized)
# @njit(fastmath=True)
def find_correspondences(model_points_transformed, scene_cloud, max_dist_sq):
    """
    For each model point, find the closest point in the scene cloud.
    Includes 'Gating': ignores correspondences further than max_dist.
    """
    n_model = model_points_transformed.shape[0]
    n_scene = scene_cloud.shape[0]

    # Arrays to store the matches
    # src = model points (transformed), dst = found scene points
    src_matches = np.zeros((n_model, 3), dtype=np.float32)
    dst_matches = np.zeros((n_model, 3), dtype=np.float32)
    valid_mask = np.zeros(n_model, dtype=np.bool_)

    match_count = 0

    for i in range(n_model):
        qx, qy, qz = model_points_transformed[i]

        best_dist_sq = np.inf
        best_idx = -1

        # Brute force search (fastest for dynamic clouds on CPU)
        for j in range(n_scene):
            dx = scene_cloud[j, 0] - qx
            dy = scene_cloud[j, 1] - qy
            dz = scene_cloud[j, 2] - qz
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = j

        # GATING: Only accept if within reasonable distance
        if best_idx != -1 and best_dist_sq < max_dist_sq:
            src_matches[match_count] = model_points_transformed[i]
            dst_matches[match_count] = scene_cloud[best_idx]
            valid_mask[match_count] = True
            match_count += 1

    return src_matches[:match_count], dst_matches[:match_count], match_count


# Rigid Body Alignment (Kabsch Algorithm)
def solve_kabsch(P, Q):
    """
    Finds optimal Rotation (R) and Translation (t) such that Q approx = R*P + t
    P: Source points (Nx3)
    Q: Target points (Nx3)
    """
    # 1. Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # 2. Center the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # 3. Compute Covariance Matrix
    H = P_centered.T @ Q_centered

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)

    # 5. Compute Rotation
    R = Vt.T @ U.T

    # 6. Handle Reflection case (determinant must be +1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 7. Compute Translation
    t = centroid_Q - R @ centroid_P

    return R, t


# The Tracker Class
class RigidBodyTracker:
    def __init__(self, marker_pattern):
        """
        marker_pattern: (4x3) numpy array of marker positions in OBJECT frame
        """
        self.model_local = marker_pattern.astype(np.float32)

        # Current Pose Estimate (4x4 Transformation Matrix)
        self.pose = np.eye(4, dtype=np.float32)

    def set_initial_pose(self, transform_matrix):
        self.pose = transform_matrix.astype(np.float32)

    def track(self, scene_cloud, max_dist=0.1, iterations=5):
        """
        Refines the pose based on the new scene cloud.
        scene_cloud: (Nx3) numpy array
        max_dist: Maximum distance (meters) to search for a marker.
                  Acts as a filter against other objects.
        """
        scene_cloud = scene_cloud.astype(np.float32)
        max_dist_sq = max_dist ** 2

        for _ in range(iterations):
            # 1. Transform Model Points to World using current Pose Estimate
            # R * p + t
            curr_R = self.pose[:3, :3]
            curr_t = self.pose[:3, 3]

            # (4x3) transformed points
            model_world = (curr_R @ self.model_local.T).T + curr_t

            # 2. Find Correspondences (Association)
            # We treat 'model_world' as the query points to find in 'scene_cloud'
            src_pts, dst_pts, n_matches = find_correspondences(
                model_world, scene_cloud, max_dist_sq
            )

            # 3. Check for tracking loss
            # We need at least 3 points to resolve 3D orientation uniquely
            if n_matches < 3:
                # Keep old pose, but warn (or return status)
                print("Warning: Tracking lost (not enough markers found)")
                break

                # 4. Compute Correction (Drift)
            # We want to move 'src_pts' (current estimate) to 'dst_pts' (observation)
            R_delta, t_delta = solve_kabsch(src_pts, dst_pts)

            # 5. Update Pose
            # New_Pose = Delta * Old_Pose
            # Construct Delta Matrix
            Delta = np.eye(4, dtype=np.float32)
            Delta[:3, :3] = R_delta
            Delta[:3, 3] = t_delta

            self.pose = Delta @ self.pose

        return self.pose


def find_closest_cpu(points, query_point):
    # 1. Vectorized Subtraction (allocates new array, but fast)
    diff = points - query_point

    # axis=1 sums across x,y,z
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    # OR: dist_sq = np.sum(diff**2, axis=1) # Slightly slower than einsum usually

    # 3. Argmin
    min_idx = np.argmin(dist_sq)

    return min_idx

def quaternion_to_euler(x, y, z, w):
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = math.sqrt(1 + 2 * (w * y - x * z))
    cosp = math.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * math.atan2(sinp, cosp) - math.pi / 2

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]


class ViconModule(mp_module.MPModule):

    def __init__(self, mpstate):
        super(ViconModule, self).__init__(mpstate, "vicon", "vicon", public=False)
        self.console.set_status('VPos', 'VPos -- -- --', row=5)
        self.console.set_status('VAtt', 'VAtt -- -- --', row=5)
        self.vicon_settings = mp_settings.MPSettings(
            [('host', str, "vicon"),
             ('origin_lat', float, -35.363261),
             ('origin_lon', float, 149.165230),
             ('origin_alt', float, 584.0),
             ('vision_rate', int, 14),
             ('vel_filter_hz', float, 30.0),
             ('gps_rate', int, 5),
             ('gps_nsats', float, 16),
             ('object_name', str, None),
             ('save_init_pos', str, None),
             ('init_x', float, 0.0),
             ('init_y', float, 0.0),
             ('init_z', float, 0.0),
             ('mode', str, "unique_marker"), # same_marker, single_marker, unique_marker
             ])
        self.add_command('vicon', self.cmd_vicon, 'VICON control',
                         ["<start>",
                          "<stop>",
                          "set (VICONSETTING)"])
        self.add_completion_function('(VICONSETTING)',
                                     self.vicon_settings.completion)
        self.vicon = None
        self.thread = threading.Thread(target=self.thread_loop)
        self.thread.start()
        self.pos = None
        self.att = None
        self.frame_count = 0
        self.gps_count = 0
        self.vision_count = 0
        self.last_frame_count = 0
        self.vel_filter = LowPassFilter2p.LowPassFilter2p(200.0, 30.0)
        self.actual_frame_rate = 0.0
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.last_pose = np.eye(4)
        self.tracker = None
        self.pose_function = None

    def detect_vicon_object(self):
        # self.vicon.get_frame()
        self.vicon.waitForNextFrame()
        object_name = self.vicon_settings.object_name
        if object_name is None:
            # We haven't specified which object we are looking for, so just find the first one
            object_name = next(iter(self.vicon.rigidBodies), None)
        if object_name is None:
            # No objects found
            return None, None
        # segment_name = self.vicon.get_subject_root_segment_name(object_name)
        segment_name = object_name
        # if segment_name is None:
            # Object we're looking for can't be found
            # return None, None

        if self.vicon_settings.save_init_pos:
            rigid_body = self.vicon.rigidBodies.get(object_name)
            vicon_pos = rigid_body.position
            x, y, z = vicon_pos
            with open(self.vicon_settings.save_init_pos, "w") as f:
                json.dump([x, y, z], f)
                print(f"Initial coords saved to {self.vicon_settings.save_init_pos}")

        print("Connected to subject '%s' segment '%s'" % (object_name, segment_name))

        return object_name, segment_name

    def get_vicon_pose(self, object_name):
        # position is x (forward) y (left) z (up)
        rigid_body = self.vicon.rigidBodies.get(object_name)
        if rigid_body is None:
            # Object is not in view
            return None, None, None, None

        vicon_pos = rigid_body.position
        forward, left, up = vicon_pos
        vicon_pos = np.array([forward, -left, -up])  # NED
        vicon_quat = rigid_body.rotation

        pos_ned = Vector3(vicon_pos)  # position in meter
        euler = quaternion_to_euler(vicon_quat.x, -vicon_quat.y, -vicon_quat.z, vicon_quat.w)
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        yaw = math.radians(mavextra.wrap_360(math.degrees(yaw)))

        return pos_ned, roll, pitch, yaw

    def get_vicon_translation_pointcloud(self, object_name):
        pointcloud = self.vicon.pointCloud

        if pointcloud is None or len(pointcloud) == 0:
            # Object is not in view
            return None, None, None, None

        closest_idx = find_closest_cpu(pointcloud, self.last_position)
        vicon_pos = pointcloud[closest_idx]
        self.last_position = vicon_pos

        forward, left, up = vicon_pos
        vicon_pos = np.array([forward, -left, -up])  # NED
        pos_ned = Vector3(vicon_pos)  # position in meter

        return pos_ned, float('nan'), float('nan'), float('nan')

    def get_vicon_pose_pointcloud(self, object_name):
        pointcloud = self.vicon.pointCloud

        if pointcloud is None or len(pointcloud) == 0:
            # Object is not in view
            return None, None, None, None

        tracker.set_initial_pose(self.last_pose)
        estimated_pose = tracker.track(pointcloud, max_dist=0.2)
        self.last_pose = estimated_pose

        vicon_pos, vicon_quat = matrix_to_vicon_format(estimated_pose)
        self.last_position = vicon_pos

        forward, left, up = vicon_pos
        vicon_pos = np.array([forward, -left, -up])  # NED
        pos_ned = Vector3(vicon_pos)  # position in meter

        euler = quaternion_to_euler(vicon_quat[0], -vicon_quat[1], -vicon_quat[2], vicon_quat[3])
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        yaw = math.radians(mavextra.wrap_360(math.degrees(yaw)))

        return pos_ned, roll, pitch, yaw

    def thread_loop(self):
        """background processing"""
        if self.vicon_settings.mode == "single_marker":
            self.last_position = np.array(
                [self.vicon_settings.init_x, self.vicon_settings.init_y, self.vicon_settings.init_z])
            self.pose_function = self.get_vicon_translation_pointcloud
        elif self.vicon_settings.mode == "same_marker":
            from numba import njit
            from scipy.spatial.transform import Rotation as sciR

            markers = np.array([
                [0.00, -0.035, 0.00],
                [0.035, 0.00, 0.00],
                [0.00, 0.035, 0.00],
                [-0.005, 0.005, 0.00]
            ], dtype=np.float32)
            self.tracker = RigidBodyTracker(markers)
            self.last_pose = np.eye(4)
            self.last_pose[0, 3] = self.last_position[0]
            self.last_pose[1, 3] = self.last_position[1]
            self.last_pose[2, 3] = self.last_position[2]
            self.pose_function = self.get_vicon_pose_pointcloud
        else:
            self.pose_function = self.get_vicon_pose

        object_name = None
        segment_name = None
        last_pos = None
        last_frame_num = None
        frame_count = 0
        frame_num = 0

        while True:
            if self.vicon is None:
                time.sleep(0.1)
                object_name = None
                continue

            if not object_name:
                object_name, segment_name = self.detect_vicon_object()
                if object_name is None:
                    continue
                last_msg_time = time.time()
                now = time.time()
                last_origin_send = now
                now_ms = int(now * 1000)
                last_gps_send_ms = now_ms
                # frame_rate = self.vicon.get_frame_rate()
                frame_rate = 100
                frame_dt = 1.0/frame_rate
                last_rate = time.time()
                frame_count = 0
                print("Vicon frame rate %.1f" % frame_rate)

            if self.vicon_settings.gps_rate > 0:
                gps_period_ms = 1000 // self.vicon_settings.gps_rate
            time.sleep(0.01)
            # self.vicon.get_frame()
            self.vicon.waitForNextFrame()
            mav = self.master
            now = time.time()
            now_ms = int(now * 1000)
            frame_num += 1
            # frame_num = self.vicon.get_frame_number()

            frame_count += 1
            if now - last_rate > 0.1:
                rate = frame_count / (now - last_rate)
                self.actual_frame_rate = 0.9 * self.actual_frame_rate + 0.1 * rate
                last_rate = now
                frame_count = 0
                self.vel_filter.set_cutoff_frequency(self.actual_frame_rate, self.vicon_settings.vel_filter_hz)

            # pos_ned, roll, pitch, yaw = self.get_vicon_pose(object_name, segment_name)

            # pos_ned, roll, pitch, yaw = self.get_vicon_translation_pointcloud()
            pos_ned, roll, pitch, yaw = self.get_vicon_pose(object_name)

            if pos_ned is None:
                continue
            
            # print(f"XYZ: {pos_ned.x}, {pos_ned.y}, {pos_ned.z}, ")

            if last_frame_num is None or frame_num - last_frame_num > 100 or frame_num <= last_frame_num:
                last_frame_num = frame_num
                last_pos = pos_ned
                continue

            dt = (frame_num - last_frame_num) * frame_dt
            vel = (pos_ned - last_pos) * (1.0/dt)
            last_pos = pos_ned
            last_frame_num = frame_num

            filtered_vel = self.vel_filter.apply(vel)

            if self.vicon_settings.vision_rate > 0:
                dt = now - last_msg_time
                if dt < 1.0 / self.vicon_settings.vision_rate:
                    continue

            last_msg_time = now

            self.pos = pos_ned
            self.att = [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]
            self.frame_count += 1

            time_us = int(now * 1.0e6)

            if now - last_origin_send > 1 and self.vicon_settings.vision_rate > 0:
                # send a heartbeat msg
                mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0)

                # send origin at 1Hz
                mav.mav.set_gps_global_origin_send(self.target_system,
                                                   int(self.vicon_settings.origin_lat*1.0e7),
                                                   int(self.vicon_settings.origin_lon*1.0e7),
                                                   int(self.vicon_settings.origin_alt*1.0e3),
                                                   time_us)
                last_origin_send = now

            if self.vicon_settings.gps_rate > 0 and now_ms - last_gps_send_ms > gps_period_ms:
                '''send GPS data at the specified rate, trying to align on the given period'''
                self.gps_input_send(now, pos_ned, yaw, filtered_vel)
                last_gps_send_ms = (now_ms//gps_period_ms) * gps_period_ms
                self.gps_count += 1

            if self.vicon_settings.vision_rate > 0:
                # send VISION_POSITION_ESTIMATE
                # we force mavlink1 to avoid the covariances which seem to make the packets too large
                # for the mavesp8266 wifi bridge
                mav.mav.global_vision_position_estimate_send(time_us,
                                                             pos_ned.x, pos_ned.y, pos_ned.z,
                                                             roll, pitch, yaw, force_mavlink1=True)
                self.vision_count += 1

    def gps_input_send(self, time, pos_ned, yaw, gps_vel):
        time_us = int(time * 1.0e6)

        gps_lat, gps_lon = mavextra.gps_offset(self.vicon_settings.origin_lat,
                                               self.vicon_settings.origin_lon,
                                               pos_ned.y, pos_ned.x)
        gps_alt = self.vicon_settings.origin_alt - pos_ned.z
        gps_week, gps_week_ms = mp_util.get_gps_time(time)
        if self.vicon_settings.gps_nsats >= 6:
            fix_type = 3
        else:
            fix_type = 1

        if math.isnan(yaw):
            yaw_cd = 0
        else:
            yaw_cd = int(mavextra.wrap_360(math.degrees(yaw)) * 100)
            if yaw_cd == 0:
                # the yaw extension to GPS_INPUT uses 0 as no yaw support
                yaw_cd = 36000

        self.master.mav.gps_input_send(time_us, 0, 0, gps_week_ms, gps_week, fix_type,
                               int(gps_lat * 1.0e7), int(gps_lon * 1.0e7), gps_alt,
                               1.0, 1.0,
                               gps_vel.x, gps_vel.y, gps_vel.z,
                               0.2, 1.0, 1.0,
                               self.vicon_settings.gps_nsats,
                               yaw_cd)

    def cmd_start(self):
        """start vicon"""
        # remove pyvicon dependency
        # vicon = pyvicon.PyVicon()
        print("Opening Vicon connection to %s" % self.vicon_settings.host)
        vicon = motioncapture.connect("vicon", {'hostname': self.vicon_settings.host})
        # vicon.connect(self.vicon_settings.host)
        # print("Configuring vicon")
        # vicon.set_stream_mode(pyvicon.StreamMode.ClientPull)
        # vicon.enable_marker_data()
        # vicon.enable_segment_data()
        # vicon.enable_unlabeled_marker_data()
        # vicon.enable_device_data()

        # Set the axis mapping to the ardupilot convention (North, East, Down)
        # vicon.set_axis_mapping(pyvicon.Direction.Forward, pyvicon.Direction.Right, pyvicon.Direction.Down)
        # print(vicon.get_axis_mapping())
        print("vicon ready")
        self.vicon = vicon

    def cmd_vicon(self, args):
        """command processing"""
        if len(args) == 0:
            print("Usage: vicon <set|start|stop>")
            return
        if args[0] == "start":
            self.cmd_start()
        if args[0] == "stop":
            self.vicon = None
        elif args[0] == "set":
            self.vicon_settings.command(args[1:])

    def idle_task(self):
        """run on idle"""
        if not self.pos or not self.att or self.frame_count == self.last_frame_count:
            return
        self.last_frame_count = self.frame_count
        self.console.set_status('VPos', 'Vicon: Pos: %.2fN %.2fE %.2fD' % (self.pos.x, self.pos.y, self.pos.z), row=5)
        self.console.set_status('VAtt', ' Att R:%.2f P:%.2f Y:%.2f GPS %u VIS %u RATE %.1f' % (self.att[0], self.att[1], self.att[2],
                                                                                         self.gps_count, self.vision_count,
                                                                                         self.actual_frame_rate), row=5)


def init(mpstate):
    """initialise module"""
    return ViconModule(mpstate)
