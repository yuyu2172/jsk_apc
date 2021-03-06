#!/usr/bin/env python

import rospy
from jsk_apc2016_common.msg import BinInfo, BinInfoArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import Pose
import jsk_apc2016_common.segmentation_in_bin.\
        segmentation_in_bin_helper as helper
import jsk_apc2016_common
from std_msgs.msg import Header

import json
import os


class BinInfoArrayPublisher(object):
    def __init__(self):
        self.bbox_dict = {}
        self.bin_contents_dict = {}
        self.targets_dict = {}
        self.cam_direction_dict = {}
        self.json_file = None

        pub_bin_info_arr = rospy.Publisher('~bin_array', BinInfoArray, queue_size=1)
        pub_bbox_arr = rospy.Publisher('~bbox_array', BoundingBoxArray, queue_size=1)
        rate = rospy.Rate(rospy.get_param('rate', 1))
        while not rospy.is_shutdown():
            json = rospy.get_param('~json', None)

            # update bin_info_arr only when rosparam: json is changd
            if self.json_file != json:
                if not os.path.isfile(json) or json[-4:] != 'json':
                    rospy.logwarn('wrong json file name')
                    rate.sleep()
                    continue
                self.json_file = json

                # get bbox from rosparam
                self.from_shelf_param('upper')
                self.from_shelf_param('lower')

                # create bounding box array
                self.bbox_array = self.get_bounding_box_array(self.bbox_dict)

                # get contents of bin from json
                self.bin_contents_dict = jsk_apc2016_common.get_bin_contents(self.json_file)
                self.targets_dict = jsk_apc2016_common.get_work_order(self.json_file)

                # create bin_msg
                self.create_bin_info_arr()

            self.bbox_array.header.stamp = rospy.Time.now()
            self.bin_info_arr.header.stamp = rospy.Time.now()
            pub_bbox_arr.publish(self.bbox_array)
            pub_bin_info_arr.publish(self.bin_info_arr)
            rate.sleep()

    def get_bounding_box_array(self, bbox_dict):
        bbox_list = [bbox_dict[bin_] for bin_ in 'abcdefghijkl']
        bbox_array = BoundingBoxArray(boxes=bbox_list)
        bbox_array.header.stamp = rospy.Time(0)
        bbox_array.header.seq = 0
        bbox_array.header.frame_id = 'kiva_pod_base'
        return bbox_array

    def from_shelf_param(self, upper_lower):
        upper_lower = upper_lower + '_shelf'
        initial_pos_list = rospy.get_param(
                '~' + upper_lower + '/initial_pos_list')
        initial_quat_list = rospy.get_param(
                '~' + upper_lower + '/initial_quat_list')
        dimensions = rospy.get_param(
                '~' + upper_lower + '/dimensions')
        frame_id_list = rospy.get_param(
                '~' + upper_lower + '/frame_id_list')
        prefixes = rospy.get_param(
                '~' + upper_lower + '/prefixes')
        camera_directions = rospy.get_param(
                '~' + upper_lower + '/camera_directions')

        for i, bin_ in enumerate(prefixes):
            bin_ = bin_.split('_')[1].lower()  # bin_A -> a
            header = Header(
                    stamp=rospy.Time.now(),
                    frame_id=frame_id_list[i])
            self.bbox_dict[bin_] = BoundingBox(
                    header=header,
                    pose=Pose(
                            position=helper.point(initial_pos_list[i]),
                            orientation=helper.quaternion(initial_quat_list[i])),
                    dimensions=helper.vector3(dimensions[i]))
            self.cam_direction_dict[bin_] = camera_directions[i]

    def create_bin_info_arr(self):
        self.bin_info_arr = BinInfoArray()
        for bin_ in 'abcdefghijkl':
            self.bin_info_arr.array.append(BinInfo(
                    header=Header(
                            stamp=rospy.Time(0),
                            seq=0,
                            frame_id='bin_'+bin_),
                    name=bin_,
                    objects=self.bin_contents_dict[bin_],
                    target=self.targets_dict[bin_],
                    bbox=self.bbox_dict[bin_],
                    camera_direction=self.cam_direction_dict[bin_]))


if __name__ == '__main__':
    rospy.init_node('set_bin_param')
    bin_publisher = BinInfoArrayPublisher()
    rate = rospy.Rate(rospy.get_param('rate', 1))
    while not rospy.is_shutdown():
        rate.sleep()
