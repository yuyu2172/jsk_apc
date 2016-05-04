#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <jsk_apc2016_common/SegmentationInBinSync.h>
#include <std_msgs/String.h>
#include <ros/ros.h>

#include <sstream>

using namespace sensor_msgs;
using namespace message_filters;

Image stored_image;
CameraInfo stored_caminfo;
PointCloud2 stored_pc;

ros::Publisher image_pub;
ros::Publisher pub_;

void callback(const ImageConstPtr& image, const CameraInfoConstPtr& cam_info, const PointCloud2ConstPtr& points)
{
    ROS_INFO("%s", "hello");

    Image image_sent;
    image_sent.header = image->header;
    image_sent.height = image->height;
    image_sent.data = image->data;
//    image_pub.publish(image_sent);

    CameraInfo caminfo_sent;

    PointCloud2 points_sent;
    points_sent.data = points->data;

    jsk_apc2016_common::SegmentationInBinSync sync_data;
    sync_data.image_color = image_sent;
    sync_data.cam_info = caminfo_sent;
    sync_data.points = points_sent;
    pub_.publish(sync_data);
    ros::Duration(10).sleep();
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "segmentation_in_bin_sync");

  ros::NodeHandle nh("~");


  image_pub = nh.advertise<Image>("image", 1000);
  pub_ = nh.advertise<jsk_apc2016_common::SegmentationInBinSync>("output", 1);

  message_filters::Subscriber<Image> image_sub(nh, "/right_softkinetic_camera/rgb/image_color", 1);
  message_filters::Subscriber<CameraInfo> info_sub(nh, "/right_softkinetic_camera/rgb/camera_info", 1);
  message_filters::Subscriber<PointCloud2> point_sub(nh, "/right_softkinetic_camera/depth/points", 1);

  typedef sync_policies::ApproximateTime<Image, CameraInfo, PointCloud2> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), image_sub, info_sub, point_sub);

  
// TimeSynchronizer<Image, CameraInfo, PointCloud2> sync(image_sub, info_sub, point_sub, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));





  ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);
  ros::Rate loop_rate(10);
  int count = 0;
  while (ros::ok())
  {
    std_msgs::String msg;

    std::stringstream ss;
    ss << "hello world " << count;
    msg.data = ss.str();

    chatter_pub.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();
    count ++;
  }

  return 0;
}
