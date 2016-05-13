#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <jsk_apc2016_common/SegmentationInBinSync.h>
#include <ros/ros.h>
#include <sstream>

using namespace sensor_msgs;
using namespace message_filters;

Image color_img;
Image dist_img;
Image height_img;
Image mask_img;

ros::Publisher pub_;

void callback(const ImageConstPtr& color_img, const ImageConstPtr& dist_img, const ImageConstPtr& height_img, const ImageConstPtr& mask_img)
{
    jsk_apc2016_common::SegmentationInBinSync sync_data;
    sync_data.color_img = *color_img;
    sync_data.dist_img  = *dist_img;
    sync_data.height_img = *height_img;
    sync_data.mask_img = *mask_img;
    pub_.publish(sync_data);
    ros::Duration(0.5).sleep();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rbo_segmentation_in_bin_sync");

    ros::NodeHandle nh("~");

    pub_ = nh.advertise<jsk_apc2016_common::SegmentationInBinSync>("output", 1);

    message_filters::Subscriber<Image> image_sub(nh, "input/image", /*queue_size*/1);
    message_filters::Subscriber<Image> dist_sub(nh, "input/dist", 1);
    message_filters::Subscriber<Image> height_sub(nh, "input/height", 1);
    message_filters::Subscriber<Image> mask_sub(nh, "input/mask", 1);

    typedef sync_policies::ApproximateTime<Image, Image, Image, Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), image_sub, dist_sub, height_sub, mask_sub);

    sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4));
    ros::spin();
    return 0;
}
