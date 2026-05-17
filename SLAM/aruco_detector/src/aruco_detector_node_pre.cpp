#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

cv::Ptr<cv::aruco::Dictionary> dictionary;
cv::Ptr<cv::aruco::DetectorParameters> detectorParams;

cv::Mat cameraMatrix, distCoeffs;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge error: %s", e.what());
        return;
    }

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(cv_ptr->image, dictionary, corners, ids, detectorParams);

    if (!ids.empty()) {
        cv::aruco::drawDetectedMarkers(cv_ptr->image, corners, ids);

        std::vector<cv::Vec3d> rvecs, tvecs;
        	// 마커 바꾸면 수정해야함 //
        float markerLength = 0.01; // 마커 한 칸의 한 변의 길이 (단위: meter)
        cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

        for (size_t i = 0; i < ids.size(); ++i) {
            cv::aruco::drawAxis(cv_ptr->image, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03);
            ROS_INFO("ID: %d | Position: [%.2f, %.2f, %.2f]", ids[i], tvecs[i][0], tvecs[i][1], tvecs[i][2]);
        }
    }

    cv::imshow("Aruco Detection", cv_ptr->image);
    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "aruco_detector");
    ros::NodeHandle nh;

	// 마커 바꾸면 수정해야함 //
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detectorParams = cv::aruco::DetectorParameters::create();

    // 카메라 파라미터 불러오기
    cv::FileStorage fs("/home/inwoong/catkin_ws/src/aruco_detector/config/camera.yaml", cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    ros::Subscriber image_sub = nh.subscribe("/usb_cam/image_raw", 1, imageCallback);

    ros::spin();
    return 0;
}

