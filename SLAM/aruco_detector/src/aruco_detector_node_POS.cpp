#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <image_transport/image_transport.h>

cv::Mat cameraMatrix, distCoeffs;
cv::Ptr<cv::aruco::Dictionary> dictionary;
cv::Ptr<cv::aruco::DetectorParameters> detectorParams;

void loadCameraParameters(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        ROS_ERROR("Cannot open camera calibration file: %s", filename.c_str());
        ros::shutdown();
        return;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();
    ROS_INFO("Camera calibration loaded successfully.");
}

void drawDetectedMarkersWithInfo(cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& corners, const std::vector<int>& ids) {
    for (size_t i = 0; i < ids.size(); ++i) {
        cv::Point2f center(0, 0);
        for (int j = 0; j < 4; ++j)
            center += corners[i][j];
        center *= 0.25;

        cv::circle(image, center, 4, cv::Scalar(0, 255, 0), -1);
        cv::putText(image, std::to_string(ids[i]), corners[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
    cv::aruco::drawDetectedMarkers(image, corners, ids);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge error: %s", e.what());
        return;
    }

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<cv::Vec3d> rvecs, tvecs;
    float markerLength = 0.06; // 실제 마커 한 변 길이 (단위: meter)

    cv::aruco::detectMarkers(cv_ptr->image, dictionary, corners, ids, detectorParams);

    if (!ids.empty()) {
        drawDetectedMarkersWithInfo(cv_ptr->image, corners, ids);
        cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

        for (size_t i = 0; i < ids.size(); ++i) {
            cv::aruco::drawAxis(cv_ptr->image, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03);
            ROS_INFO("ID: %d | Pos: [%.2f, %.2f, %.2f]", ids[i], tvecs[i][0], tvecs[i][1], tvecs[i][2]);
        }
    }

    cv::imshow("ArUco Marker Detection", cv_ptr->image);
    cv::waitKey(1);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "aruco_detector");
    ros::NodeHandle nh;

    std::string calibration_file = "/home/inwoong/catkin_ws/src/aruco_detector/config/camera.yaml";
    loadCameraParameters(calibration_file);

    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detectorParams = cv::aruco::DetectorParameters::create();

    ros::Subscriber image_sub = nh.subscribe("/usb_cam/image_raw", 1, imageCallback);

    ROS_INFO("ArUco Detector Node Started");
    ros::spin();
    return 0;
}

