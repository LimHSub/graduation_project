#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <image_transport/image_transport.h>

#include <cmath>  // atan2 사용

sensor_msgs::CameraInfo latest_cam_info;
bool camera_info_received = false;

cv::Mat cameraMatrix, distCoeffs;
cv::Ptr<cv::aruco::Dictionary> dictionary;
cv::Ptr<cv::aruco::DetectorParameters> detectorParams;

float markerLength = 0.06;  // 6cm (한 칸 1cm × 6칸 마커)

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    latest_cam_info = *msg;
    camera_info_received = true;
}

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

void rvecToEuler(const cv::Vec3d& rvec, double& roll, double& pitch, double& yaw) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);  // 3x3 회전 행렬로 변환

    // R에서 Euler angle 추출
    pitch = atan2(-R.at<double>(2, 0), sqrt(R.at<double>(0, 0)*R.at<double>(0, 0) + R.at<double>(1, 0)*R.at<double>(1, 0)));
    roll  = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    yaw   = atan2(R.at<double>(2, 1), R.at<double>(2, 2));

    // 라디안을 도(degree)로 변환
    roll  *= 180.0 / CV_PI;
    pitch *= 180.0 / CV_PI;
    yaw   *= 180.0 / CV_PI;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (!camera_info_received) return;

    cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;

    // camera matrix
    cv::Mat cam_matrix = cv::Mat(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat(1, 5, CV_64F);
    for (int i = 0; i < 9; i++) cam_matrix.at<double>(i/3, i%3) = latest_cam_info.K[i];
    for (int i = 0; i < 5; i++) dist_coeffs.at<double>(0, i) = latest_cam_info.D[i];

    // detect markers
    cv::Mat undistorted;
    cv::undistort(image, undistorted, cam_matrix, dist_coeffs);
    
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(undistorted, dictionary, corners, ids);

    if (!ids.empty()) {
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cam_matrix, dist_coeffs, rvecs, tvecs);

        for (int i = 0; i < ids.size(); ++i) {
            ROS_INFO("ID: %d | Pos: [%.2f, %.2f, %.2f] | Rot: [%.2f, %.2f, %.2f]",
                ids[i], tvecs[i][0], tvecs[i][1], tvecs[i][2],
                rvecs[i][0], rvecs[i][1], rvecs[i][2]);
        }
        drawDetectedMarkersWithInfo(undistorted, corners, ids);
    }

    cv::imshow("Image", undistorted);
    cv::waitKey(1);
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "aruco_detector_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    image_transport::Subscriber image_sub = it.subscribe("/usb_cam/image_raw", 1, imageCallback);
    ros::Subscriber camera_info_sub = nh.subscribe("/usb_cam/camera_info", 1, cameraInfoCallback);

    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    ros::spin();
    return 0;
}

