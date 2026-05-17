#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <image_transport/image_transport.h>
#include <std_msgs/Float64.h>

#include <cmath>  // atan2 사용

ros::Publisher pitch_pub;
ros::Publisher yaw_pub;
ros::Publisher z_pub;
cv::Mat cameraMatrix, distCoeffs;
cv::Ptr<cv::aruco::Dictionary> dictionary;
cv::Ptr<cv::aruco::DetectorParameters> detectorParams;
ros::Publisher x_pub;





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
    
    if (yaw >= 180.0) yaw -= 360.0;
    else if (yaw < -180.0) yaw += 360.0;
    
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (cameraMatrix.empty() || distCoeffs.empty()) {
        ROS_ERROR("Camera parameters not loaded. Exiting...");
        return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge error: %s", e.what());
        return;
    }

    cv::Mat newCamMat = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, cv_ptr->image.size(), 1, cv_ptr->image.size(), 0);
    cv::Mat undistorted;
    cv::undistort(cv_ptr->image, undistorted, cameraMatrix, distCoeffs, newCamMat);

    // 언샤프 마스크 적용
    cv::Mat blurred, sharpened;
    cv::GaussianBlur(undistorted, blurred, cv::Size(0, 0), 3);
    cv::addWeighted(undistorted, 1.3, blurred, -0.3, 0, sharpened);  // sharped → sharpened

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<cv::Vec3d> rvecs, tvecs;
    float markerLength = 0.06;

    // ★ 선명해진 이미지로 마커 검출
    cv::aruco::detectMarkers(sharpened, dictionary, corners, ids, detectorParams);

    if (!ids.empty()) {
        drawDetectedMarkersWithInfo(sharpened, corners, ids);  // undistorted → sharpened
        cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

        for (size_t i = 0; i < ids.size(); ++i) {
            cv::aruco::drawAxis(sharpened, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03);

            double roll, pitch, yaw;
            rvecToEuler(rvecs[i], roll, pitch, yaw);

            std_msgs::Float64 pitch_msg;
            pitch_msg.data = pitch;
            pitch_pub.publish(pitch_msg);

            std_msgs::Float64 yaw_msg;
            yaw_msg.data = yaw;
            yaw_pub.publish(yaw_msg);

            std_msgs::Float64 z_msg;
            z_msg.data = tvecs[i][2];
            z_pub.publish(z_msg);
            
            std_msgs::Float64 x_msg;
            x_msg.data = tvecs[i][0];
            x_pub.publish(x_msg);     

            ROS_INFO("ID: %d | Pos: [%.2f, %.2f, %.2f] | Rot: [R: %.1f°, P: %.1f°, Y: %.1f°]",
                     ids[i], tvecs[i][0], tvecs[i][1], tvecs[i][2], roll, pitch, yaw);
        }
    }

    // 중심점은 sharpened에 표시
    cv::Point center(sharpened.cols / 2, sharpened.rows / 2);
    cv::circle(sharpened, center, 5, cv::Scalar(255, 0, 0), -1);

    // ★ 윈도우 출력은 sharpened로
    cv::imshow("ArUco Marker Detection", sharpened);
    cv::waitKey(10);
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "aruco_detector");
    ros::NodeHandle nh;
    x_pub = nh.advertise<std_msgs::Float64>("/aruco/pose_x", 10);  // 전역 변수 사용


    std::string calibration_file = "/home/inwoong/catkin_ws/src/aruco_detector/config/camera.yaml";
    loadCameraParameters(calibration_file);

    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detectorParams = cv::aruco::DetectorParameters::create();

    ros::Subscriber image_sub = nh.subscribe("/usb_cam/image_raw", 1, imageCallback);
    
    pitch_pub = nh.advertise<std_msgs::Float64>("/aruco/pitch", 10);
    yaw_pub = nh.advertise<std_msgs::Float64>("/aruco/yaw", 10);
    z_pub = nh.advertise<std_msgs::Float64>("/aruco/pose_z", 10);

    ROS_INFO("ArUco Detector Node Started");
    ros::spin();
    return 0;
}

