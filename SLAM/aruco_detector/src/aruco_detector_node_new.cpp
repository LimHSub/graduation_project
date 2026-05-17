#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float64.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <atomic>
#include <cmath>

using std::size_t;

static std::atomic<bool> calib_ready(false);

ros::Publisher pitch_pub;
ros::Publisher yaw_pub;
ros::Publisher z_pub;
ros::Publisher x_pub;

cv::Mat cameraMatrix, distCoeffs;
cv::Ptr<cv::aruco::Dictionary> dictionary;
cv::Ptr<cv::aruco::DetectorParameters> detectorParams;

// CameraInfo에서 K, D 읽기
void camInfoCallback(const sensor_msgs::CameraInfoConstPtr& info)
{
    cameraMatrix = (cv::Mat_<double>(3,3) <<
        info->K[0], info->K[1], info->K[2],
        info->K[3], info->K[4], info->K[5],
        info->K[6], info->K[7], info->K[8]);

    distCoeffs = cv::Mat(1, (int)info->D.size(), CV_64F);
    for (size_t i = 0; i < info->D.size(); ++i)
        distCoeffs.at<double>(0, (int)i) = info->D[i];

    calib_ready = true;
}

static inline void rvecToEuler(const cv::Vec3d& rvec, double& roll, double& pitch, double& yaw)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    pitch = atan2(-R.at<double>(2, 0),
                  std::sqrt(R.at<double>(0, 0)*R.at<double>(0, 0) +
                            R.at<double>(1, 0)*R.at<double>(1, 0)));
    roll  = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    yaw   = atan2(R.at<double>(2, 1), R.at<double>(2, 2));

    roll  *= 180.0 / CV_PI;
    pitch *= 180.0 / CV_PI;
    yaw   *= 180.0 / CV_PI;

    if (yaw >= 180.0) yaw -= 360.0;
    else if (yaw < -180.0) yaw += 360.0;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if (!calib_ready) {
        ROS_WARN_THROTTLE(2.0, "Waiting for camera_info...");
        return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge error: %s", e.what());
        return;
    }

    // 왜곡 보정
    cv::Mat undistorted;
    cv::undistort(cv_ptr->image, undistorted, cameraMatrix, distCoeffs);

    // 마커 검출
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<cv::Vec3d> rvecs, tvecs;
    const float markerLength = 0.06f; // 실제 마커 변 길이(m)

    cv::aruco::detectMarkers(undistorted, dictionary, corners, ids, detectorParams);
    
    if (!ids.empty()) {
    cv::aruco::drawDetectedMarkers(undistorted, corners, ids);
    cv::aruco::estimatePoseSingleMarkers(
        corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

    const float axisLen = 0.03f; // 축 길이 3cm

    for (size_t i = 0; i < ids.size(); ++i) {
        // 축 그리기
        cv::aruco::drawAxis(undistorted, cameraMatrix, distCoeffs,
                            rvecs[i], tvecs[i], axisLen);

        double roll, pitch, yaw;
        rvecToEuler(rvecs[i], roll, pitch, yaw);

        // 퍼블리시
        std_msgs::Float64 m;
        m.data = pitch;      pitch_pub.publish(m);
        m.data = yaw;        yaw_pub.publish(m);
        m.data = tvecs[i][2]; z_pub.publish(m);
        m.data = tvecs[i][0]; x_pub.publish(m);

        ROS_INFO("ID: %d | Pos: [%.2f, %.2f, %.2f] | Rot: [R: %.1f°, P: %.1f°, Y: %.1f°]",
                 ids[i], tvecs[i][0], tvecs[i][1], tvecs[i][2], roll, pitch, yaw);

        // ======== 중점 좌표 계산 ========
        cv::Point2f center(0,0);
        for (const auto& pt : corners[i]) center += pt;
        center *= (1.0f / 4.0f);

        // ======== 표시할 문자열 생성 ========
        char pos_text[50], rot_text[50];
        snprintf(pos_text, sizeof(pos_text), "Pos: %.2f %.2f %.2f",
                 tvecs[i][0], tvecs[i][1], tvecs[i][2]);
        snprintf(rot_text, sizeof(rot_text), "Rot: %.1f %.1f %.1f",
                 roll, pitch, yaw);

        // ======== 글자 표시 ========
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;

        cv::putText(undistorted, pos_text, center + cv::Point2f(-60, 20),
                    fontFace, fontScale, cv::Scalar(0,255,255), thickness);
        cv::putText(undistorted, rot_text, center + cv::Point2f(-60, 40),
                    fontFace, fontScale, cv::Scalar(0,255,255), thickness);
    }
}
    // 카메라 중심점 표시 (파란색 점)
    cv::Point center(undistorted.cols / 2, undistorted.rows / 2);
    cv::circle(undistorted, center, 4, cv::Scalar(255, 0, 0), -1);  // radius=4, filled

    cv::imshow("ArUco Marker Detection", undistorted);
    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "aruco_detector");
    ros::NodeHandle nh;

    dictionary     = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detectorParams = cv::aruco::DetectorParameters::create();

    // 상대 토픽명 (런치에서 remap 권장)
    ros::Subscriber sub_img  = nh.subscribe("image", 1, imageCallback);
    ros::Subscriber sub_info = nh.subscribe("camera_info", 1, camInfoCallback);

    pitch_pub = nh.advertise<std_msgs::Float64>("aruco/pitch", 10);
    yaw_pub   = nh.advertise<std_msgs::Float64>("aruco/yaw", 10);
    z_pub     = nh.advertise<std_msgs::Float64>("aruco/pose_z", 10);
    x_pub     = nh.advertise<std_msgs::Float64>("aruco/pose_x", 10);

    ROS_INFO("ArUco Detector Node Started (CameraInfo-based calibration).");
    ros::spin();
    return 0;
}

