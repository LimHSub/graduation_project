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

// ---- Extrinsic params (ROS param으로 세팅) ----
bool use_optical_to_base;                 // 기본 true
cv::Vec3d cam_rpy_deg(0.0, 0.0, 0.0);     // roll, pitch, yaw (deg) - 카메라 추가 보정
cv::Vec3d cam_xyz(0.0, 0.0, 0.0);         // base_link에서 본 카메라 위치 (m)

// R_optical_to_base (카메라가 정면/수평 설치 가정 시 고정 회전)
// base_link: x forward, y left, z up
// optical:   x right,   y down, z forward
// 매핑: x_b =  z_c,  y_b = -x_c,  z_b = -y_c
static cv::Mat R_optical_to_base()
{
    cv::Mat R = (cv::Mat_<double>(3,3) <<
        0,  0,  1,   // x_b = z_c
       -1,  0,  0,   // y_b = -x_c
        0, -1,  0    // z_b = -y_c
    );
    return R;
}

static cv::Mat R_from_rpy_deg(const cv::Vec3d& rpy_deg)
{
    const double r = rpy_deg[0] * CV_PI/180.0;
    const double p = rpy_deg[1] * CV_PI/180.0;
    const double y = rpy_deg[2] * CV_PI/180.0;

    // Z(yaw) * Y(pitch) * X(roll)
    cv::Mat Rz = (cv::Mat_<double>(3,3) <<
        cos(y), -sin(y), 0,
        sin(y),  cos(y), 0,
             0,       0, 1);

    cv::Mat Ry = (cv::Mat_<double>(3,3) <<
        cos(p), 0, sin(p),
            0, 1,     0,
       -sin(p),0, cos(p));

    cv::Mat Rx = (cv::Mat_<double>(3,3) <<
        1,      0,       0,
        0, cos(r), -sin(r),
        0, sin(r),  cos(r));

    return Rz * Ry * Rx;
}

// CameraInfo에서 K,D 설정
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
    const float markerLength = 0.06f;

    cv::aruco::detectMarkers(undistorted, dictionary, corners, ids, detectorParams);

    if (!ids.empty()) {
        cv::aruco::drawDetectedMarkers(undistorted, corners, ids);
        cv::aruco::estimatePoseSingleMarkers(corners, markerLength,
                                             cameraMatrix, distCoeffs,
                                             rvecs, tvecs);

        // ---- 카메라→로봇 변환 행렬 만들기 ----
        cv::Mat R_cb = cv::Mat::eye(3,3,CV_64F);
        if (use_optical_to_base) R_cb = R_optical_to_base();
        cv::Mat R_off = R_from_rpy_deg(cam_rpy_deg);
        cv::Mat R_total = R_cb * R_off; // optical→base 후 설치 보정

        cv::Mat T_base_cam = (cv::Mat_<double>(3,1) << cam_xyz[0], cam_xyz[1], cam_xyz[2]);

        for (size_t i = 0; i < ids.size(); ++i) {
            // 카메라 좌표계에서 마커까지 벡터
            cv::Mat Pc = (cv::Mat_<double>(3,1) << tvecs[i][0], tvecs[i][1], tvecs[i][2]);

            // 로봇(base_link) 좌표계로 변환
            cv::Mat Pb = R_total * Pc + T_base_cam;

            const double x_base = Pb.at<double>(0); // 전방
            const double y_base = Pb.at<double>(1); // 좌(+)/우(-)
            // const double z_base = Pb.at<double>(2); // 상(+)/하(-) — 필요 시 사용

            // 퍼블리시: pose_x= 횡방향(y_base), pose_z= 전방(x_base)
            std_msgs::Float64 m;
            m.data = x_base; z_pub.publish(m);
            m.data = y_base; x_pub.publish(m);

            // rvec은 카메라 기준의 회전이므로 그대로 사용(원하면 로봇 기준으로 합성 가능)
            double roll, pitch, yaw;
            rvecToEuler(rvecs[i], roll, pitch, yaw);
            m.data = pitch; pitch_pub.publish(m);
            m.data = yaw;   yaw_pub.publish(m);

            ROS_INFO("ID:%d | base Pos: [X=%.3f m, Y=%.3f m] | cam tvec:[%.3f, %.3f, %.3f]",
                     ids[i], x_base, y_base, tvecs[i][0], tvecs[i][1], tvecs[i][2]);

            // 축도 계속 그림(카메라 이미지 위)
            cv::aruco::drawAxis(undistorted, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03);
        }
    }
    cv::Point center(undistorted.cols / 2, undistorted.rows / 2);
    cv::circle(undistorted, center, 4, cv::Scalar(255, 0, 0), -1);

    cv::imshow("ArUco Marker Detection", undistorted);
    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "aruco_detector");
    ros::NodeHandle nh("~"); // private ns에서 파라미터 읽기

    // 파라미터 로드
    nh.param("use_optical_to_base", use_optical_to_base, true);

    std::vector<double> rpy_deg_vec, xyz_vec;
    if (nh.getParam("cam_rpy_deg", rpy_deg_vec) && rpy_deg_vec.size()==3)
        cam_rpy_deg = cv::Vec3d(rpy_deg_vec[0], rpy_deg_vec[1], rpy_deg_vec[2]);
    if (nh.getParam("cam_xyz", xyz_vec) && xyz_vec.size()==3)
        cam_xyz = cv::Vec3d(xyz_vec[0], xyz_vec[1], xyz_vec[2]);

    dictionary     = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detectorParams = cv::aruco::DetectorParameters::create();

    // 상대 토픽명(런치에서 remap 권장)
    ros::Subscriber sub_img  = nh.subscribe("image", 1, imageCallback);
    ros::Subscriber sub_info = nh.subscribe("camera_info", 1, camInfoCallback);

    // 퍼블리셔 (전역 네임스페이스로 뺄 거면 다른 NodeHandle 사용)
    ros::NodeHandle nh_pub;
    pitch_pub = nh_pub.advertise<std_msgs::Float64>("aruco/pitch", 10);
    yaw_pub   = nh_pub.advertise<std_msgs::Float64>("aruco/yaw", 10);
    z_pub     = nh_pub.advertise<std_msgs::Float64>("aruco/pose_z", 10); // 전방(X_base)
    x_pub     = nh_pub.advertise<std_msgs::Float64>("aruco/pose_x", 10); // 횡(Y_base)

    ROS_INFO("ArUco Detector Node Started (with cam->robot transform). use_optical_to_base=%s",
             use_optical_to_base ? "true":"false");
    ros::spin();
    return 0;
}

