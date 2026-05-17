#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float64.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <atomic>
#include <cmath>
#include <vector>
#include <string>

using std::size_t;

static std::atomic<bool> calib_ready(false);

// pubs
ros::Publisher pitch_pub;
ros::Publisher yaw_pub;
ros::Publisher z_pub;  // forward distance (X_base)
ros::Publisher x_pub;  // lateral offset  (Y_base)

// intrinsics
cv::Mat cameraMatrix, distCoeffs;
// aruco
cv::Ptr<cv::aruco::Dictionary> dictionary;
cv::Ptr<cv::aruco::DetectorParameters> detectorParams;

// ==== Extrinsics (ROS params) ====
bool    use_optical_to_base = true;          // optical->base 고정 회전 사용
cv::Vec3d cam_rpy_deg(0.0,0.0,0.0);          // 카메라 장착 RPY(deg)
cv::Vec3d cam_xyz(0.0,0.0,0.0);              // base_link에서 카메라 위치(m)

// Optical -> Base 고정 회전
// base_link: x forward, y left, z up
// optical  : x right,   y down, z forward
// 매핑: x_b =  z_c,  y_b = -x_c,  z_b = -y_c
static cv::Mat R_optical_to_base()
{
    return (cv::Mat_<double>(3,3) <<
         0,  0,  1,
        -1,  0,  0,
         0, -1,  0
    );
}

// RPY(deg) → 회전행렬 (Z*Y*X)
static cv::Mat R_from_rpy_deg(const cv::Vec3d& rpy_deg)
{
    const double r = rpy_deg[0] * CV_PI/180.0;
    const double p = rpy_deg[1] * CV_PI/180.0;
    const double y = rpy_deg[2] * CV_PI/180.0;

    cv::Mat Rz = (cv::Mat_<double>(3,3) <<
        std::cos(y), -std::sin(y), 0,
        std::sin(y),  std::cos(y), 0,
        0,            0,           1);

    cv::Mat Ry = (cv::Mat_<double>(3,3) <<
        std::cos(p), 0, std::sin(p),
        0,           1, 0,
        -std::sin(p),0, std::cos(p));

    cv::Mat Rx = (cv::Mat_<double>(3,3) <<
        1, 0,          0,
        0, std::cos(r),-std::sin(r),
        0, std::sin(r), std::cos(r));

    return Rz * Ry * Rx;
}

// CameraInfo에서 K, D 수신
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

static inline void rvecToEuler_cam(const cv::Vec3d& rvec, double& roll, double& pitch, double& yaw)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // OpenCV Rodrigues 기준
    pitch = std::atan2(-R.at<double>(2, 0),
                       std::sqrt(R.at<double>(0, 0)*R.at<double>(0, 0) +
                                 R.at<double>(1, 0)*R.at<double>(1, 0)));
    roll  = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    yaw   = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));

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

    // ArUco 검출
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<cv::Vec3d> rvecs, tvecs;
    const float markerLength = 0.06f; // m

    cv::aruco::detectMarkers(undistorted, dictionary, corners, ids, detectorParams);

    // optical→base 전체 회전, 평행이동
    cv::Mat R_cb = cv::Mat::eye(3,3,CV_64F);
    if (use_optical_to_base) R_cb = R_optical_to_base();
    cv::Mat R_off   = R_from_rpy_deg(cam_rpy_deg);
    cv::Mat R_total = R_cb * R_off;
    cv::Mat T_base_cam = (cv::Mat_<double>(3,1) << cam_xyz[0], cam_xyz[1], cam_xyz[2]);

    // 화면 중앙점 (파란 점)
    cv::Point img_center(undistorted.cols/2, undistorted.rows/2);
    cv::circle(undistorted, img_center, 4, cv::Scalar(255,0,0), -1);

    // 표시용 변수(첫 번째 마커 기준)
    double disp_pos_x = 0, disp_pos_y = 0, disp_pos_z = 0;   // base 기준
    double disp_roll = 0, disp_pitch = 0, disp_yaw = 0;      // 카메라 기준 RPY

    if (!ids.empty()) {
        cv::aruco::drawDetectedMarkers(undistorted, corners, ids);

        // ★ Pose 추정 (이게 있어야 rvecs/tvecs가 채워집니다)
        cv::aruco::estimatePoseSingleMarkers(
            corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

        for (size_t i = 0; i < ids.size(); ++i) {
            // 축(카메라 영상) 그리기
            cv::aruco::drawAxis(undistorted, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03);

            // 카메라 좌표 → 로봇(base_link) 좌표
            cv::Mat Pc = (cv::Mat_<double>(3,1) << tvecs[i][0], tvecs[i][1], tvecs[i][2]);
            cv::Mat Pb = R_total * Pc + T_base_cam;

            const double Xb = Pb.at<double>(0); // 전방(+)
            const double Yb = Pb.at<double>(1); // 좌(+)
            const double Zb = Pb.at<double>(2); // 위(+)

            // 퍼블리시: pose_z=전방(Xb), pose_x=좌우(Yb)
            std_msgs::Float64 m;
            m.data = Xb; z_pub.publish(m);
            m.data = Yb; x_pub.publish(m);

            // 각도 계산 & 퍼블리시 (deg, [-180,180])
            rvecToEuler_cam(rvecs[i], disp_roll, disp_pitch, disp_yaw);
            std_msgs::Float64 yaw_msg;   yaw_msg.data   = disp_yaw;   yaw_pub.publish(yaw_msg);
            std_msgs::Float64 pitch_msg; pitch_msg.data = disp_pitch; pitch_pub.publish(pitch_msg);

            // 첫 마커 값을 하단 표시에 사용
            if (i == 0) {
                disp_pos_x = Xb;
                disp_pos_y = Yb;
                disp_pos_z = Zb;
            }

            ROS_INFO("ID:%d | base Pos: [X=%.3f, Y=%.3f, Z=%.3f] | cam tvec:[%.3f, %.3f, %.3f] | Rot(R,P,Y)=[%.1f, %.1f, %.1f]",
                     ids[i], Xb, Yb, Zb, tvecs[i][0], tvecs[i][1], tvecs[i][2],
                     disp_roll, disp_pitch, disp_yaw);
        }
    }

    // ===== 중앙 하단에 2행×3열 텍스트 표시 =====
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.6;
    int thickness = 1;

    char line1[128], line2[128];
    std::snprintf(line1, sizeof(line1), "Pos: X=%.2f  Y=%.2f  Z=%.2f",
                  disp_pos_x, disp_pos_y, disp_pos_z);
    std::snprintf(line2, sizeof(line2), "Rot:  R=%.1f  P=%.1f  Y=%.1f",
                  disp_roll, disp_pitch, disp_yaw);

    cv::Point base_pt(img_center.x - 220, img_center.y + 40);
    cv::putText(undistorted, line1, base_pt,                  fontFace, fontScale, cv::Scalar(0,255,255), thickness, cv::LINE_AA);
    cv::putText(undistorted, line2, base_pt + cv::Point(0,24),fontFace, fontScale, cv::Scalar(0,255,255), thickness, cv::LINE_AA);

    cv::imshow("ArUco Marker Detection", undistorted);
    cv::waitKey(1);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "aruco_detector");
    ros::NodeHandle pnh("~");   // private ns (파라미터)
    ros::NodeHandle nh;         // pub/sub

    // params
    pnh.param("use_optical_to_base", use_optical_to_base, true);

    std::vector<double> rpy_deg_vec, xyz_vec;
    if (pnh.getParam("cam_rpy_deg", rpy_deg_vec) && rpy_deg_vec.size()==3)
        cam_rpy_deg = cv::Vec3d(rpy_deg_vec[0], rpy_deg_vec[1], rpy_deg_vec[2]);
    if (pnh.getParam("cam_xyz", xyz_vec) && xyz_vec.size()==3)
        cam_xyz = cv::Vec3d(xyz_vec[0], xyz_vec[1], xyz_vec[2]);

    // aruco
    dictionary     = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detectorParams = cv::aruco::DetectorParameters::create();

    // 구독 (런치에서 remap 권장)
    ros::Subscriber sub_img  = pnh.subscribe("image", 1, imageCallback);        // 예: /usb_cam/image_rect_color 로 remap
    ros::Subscriber sub_info = pnh.subscribe("camera_info", 1, camInfoCallback); // 예: /usb_cam/camera_info 로 remap

    // 퍼블리시 (전역 네임스페이스)
    pitch_pub = nh.advertise<std_msgs::Float64>("aruco/pitch", 10);
    yaw_pub   = nh.advertise<std_msgs::Float64>("aruco/yaw", 10);
    z_pub     = nh.advertise<std_msgs::Float64>("aruco/pose_z", 10); // forward (X_base)
    x_pub     = nh.advertise<std_msgs::Float64>("aruco/pose_x", 10); // lateral (Y_base)

    ROS_INFO_STREAM("ArUco Detector Node (cam->robot transform). "
                    << "use_optical_to_base=" << (use_optical_to_base?"true":"false")
                    << ", cam_rpy_deg=[" << cam_rpy_deg[0] << "," << cam_rpy_deg[1] << "," << cam_rpy_deg[2] << "]"
                    << ", cam_xyz=[" << cam_xyz[0] << "," << cam_xyz[1] << "," << cam_xyz[2] << "]");

    ros::spin();
    return 0;
}

