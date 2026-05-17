// src/aruco_detector_node.cpp
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <atomic>
#include <cmath>
#include <vector>
#include <string>
#include <deque>
#include <algorithm>
#include <cstdio>  // ★ std::snprintf

using std::size_t;

static std::atomic<bool> calib_ready(false);

// pubs
ros::Publisher pitch_pub;
ros::Publisher yaw_pub;            // 카메라 기준 yaw (옵션)
ros::Publisher yaw_b_m_pub;        // 안정화된 base 기준 마커 yaw
ros::Publisher yaw_b_m_raw_pub;    // 원시 base 기준 마커 yaw
ros::Publisher z_pub;              // forward distance (X_base)
ros::Publisher x_pub;              // lateral offset  (Y_base)
ros::Publisher marker_id_pub;   // ★ 마커 ID 퍼블리셔


// intrinsics
cv::Mat cameraMatrix, distCoeffs;
// aruco
cv::Ptr<cv::aruco::Dictionary> dictionary;
cv::Ptr<cv::aruco::DetectorParameters> detectorParams;

// ==== Extrinsics (ROS params) ====
bool    use_optical_to_base = true;
cv::Vec3d cam_rpy_deg(0.0,0.0,0.0);
cv::Vec3d cam_xyz(0.0,0.0,0.0);

// ---------- angle helpers & filters ----------
static inline double wrap180(double a){ a=fmod(a+180.0,360.0); if(a<0)a+=360.0; return a-180.0; }
static inline double angDiff(double a, double b){ return wrap180(a-b); }

// 원형 평균
static double circularMean(const std::deque<double>& ang_deg){
    if (ang_deg.empty()) return 0.0;
    double sx=0.0, sy=0.0;
    for(double a: ang_deg){
        double r = a * CV_PI/180.0;
        sx += std::cos(r);
        sy += std::sin(r);
    }
    return std::atan2(sy, sx) * 180.0 / CV_PI;
}

// EMA(각도 래핑 고려) + step limit
struct EmaYaw {
    bool init=false; double yaw=0.0; double a=0.25; double max_step_deg=2.0;
    void setParams(double alpha, double maxstep){
        a = std::min(0.99,std::max(0.0,alpha));
        max_step_deg = std::max(0.0,maxstep);
    }
    void seed(double yaw_meas){ yaw = wrap180(yaw_meas); init=true; }
    void updateTowards(double target_deg){
        if(!init){ seed(target_deg); return; }
        double d = angDiff(target_deg, yaw);
        double step = std::clamp(a*d, -max_step_deg, max_step_deg);
        yaw = wrap180(yaw + step);
    }
} gYaw;

struct YawOkJudge {
    double tol_in=1.0, tol_out=1.8;
    int ok_need=5, ok_cnt=0; bool aligned=false;
    bool update(double err_abs_deg){
        double th = aligned ? tol_out : tol_in;
        if (err_abs_deg <= th) ok_cnt = std::min(ok_need, ok_cnt+1);
        else { ok_cnt=0; aligned=false; }
        if (ok_cnt>=ok_need) aligned=true;
        return aligned;
    }
} gYawOk;

// ---------- 안정화 파라미터 ----------
double ema_alpha_yaw   = 0.25;
double ema_alpha_min   = 0.05;
double ema_alpha_max   = 0.35;
double reproj_good_px  = 1.0;
double reproj_bad_px   = 4.0;
int    median_win      = 5;
double max_step_deg    = 2.0;
double jump_reject_deg = 25.0; // (참고용)
bool   do_subpix       = true;
bool   use_ippe_square = true;

// 최근 raw yaw buffer
std::deque<double> yaw_buf;

// Optical -> Base 고정 회전
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

// 카메라 기준 RPY (디버깅/기존 호환용)
static inline void rvecToEuler_cam(const cv::Vec3d& rvec, double& roll, double& pitch, double& yaw)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

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

    // SubPix 정련 (옵션)
    if (do_subpix && !corners.empty()) {
        cv::Mat gray; cv::cvtColor(undistorted, gray, cv::COLOR_BGR2GRAY);
        for (auto& c : corners) {
            if (c.size()!=4) continue;
            cv::cornerSubPix(
                gray, c, cv::Size(5,5), cv::Size(-1,-1),
                cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 20, 0.01)
            );
        }
    }

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
    double disp_pos_x = 0, disp_pos_y = 0, disp_pos_z = 0;   // base 기준 위치
    double disp_roll = 0, disp_pitch = 0, disp_yaw = 0;      // 카메라 기준 RPY
    double disp_yaw_b_m_raw = 0.0;                           // ★ 화면에 표시할 RAW yaw
    int    disp_id = -1;                                     // ★ 화면에 표시할 마커 ID (없으면 -1)

    if (!ids.empty()) {
        cv::aruco::drawDetectedMarkers(undistorted, corners, ids);

        // Pose 추정
        cv::aruco::estimatePoseSingleMarkers(
            corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

        for (size_t i = 0; i < ids.size(); ++i) {
            if (corners[i].size()!=4) continue;

            // 축(카메라 영상) 그리기
            cv::aruco::drawAxis(undistorted, cameraMatrix, distCoeffs,
                                rvecs[i], tvecs[i], 0.03);

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

            // 카메라 기준 yaw(디버깅/호환)
            rvecToEuler_cam(rvecs[i], disp_roll, disp_pitch, disp_yaw);
            std_msgs::Float64 yaw_msg; yaw_msg.data = disp_yaw; yaw_pub.publish(yaw_msg);

            // base 기준 마커 yaw: R_bm = R_bc * R_cm
            cv::Mat R_cm;
            cv::Rodrigues(rvecs[i], R_cm);  // cam→marker 회전
            cv::Mat R_bm = R_total * R_cm;  // base→marker 회전
            double yaw_b_m_raw = std::atan2(R_bm.at<double>(1,0), R_bm.at<double>(0,0)) * 180.0 / CV_PI;

            // ---- 재투영 오차로 품질 계산 (평균 픽셀 오차) ----
            std::vector<cv::Point3f> obj;
            double L = markerLength;
            obj.emplace_back(-L/2,  L/2, 0);
            obj.emplace_back( L/2,  L/2, 0);
            obj.emplace_back( L/2, -L/2, 0);
            obj.emplace_back(-L/2, -L/2, 0);
            std::vector<cv::Point2f> proj;
            cv::projectPoints(obj, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, proj);
            double rms=0;
            for(int k=0;k<4;++k){
                cv::Point2f d = proj[k]-corners[i][k];
                rms += d.dot(d);
            }
            rms = std::sqrt(rms/4.0);

            // 1) RAW 항상 퍼블리시
            std_msgs::Float64 raw; raw.data = yaw_b_m_raw; yaw_b_m_raw_pub.publish(raw);

            // 2) 안정화(퍼블리시는 유지하되, 필터는 제어용으로만)
            yaw_buf.push_back(yaw_b_m_raw);
            int win = std::max(1, median_win);
            if ((int)yaw_buf.size() > win) yaw_buf.pop_front();

            double yaw_mean = circularMean(yaw_buf);
            double alpha = ema_alpha_yaw;
            if (rms <= reproj_good_px) alpha = ema_alpha_max;
            else if (rms >= reproj_bad_px) alpha = ema_alpha_min;
            else {
                double t = (rms - reproj_good_px) / (reproj_bad_px - reproj_good_px);
                t = std::clamp(t, 0.0, 1.0);
                alpha = ema_alpha_max*(1.0 - t) + ema_alpha_min*t;
            }
            gYaw.setParams(alpha, max_step_deg);
            if (!gYaw.init) gYaw.seed(yaw_mean);
            else gYaw.updateTowards(yaw_mean);

            // 필터값도 퍼블리시(제어/FSM에서 선택 사용)
            std_msgs::Float64 ybmsg; ybmsg.data = gYaw.yaw; yaw_b_m_pub.publish(ybmsg);

            // ★ 화면에 표시할 값(첫 번째 마커 기준) 업데이트
            if (i == 0) {
                disp_pos_x       = Xb;
                disp_pos_y       = Yb;
                disp_pos_z       = Zb;
                disp_yaw_b_m_raw = yaw_b_m_raw;   // RAW로 표시
                disp_id          = ids[i];        // ★ 첫 번째 마커 ID 저장
                
                std_msgs::Int32 idmsg;
    		idmsg.data = disp_id;
    		marker_id_pub.publish(idmsg);

            }

            ROS_INFO("ID:%d | base Pos: [X=%.3f, Y=%.3f, Z=%.3f] | cam RPY=[%.1f, %.1f, %.1f] | yaw_b raw=%.1f filt=%.1f | reproj=%.2fpx a=%.2f",
                     ids[i], Xb, Yb, Zb, disp_roll, disp_pitch, disp_yaw,
                     yaw_b_m_raw, gYaw.yaw, rms, alpha);
        }
    }

    // ===== 중앙 하단 텍스트 표시 (RAW 표기 + ID) =====
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.6;
    int thickness = 1;

    char line0[64], line1[128], line2[128];
    // ★ ID 라인: 마커가 없으면 ID: -
    if (disp_id >= 0) std::snprintf(line0, sizeof(line0), "ID: %d", disp_id);
    else               std::snprintf(line0, sizeof(line0), "ID: -");

    std::snprintf(line1, sizeof(line1), "Pos: X=%.2f  Y=%.2f  Z=%.2f",
                  disp_pos_x, disp_pos_y, disp_pos_z);
    std::snprintf(line2, sizeof(line2), "Yaw_b(m): %.1f deg", disp_yaw_b_m_raw); // RAW

    cv::Point base_pt(img_center.x - 220, img_center.y + 40);
    // ★ ID, Pos, Yaw_b 순서로 출력
    cv::putText(undistorted, line0, base_pt - cv::Point(0,24),
                fontFace, fontScale, cv::Scalar(0,255,255), thickness, cv::LINE_AA);
    cv::putText(undistorted, line1, base_pt,
                fontFace, fontScale, cv::Scalar(0,255,255), thickness, cv::LINE_AA);
    cv::putText(undistorted, line2, base_pt + cv::Point(0,24),
                fontFace, fontScale, cv::Scalar(0,255,255), thickness, cv::LINE_AA);

    // cv::imshow("ArUco Marker Detection", undistorted);
    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "aruco_detector");
    ros::NodeHandle pnh("~");
    ros::NodeHandle nh;

    // params
    pnh.param("use_optical_to_base", use_optical_to_base, true);

    std::vector<double> rpy_deg_vec, xyz_vec;
    if (pnh.getParam("cam_rpy_deg", rpy_deg_vec) && rpy_deg_vec.size()==3)
        cam_rpy_deg = cv::Vec3d(rpy_deg_vec[0], rpy_deg_vec[1], rpy_deg_vec[2]);
    if (pnh.getParam("cam_xyz", xyz_vec) && xyz_vec.size()==3)
        cam_xyz = cv::Vec3d(xyz_vec[0], xyz_vec[1], xyz_vec[2]);

    // 안정화 파라미터
    pnh.param("ema_alpha_yaw",    ema_alpha_yaw,    0.25);
    pnh.param("ema_alpha_min",    ema_alpha_min,    0.05);
    pnh.param("ema_alpha_max",    ema_alpha_max,    0.35);
    pnh.param("reproj_good_px",   reproj_good_px,   1.0);
    pnh.param("reproj_bad_px",    reproj_bad_px,    4.0);
    pnh.param("median_win",       median_win,       5);
    pnh.param("max_step_deg",     max_step_deg,     2.0);
    pnh.param("jump_reject_deg",  jump_reject_deg,  25.0);
    pnh.param("do_subpix",        do_subpix,        true);
    pnh.param("use_ippe_square",  use_ippe_square,  true);

    gYaw.setParams(ema_alpha_yaw, max_step_deg);
    gYawOk.tol_in  = 1.0;
    gYawOk.tol_out = 1.8;
    gYawOk.ok_need = 5;

    // aruco
    dictionary     = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detectorParams = cv::aruco::DetectorParameters::create();
    detectorParams->cornerRefinementMethod = do_subpix ? cv::aruco::CORNER_REFINE_SUBPIX
                                                       : cv::aruco::CORNER_REFINE_NONE;
    detectorParams->adaptiveThreshWinSizeMin = 5;
    detectorParams->adaptiveThreshWinSizeMax = 23;
    detectorParams->adaptiveThreshWinSizeStep = 4;

    // sub
    ros::Subscriber sub_img  = pnh.subscribe("image", 1, imageCallback);
    ros::Subscriber sub_info = pnh.subscribe("camera_info", 1, camInfoCallback);

    // pubs
    pitch_pub       = nh.advertise<std_msgs::Float64>("aruco/pitch",       10);
    yaw_pub         = nh.advertise<std_msgs::Float64>("aruco/yaw",         10);
    yaw_b_m_pub     = nh.advertise<std_msgs::Float64>("aruco/yaw_b_m",     10);  // 안정화(제어용)
    yaw_b_m_raw_pub = nh.advertise<std_msgs::Float64>("aruco/yaw_b_m_raw", 10);  // 원시(표시/디버그용)
    z_pub           = nh.advertise<std_msgs::Float64>("aruco/pose_z",      10);
    x_pub           = nh.advertise<std_msgs::Float64>("aruco/pose_x",      10);
    marker_id_pub   = nh.advertise<std_msgs::Int32>("aruco/marker_id", 10);


    ROS_INFO_STREAM("ArUco Detector Node (show RAW on screen, publish RAW+FILT). "
                    << "ema_alpha=" << ema_alpha_yaw
                    << " [" << ema_alpha_min << "~" << ema_alpha_max << "]"
                    << ", reproj good/bad px=" << reproj_good_px << "/" << reproj_bad_px
                    << ", median_win=" << median_win
                    << ", max_step=" << max_step_deg << "deg");

    ros::spin();
    return 0;
}

