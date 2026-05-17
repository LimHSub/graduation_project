#include <ros/ros.h>
#include <std_msgs/String.h>

#include <hardware_interface/robot_hw.h>
#include <hardware_interface/joint_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <controller_manager/controller_manager.h>

#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <mutex>
#include <iomanip>
#include <stdexcept>

// POSIX serial
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <string.h>

static inline double wrapToPiD(double a){
  const double PI = M_PI;
  const double TWO_PI = 2.0 * M_PI;
  while(a >  PI) a -= TWO_PI;
  while(a < -PI) a += TWO_PI;
  return a;
}

class ArmHW : public hardware_interface::RobotHW
{
public:
  explicit ArmHW(ros::NodeHandle& nh)
  : nh_(nh)
  {
    nh_.param<std::string>("port", port_name_, std::string("/dev/ttyACM0"));
    nh_.param<int>("baud", baud_, 115200);
    nh_.param<int>("loop_hz", loop_hz_, 100);
    nh_.param<bool>("debug_tx", debug_tx_, true);
    nh_.param<bool>("debug_rx", debug_rx_, true);
    nh_.param<bool>("unwrap_enable", unwrap_enable_, false);
    nh_.param<bool>("write_enable", write_enable_, true);

    home_rad_ = {1.573, 0.081, 0.081, -3.142, 0.003};
    sign_     = {1, 1, 1, 1, 1};

    nh_.getParam("home_rad", home_rad_);
    nh_.getParam("sign", sign_);

    if(home_rad_.size() != 5 || sign_.size() != 5){
      ROS_WARN("home_rad/sign size must be 5. Using defaults.");
      home_rad_ = {1.573, 0.081, 0.081, -3.142, 0.003};
      sign_     = {1, 1, 1, 1, 1};
    }

    joint_names_ = {"Revolute1", "Revolute2", "Revolute3", "Revolute4", "Revolute5"};
    const int N = (int)joint_names_.size();

    pos_.assign(N, 0.0);
    vel_.assign(N, 0.0);
    eff_.assign(N, 0.0);
    cmd_.assign(N, 0.0);

    pos_unwrap_.assign(N, 0.0);
    last_fb_wrap_.assign(N, 0.0);
    last_fb_abs_.assign(N, 0.0);

    for(int i=0; i<N; i++){
      hardware_interface::JointStateHandle sh(joint_names_[i], &pos_[i], &vel_[i], &eff_[i]);
      jnt_state_interface_.registerHandle(sh);

      hardware_interface::JointHandle ch(jnt_state_interface_.getHandle(joint_names_[i]), &cmd_[i]);
      jnt_pos_interface_.registerHandle(ch);
    }

    registerInterface(&jnt_state_interface_);
    registerInterface(&jnt_pos_interface_);

    openSerialPosix();

    serial_tx_sub_  = nh_.subscribe("/arm/serial_tx", 10, &ArmHW::serialTxCb, this);
    opencr_tx_sub_  = nh_.subscribe("/opencr_tx", 10, &ArmHW::serialTxCb, this);
    opencr_key_sub_ = nh_.subscribe("/opencr_key", 10, &ArmHW::opencrKeyCb, this);
    serial_rx_pub_  = nh_.advertise<std_msgs::String>("/arm/serial_rx", 50);

    ROS_INFO("arm_hw_node ready. port=%s baud=%d loop_hz=%d", port_name_.c_str(), baud_, loop_hz_);
    ROS_INFO("Protocol: FB_R(abs rad x6), CMD_R(abs rad x6; 6th keep token by default)");
    ROS_INFO("Serial TX topics: /arm/serial_tx, /opencr_tx");
    ROS_INFO("Key TX topic   : /opencr_key");
    ROS_INFO("Serial RX topic: /arm/serial_rx");
    ROS_INFO("unwrap_enable=%s", unwrap_enable_ ? "true" : "false");
    ROS_INFO("write_enable=%s", write_enable_ ? "true" : "false");
    ROS_INFO("Mode: OpenCR ABS <-> ROS HOME_RAD-relative");
  }

  int loopHz() const { return loop_hz_; }

  void read(const ros::Duration& /*period*/)
  {
    char buf[512];
    while(true){
      ssize_t n = ::read(fd_, buf, sizeof(buf));
      if(n > 0){
        rx_buf_.append(buf, buf + n);
      }else{
        if(n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        break;
      }
    }

    while(true){
      std::size_t p = rx_buf_.find('\n');
      if(p == std::string::npos) break;

      std::string line = rx_buf_.substr(0, p);
      rx_buf_.erase(0, p+1);
      if(!line.empty() && line.back() == '\r') line.pop_back();
      if(line.empty()) continue;

      if(debug_rx_){
        std_msgs::String m;
        m.data = line;
        serial_rx_pub_.publish(m);
      }

      if(line.rfind("FB_R", 0) == 0){
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;

        bool ok = true;
        double r[6] = {0,0,0,0,0,0};
        for(int k=0; k<6; k++){
          if(!(iss >> r[k])) { ok = false; break; }
        }
        if(!ok) continue;

        for(int j=0; j<5; j++){
          const double cur_abs = wrapToPiD(r[j]);
          last_fb_abs_[j] = cur_abs;

          // OpenCR 절대각 -> ROS(home 기준 상대각)
          double cur_moveit = wrapToPiD((double)sign_[j] * (cur_abs - home_rad_[j]));

          if(!got_first_fb_){
            pos_unwrap_[j]   = cur_moveit;
            last_fb_wrap_[j] = cur_moveit;
            pos_[j] = unwrap_enable_ ? pos_unwrap_[j] : cur_moveit;
          }else{
            if(unwrap_enable_){
              const double d = wrapToPiD(cur_moveit - last_fb_wrap_[j]);
              pos_unwrap_[j] += d;
              last_fb_wrap_[j] = cur_moveit;
              pos_[j] = pos_unwrap_[j];
            }else{
              pos_[j] = cur_moveit;
              last_fb_wrap_[j] = cur_moveit;
            }
          }

          // 간단한 속도/effort는 0으로 둠
          vel_[j] = 0.0;
          eff_[j] = 0.0;
        }

        last_gripper_rad_ = r[5];

        if(!got_first_fb_){
          got_first_fb_ = true;
          for(int j=0; j<5; j++){
            cmd_[j] = pos_[j];
          }
          ROS_INFO("First FB_R received -> cmd_ synced to current HOME_RAD-relative pose");
          skip_cmd_cycles_ = 20;
        }
      }
    }
  }

  void write(const ros::Duration& /*period*/)
  {
    // 수동 관절 상태 확인용: 송신 완전 차단
    if(!write_enable_) return;

    std::string tx;
    {
      std::lock_guard<std::mutex> lock(tx_mtx_);
      if(!pending_tx_.empty()) tx.swap(pending_tx_);
    }

    if(!tx.empty()){
      if(debug_tx_){
        std::string printable = tx;
        if(!printable.empty() && printable.back() == '\n') printable.pop_back();
        ROS_INFO_STREAM("TX(raw): " << printable);
      }
      writeStr(tx);
      return;
    }

    if(!got_first_fb_) return;

    if(skip_cmd_cycles_ > 0){
      skip_cmd_cycles_--;
      return;
    }

    // ROS(home 기준 상대각) -> OpenCR 절대각
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);

    oss << "CMD_R";
    for(int j=0; j<5; j++){
      double out_abs = ((double)sign_[j] * cmd_[j]) + home_rad_[j];
      out_abs = wrapToPiD(out_abs);
      oss << " " << out_abs;
    }
    oss << " x\n";

    const std::string s = oss.str();

    if(debug_tx_){
      std::string printable = s;
      if(!printable.empty() && printable.back() == '\n') printable.pop_back();
      ROS_INFO_STREAM_THROTTLE(0.5, "TX(cmd_r): " << printable);
    }
    writeStr(s);
  }

private:
  void serialTxCb(const std_msgs::String::ConstPtr& msg)
  {
    if(!write_enable_) return;

    std::lock_guard<std::mutex> lock(tx_mtx_);
    pending_tx_ += msg->data;
    if(pending_tx_.empty() || pending_tx_.back() != '\n'){
      pending_tx_ += "\n";
    }
    skip_cmd_cycles_ = 10;
  }

  void opencrKeyCb(const std_msgs::String::ConstPtr& msg)
  {
    if(!write_enable_) return;

    std::string key = msg->data;
    if(key.empty()) return;
    std::string one(1, key[0]);

    std::lock_guard<std::mutex> lock(tx_mtx_);
    pending_tx_ += one;
    if(pending_tx_.empty() || pending_tx_.back() != '\n'){
      pending_tx_ += "\n";
    }
    skip_cmd_cycles_ = 10;
  }

  void openSerialPosix()
  {
    fd_ = ::open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if(fd_ < 0){
      ROS_FATAL("Failed to open %s : %s", port_name_.c_str(), strerror(errno));
      throw std::runtime_error("open serial failed");
    }

    termios tio;
    if(tcgetattr(fd_, &tio) != 0){
      ROS_FATAL("tcgetattr failed: %s", strerror(errno));
      throw std::runtime_error("tcgetattr failed");
    }

    cfmakeraw(&tio);

    speed_t sp = B115200;
    if(baud_ == 57600) sp = B57600;
    else if(baud_ == 9600) sp = B9600;

    cfsetispeed(&tio, sp);
    cfsetospeed(&tio, sp);

    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cflag &= ~CSTOPB;
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cc[VMIN]  = 0;
    tio.c_cc[VTIME] = 0;

    if(tcsetattr(fd_, TCSANOW, &tio) != 0){
      ROS_FATAL("tcsetattr failed: %s", strerror(errno));
      throw std::runtime_error("tcsetattr failed");
    }

    rx_buf_.clear();
  }

  void writeStr(const std::string& s)
  {
    if(fd_ < 0) return;

    const char* p = s.c_str();
    size_t left = s.size();

    while(left > 0){
      ssize_t n = ::write(fd_, p, left);
      if(n > 0){
        p += n;
        left -= (size_t)n;
      }else{
        if(n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        ROS_WARN_THROTTLE(1.0, "Serial write error: %s", strerror(errno));
        break;
      }
    }
  }

private:
  ros::NodeHandle nh_;

  hardware_interface::JointStateInterface jnt_state_interface_;
  hardware_interface::PositionJointInterface jnt_pos_interface_;

  std::vector<std::string> joint_names_;
  std::vector<double> pos_, vel_, eff_, cmd_;

  bool unwrap_enable_ = false;
  bool write_enable_ = true;

  std::vector<double> pos_unwrap_;
  std::vector<double> last_fb_wrap_;
  std::vector<double> last_fb_abs_;

  std::vector<double> home_rad_;
  std::vector<int> sign_;

  int fd_ = -1;
  std::string port_name_;
  int baud_ = 115200;
  std::string rx_buf_;

  ros::Subscriber serial_tx_sub_;
  ros::Subscriber opencr_tx_sub_;
  ros::Subscriber opencr_key_sub_;
  ros::Publisher  serial_rx_pub_;

  std::mutex tx_mtx_;
  std::string pending_tx_;

  bool got_first_fb_ = false;
  int  skip_cmd_cycles_ = 0;
  int  loop_hz_ = 100;
  bool debug_tx_ = true;
  bool debug_rx_ = true;

  double last_gripper_rad_ = 0.0;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "arm_hw_node");
  ros::NodeHandle nh("~");

  ArmHW robot(nh);
  controller_manager::ControllerManager cm(&robot);

  ros::AsyncSpinner spinner(2);
  spinner.start();

  ros::Rate rate(robot.loopHz());
  ros::Time last = ros::Time::now();

  while(ros::ok()){
    ros::Time now = ros::Time::now();
    ros::Duration period = now - last;
    last = now;

    robot.read(period);
    cm.update(now, period);
    robot.write(period);

    rate.sleep();
  }
  return 0;
}
