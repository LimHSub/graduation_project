#include <Dynamixel2Arduino.h>

#define DXL_SERIAL   Serial1
#define DEBUG_SERIAL Serial
#define DXL_DIR_PIN  -1

using namespace ControlTableItem;

// ===== 사용자 설정 =====
// 적재함 다이나믹셀 ID
// 1개면 {100}
// 2개면 {100, 101} 이런 식으로 수정
const uint8_t DXL_ID[] = {100};
const int NUM = sizeof(DXL_ID) / sizeof(DXL_ID[0]);

// 프로토콜 / 보드레이트
const float DXL_PROTOCOL_VERSION = 2.0;
const uint32_t DXL_BAUDRATE = 1000000;

// 적재함 위치
const int32_t OPEN_POS  = 1700;  // b
const int32_t CLOSE_POS = 3073;  // n

// 전류 제한
const int32_t USER_CURRENT_LIMIT = 300;

// ===== 프로파일 설정 =====
// 여기 숫자만 바꾸면 됨
const int32_t USER_VELOCITY_LIMIT      = 150;
const int32_t USER_PROFILE_VELOCITY    = 80;
const int32_t USER_PROFILE_ACCELERATION = 20;

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

void torqueOffAll() {
  for (int i = 0; i < NUM; i++) {
    dxl.torqueOff(DXL_ID[i]);
  }
}

void torqueOnAll() {
  for (int i = 0; i < NUM; i++) {
    dxl.torqueOn(DXL_ID[i]);
  }
}

bool writeItemAll(uint16_t item_addr, int32_t value) {
  bool ok = true;
  for (int i = 0; i < NUM; i++) {
    if (!dxl.writeControlTableItem(item_addr, DXL_ID[i], value)) {
      ok = false;
    }
  }
  return ok;
}

void applyProfile() {
  writeItemAll(VELOCITY_LIMIT, USER_VELOCITY_LIMIT);
  writeItemAll(PROFILE_VELOCITY, USER_PROFILE_VELOCITY);
  writeItemAll(PROFILE_ACCELERATION, USER_PROFILE_ACCELERATION);

  DEBUG_SERIAL.print("[Profile] VEL_LIMIT=");
  DEBUG_SERIAL.print(USER_VELOCITY_LIMIT);
  DEBUG_SERIAL.print(", PROF_V=");
  DEBUG_SERIAL.print(USER_PROFILE_VELOCITY);
  DEBUG_SERIAL.print(", PROF_A=");
  DEBUG_SERIAL.println(USER_PROFILE_ACCELERATION);
}

void setPositionMode() {
  torqueOffAll();

  for (int i = 0; i < NUM; i++) {
    dxl.writeControlTableItem(DRIVE_MODE, DXL_ID[i], 0);  // normal
    dxl.setOperatingMode(DXL_ID[i], OP_POSITION);
  }

  if (!writeItemAll(CURRENT_LIMIT, USER_CURRENT_LIMIT)) {
    DEBUG_SERIAL.println("[WARN] CURRENT_LIMIT write failed");
  }

  applyProfile();
  torqueOnAll();
}

void moveAll(int32_t goal_pos) {
  for (int i = 0; i < NUM; i++) {
    dxl.writeControlTableItem(GOAL_POSITION, DXL_ID[i], goal_pos);
  }

  DEBUG_SERIAL.print("[MOVE] Goal Position = ");
  DEBUG_SERIAL.println(goal_pos);
}

void readPositionAll() {
  for (int i = 0; i < NUM; i++) {
    int32_t pos = dxl.getPresentPosition(DXL_ID[i]);
    DEBUG_SERIAL.print("ID ");
    DEBUG_SERIAL.print(DXL_ID[i]);
    DEBUG_SERIAL.print(" present position: ");
    DEBUG_SERIAL.println(pos);
  }
}

void printHelp() {
  DEBUG_SERIAL.println("===== Cargo Control =====");
  DEBUG_SERIAL.println("b : open  (2048)");
  DEBUG_SERIAL.println("n : close (3073)");
  DEBUG_SERIAL.println("o : torque on");
  DEBUG_SERIAL.println("p : torque off");
  DEBUG_SERIAL.println("z : read present position");
  DEBUG_SERIAL.println("=========================");
}

void setup() {
  DEBUG_SERIAL.begin(115200);
  while (!DEBUG_SERIAL) {}

  dxl.begin(DXL_BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  setPositionMode();
  printHelp();
  DEBUG_SERIAL.println("cargo openrb ready");
}

void loop() {
  if (DEBUG_SERIAL.available()) {
    char c = DEBUG_SERIAL.read();

    switch (c) {
      case 'b':
        moveAll(OPEN_POS);
        break;

      case 'n':
        moveAll(CLOSE_POS);
        break;

      case 'o':
        torqueOnAll();
        DEBUG_SERIAL.println("[TORQUE] ON");
        break;

      case 'p':
        torqueOffAll();
        DEBUG_SERIAL.println("[TORQUE] OFF");
        break;

      case 'z':
        readPositionAll();
        break;

      case '\n':
      case '\r':
        break;

      default:
        DEBUG_SERIAL.print("[WARN] Unknown cmd: ");
        DEBUG_SERIAL.println(c);
        break;
    }
  }
}
