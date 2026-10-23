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

// ===== 전류 감시 / 재시도 설정 =====
const float CURRENT_DELTA_THRESHOLD = 97.0f; // RAW 단위, 실측 후 조정
const uint8_t CONTACT_COUNT_REQUIRED = 2;

const uint32_t SAMPLE_INTERVAL_MS = 20;
const uint32_t BASELINE_SETTLE_MS = 300;
const uint8_t BASELINE_SAMPLES = 10;
const uint32_t RETRY_WAIT_MS = 2000;
const uint32_t MOTION_TIMEOUT_MS = 10000;
const int32_t POSITION_TOLERANCE = 20;

// delay() 없이 동작 단계를 관리
enum CargoState {
  IDLE,
  BASELINE_SETTLE,
  BASELINE_MEASURE,
  CLOSING,
  OPENING,
  RETRY_WAIT,
  FAULT
};

CargoState cargoState = IDLE;

bool retryAfterOpen = false;
bool torqueEnabled = true;

uint32_t stateStartMs = 0;
uint32_t lastSampleMs = 0;
uint32_t lastLogMs = 0;

float baselineCurrent[NUM] = {};
float currentSum[NUM] = {};
uint8_t overCount[NUM] = {};
uint8_t baselineCount = 0;

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

// 목표 위치 전송 성공 여부까지 확인
bool moveAll(int32_t goal_pos) {
  bool ok = true;

  for (int i = 0; i < NUM; i++) {
    if (!dxl.writeControlTableItem(
          GOAL_POSITION, DXL_ID[i], goal_pos)) {
      ok = false;
    }
  }

  DEBUG_SERIAL.print("[MOVE] Goal Position = ");
  DEBUG_SERIAL.println(goal_pos);
  return ok;
}

void changeState(CargoState next) {
  cargoState = next;
  stateStartMs = millis();
  lastSampleMs = stateStartMs;
}

// 현재 위치를 새로운 목표로 지정
// 통신 장애 시 실제 정지까지 보장되는 것은 아님
bool holdCurrentPositionAll() {
  bool ok = true;

  for (int i = 0; i < NUM; i++) {
    int32_t pos =
      dxl.readControlTableItem(PRESENT_POSITION, DXL_ID[i]);

    if (dxl.getLastLibErrCode() != 0) {
      ok = false;
      continue;
    }

    if (!dxl.writeControlTableItem(
          GOAL_POSITION, DXL_ID[i], pos)) {
      ok = false;
    }
  }

  return ok;
}

void enterFault(const char* reason) {
  retryAfterOpen = false;
  cargoState = FAULT;

  DEBUG_SERIAL.print("[FAULT] ");
  DEBUG_SERIAL.println(reason);

  if (!holdCurrentPositionAll()) {
    DEBUG_SERIAL.println("[FAULT] Hold command failed");
  }
}

// 성공한 읽기만 사용
bool readCurrentRaw(int index, float& value) {
  value = dxl.getPresentCurrent(DXL_ID[index], UNIT_RAW);
  return dxl.getLastLibErrCode() == 0;
}

bool readArrival(int32_t target, bool& arrived) {
  arrived = true;

  for (int i = 0; i < NUM; i++) {
    int32_t pos =
      dxl.readControlTableItem(PRESENT_POSITION, DXL_ID[i]);

    if (dxl.getLastLibErrCode() != 0) {
      return false;
    }

    if (labs(pos - target) > POSITION_TOLERANCE) {
      arrived = false;
    }
  }

  return true;
}

// 닫기 전 정지 및 기준 전류 측정 준비
void prepareClose() {
  if (!holdCurrentPositionAll()) {
    enterFault("Cannot hold before baseline");
    return;
  }

  baselineCount = 0;

  for (int i = 0; i < NUM; i++) {
    currentSum[i] = 0.0f;
    overCount[i] = 0;
  }

  changeState(BASELINE_SETTLE);
  DEBUG_SERIAL.println("[CLOSE] Preparing current baseline");
}

void startOpening(bool retry) {
  retryAfterOpen = retry;

  if (!moveAll(OPEN_POS)) {
    enterFault("Open command failed");
    return;
  }

  changeState(OPENING);
  DEBUG_SERIAL.println(
    retry ? "[OPEN] Obstruction: reopening"
          : "[OPEN] Manual open");
}

void updateCargo() {
  uint32_t now = millis();

  if (cargoState == IDLE || cargoState == FAULT) {
    return;
  }

  // 정지 후 전류가 안정될 시간을 확보
  if (cargoState == BASELINE_SETTLE) {
    if (now - stateStartMs >= BASELINE_SETTLE_MS) {
      changeState(BASELINE_MEASURE);
    }
    return;
  }

  // 열린 위치 도착 이후부터 2초 대기
  if (cargoState == RETRY_WAIT) {
    if (now - stateStartMs >= RETRY_WAIT_MS) {
      DEBUG_SERIAL.println("[RETRY] Close again");
      prepareClose();
    }
    return;
  }

  // 이동 완료가 확인되지 않으면 자동 반복 중단
  if ((cargoState == OPENING || cargoState == CLOSING) &&
      now - stateStartMs >= MOTION_TIMEOUT_MS) {
    enterFault("Motion timeout");
    return;
  }

  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = now;

  // ===== 기준 전류: 10회 평균 =====
  if (cargoState == BASELINE_MEASURE) {
    for (int i = 0; i < NUM; i++) {
      float current;

      if (!readCurrentRaw(i, current)) {
        enterFault("Baseline current read failed");
        return;
      }

      currentSum[i] += current;
    }

    baselineCount++;

    if (baselineCount >= BASELINE_SAMPLES) {
      for (int i = 0; i < NUM; i++) {
        baselineCurrent[i] = currentSum[i] / baselineCount;

        DEBUG_SERIAL.print("[BASE] ID=");
        DEBUG_SERIAL.print(DXL_ID[i]);
        DEBUG_SERIAL.print(" current=");
        DEBUG_SERIAL.println(baselineCurrent[i]);
      }

      if (!moveAll(CLOSE_POS)) {
        enterFault("Close command failed");
        return;
      }

      changeState(CLOSING);
    }
    return;
  }

  // ===== 닫는 중에만 전류 감시 =====
  if (cargoState == CLOSING) {
    bool printLog = (now - lastLogMs >= 100);

    for (int i = 0; i < NUM; i++) {
      float current;

      if (!readCurrentRaw(i, current)) {
        enterFault("Current read failed while closing");
        return;
      }

      float delta = current - baselineCurrent[i];
      if (delta < 0.0f) delta = -delta;

      if (delta >= CURRENT_DELTA_THRESHOLD) {
        overCount[i]++;
      } else {
        overCount[i] = 0;
      }

      if (printLog) {
        DEBUG_SERIAL.print("[CURRENT] ID=");
        DEBUG_SERIAL.print(DXL_ID[i]);
        DEBUG_SERIAL.print(" raw=");
        DEBUG_SERIAL.print(current);
        DEBUG_SERIAL.print(" delta=");
        DEBUG_SERIAL.println(delta);
      }

      // 모터가 여러 개라면 하나라도 감지했을 때 전체 열기
      if (overCount[i] >= CONTACT_COUNT_REQUIRED) {
        DEBUG_SERIAL.println("[CONTACT] Reopen and retry");
        startOpening(true);
        return;
      }
    }

    if (printLog) lastLogMs = now;

    bool arrived;
    if (!readArrival(CLOSE_POS, arrived)) {
      enterFault("Close position read failed");
      return;
    }

    if (arrived) {
      retryAfterOpen = false;
      changeState(IDLE);
      DEBUG_SERIAL.println("[DONE] Closed");
    }
    return;
  }

  // ===== 열림 완료 확인 =====
  if (cargoState == OPENING) {
    bool arrived;

    if (!readArrival(OPEN_POS, arrived)) {
      enterFault("Open position read failed");
      return;
    }

    if (arrived) {
      if (retryAfterOpen) {
        changeState(RETRY_WAIT);
        DEBUG_SERIAL.println("[WAIT] Open reached, wait 2 sec");
      } else {
        changeState(IDLE);
        DEBUG_SERIAL.println("[DONE] Opened");
      }
    }
  }
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
        if (torqueEnabled) {
          startOpening(false);
        } else {
          DEBUG_SERIAL.println("[WARN] Torque is OFF");
        }
        break;

      case 'n':
        if (!torqueEnabled) {
          DEBUG_SERIAL.println("[WARN] Torque is OFF");
        } else if (cargoState == IDLE) {
          prepareClose();
        } else {
          DEBUG_SERIAL.println(
            "[WARN] Busy or fault; close command ignored");
        }
        break;

      case 'o':
        // 기존 목표로 갑자기 움직이지 않도록 현재 위치 지정
        if (!holdCurrentPositionAll()) {
          enterFault("Cannot prepare torque ON");
          break;
        }

        {
          bool ok = true;
          for (int i = 0; i < NUM; i++) {
            if (!dxl.torqueOn(DXL_ID[i])) ok = false;
          }

          if (!ok) {
            enterFault("Torque ON failed");
            break;
          }
        }

        torqueEnabled = true;
        retryAfterOpen = false;
        changeState(IDLE);
        DEBUG_SERIAL.println("[TORQUE] ON");
        break;

      case 'p':
        retryAfterOpen = false;
        changeState(IDLE);
        torqueOffAll();
        torqueEnabled = false;
        DEBUG_SERIAL.println("[TORQUE] OFF requested");
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

  updateCargo();
}
