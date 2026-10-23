#include <Dynamixel2Arduino.h>
#define DXL_SERIAL Serial1
#define DEBUG_SERIAL Serial
#define DXL_DIR_PIN -1

const uint8_t DXL_ID[]={100};
const int NUM = sizeof(DXL_ID)/sizeof(DXL_ID[0]);

const float DXL_PROTOCOL_VERSION = 2.0;
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

using namespace ControlTableItem;


// ===== 속도/가속 프로파일 프리셋(아주 낮게 시작 → 점증 권장) =====
struct Profile {
  int32_t vel_limit;   // VELOCITY_LIMIT
  int32_t prof_vel;    // PROFILE_VELOCITY
  int32_t prof_acc;    // PROFILE_ACCELERATION
};

// 모델/부하에 따라 숫자는 조정하세요. (낮게 시작!)
Profile LV0 = {  50,  10,   3};   // 가장 안전 (시작권장)
Profile LV1 = {  80,  30,   5};
Profile LV2 = { 140,  60,  10};
Profile LV3 = { 220, 110,  18};
Profile LV4 = { 320, 160,  25};
Profile CUR = LV4;

void torqueOffAll(){
  for (int i=0;i<NUM;i++) dxl.torqueOff(DXL_ID[i]);
}
void torqueOnAll(){
  for (int i=0;i<NUM;i++) dxl.torqueOn(DXL_ID[i]);
}

// ControlTableItem은 상수 주소로 동작 → uint16_t로 취급
bool writeItemAll(uint16_t item_addr, int32_t value){
  bool ok = true;
  for(int i=0;i<NUM;i++){
    if(!dxl.writeControlTableItem(item_addr, DXL_ID[i], value)) ok = false;
  }
  return ok;
}


void applyProfile(const Profile& pf){
  CUR = pf;
  writeItemAll(VELOCITY_LIMIT,       pf.vel_limit);
  writeItemAll(PROFILE_VELOCITY,     pf.prof_vel);
  writeItemAll(PROFILE_ACCELERATION, pf.prof_acc);

  DEBUG_SERIAL.print("[Profile] VEL_LIMIT="); DEBUG_SERIAL.print(pf.vel_limit);
  DEBUG_SERIAL.print(", PROF_V="); DEBUG_SERIAL.print(pf.prof_vel);
  DEBUG_SERIAL.print(", PROF_A="); DEBUG_SERIAL.println(pf.prof_acc);
}

// void setPositionModeAndProfile(const Profile& pf){
//   torqueOffAll();
//   // Position 모드로 전환
//   for(int i=0;i<NUM;i++){
//     dxl.setOperatingMode(DXL_ID[i], OP_POSITION);
//   }
//   // 보호차원에서 전류 제한(가능 모델) — 실패해도 진행
//   if(!writeItemAll(CURRENT_LIMIT, 300)){ // 값은 모델/부하에 맞게 조정
//     DEBUG_SERIAL.println("[WARN] CURRENT_LIMIT write failed (model may not support)");
//   }
//   // 속도/가속 관련 파라미터 적용
//  // applyProfile(pf);
//   torqueOnAll();
// }

void setPositionModeAndProfile(const Profile& pf){
  torqueOffAll();
  // Position 모드로 전환
  for(int i=0;i<NUM;i++){
    //Normal drvie CCW mode setup
    dxl.writeControlTableItem(ControlTableItem::DRIVE_MODE, DXL_ID[i], 0);
    dxl.setOperatingMode(DXL_ID[i], OP_POSITION);
  }
  // 보호차원에서 전류 제한(가능 모델) — 실패해도 진행
  if(!writeItemAll(CURRENT_LIMIT, 300)){ // 값은 모델/부하에 맞게 조정
    DEBUG_SERIAL.println("[WARN] CURRENT_LIMIT write failed (model may not support)");
  }
  // 속도/가속 관련 파라미터 적용
 // applyProfile(pf);
  torqueOnAll();
}

void setup() {
  // put your setup code here, to run once:
  DEBUG_SERIAL.begin(115200);
  while(!DEBUG_SERIAL){;}
 
  dxl.begin(1000000);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  setPositionModeAndProfile(CUR);
}
//int J1 =1, J2=1, J3=1, J4=1, J5=1;

void loop() {
  if (DEBUG_SERIAL.available()) {
   char c = DEBUG_SERIAL.read();
 
   switch(c){

   case 'q': applyProfile(LV0);break;
   case 'w': applyProfile(LV1);break;
   case 'e': applyProfile(LV2);break;
   case 'r': applyProfile(LV3);break;
   case 't': applyProfile(LV4);break;
   case 'i': init_pose(); break;
   case 'p':  torqueOffAll(); break; //정지
   case 'o': torqueOnAll(); break;  //재시작
   case 'z': read_position(); break; //각도 읽기
   case 'm': {
    get_pose1(3073,2428,1318,0,2402,2049);
    delay(2000);
    get_pose1(3507,1550,2167,0,2560,1078);
    delay(500);
    get_pose1(-1,-1,-1,-1,-1,130);
    delay(5000);
    get_pose1(3073,2428,1318,0,2402,130);
    delay(2000);
    get_pose1(2331,2502,1200,0,2507,130);
    delay(1000);
    get_pose1(2331,2502,1200,0,2507,1078);
    delay(1000);
    get_pose1(3073,2428,1318,0,2402,2049);



    break;
   }
   case 'a':{
    Serial.print("put angle(1~4096):");
      while(1){
        int J1 = DEBUG_SERIAL.parseInt();
        if(J1 > 0) {get_pose1((J1-1),-1,-1,-1,-1,-1);
                    Serial.print("moving");
                    break;}
      }
      break;
    }
    case 's':{
    Serial.print("put angle(1~4096):");
      while(1){
        int J2 = DEBUG_SERIAL.parseInt();
        if(J2 > 0) {get_pose1(-1,(J2-1),-1,-1,-1,-1);
                    Serial.print("moving");
                    break;}
      }
      break;
    }
    case 'd':{
    Serial.print("put angle(1~4096):");
      while(1){
        int J3 = DEBUG_SERIAL.parseInt();
        if(J3 > 0) {get_pose1(-1,-1,(J3-1),-1,-1,-1);
                    Serial.print("moving");
                    break;}
      }
      break;
    }
    case 'f':{
    Serial.print("put angle(1~4096):");
      while(1){
        int J4 = DEBUG_SERIAL.parseInt();
        if(J4 > 0) {get_pose1(-1,-1,-1,(J4-1),-1,-1);
                    Serial.print("moving");
                    break;}
      }
      break;
    }
    case 'g':{
    Serial.print("put angle(1~4096):");
      while(1){
        int J5 = DEBUG_SERIAL.parseInt();
        if(J5 > 0) {get_pose1(-1,-1,-1,-1,(J5-1),-1);
                    Serial.print("moving");
                    break;}
      }
      break;
    case 'h':{
    Serial.print("put angle(1~4096):");
      while(1){
        int J6 = DEBUG_SERIAL.parseInt();
        if(J6 > 0) {get_pose1(-1,-1,-1,-1,-1,(J6-1));
                    Serial.print("moving");
                    break;}
      }
      break;
    }

   }
   }

}
}
//1=0도
//4096=360도
//3073=270도
//2049=180도
//1025=90도

void get_pose1(int j1, int j2,int j3,int j4,int j5, int j6){
  //const int32_t pos[NUM]={1024,0,0,0,0};
  DEBUG_SERIAL.println("동작 1실행");
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[0], j1);
  delay(100);
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[1], j2);
  delay(100);
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[2], j3);
  delay(100);
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[3], j4);
  delay(100);
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[4], j5);
  delay(100);
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[5], j6);


}

void init_pose(){
    DEBUG_SERIAL.println("초기 위치 실행");
   dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[0], 0);
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[1], 2100);
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[2], 2100);
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[3], 0);
  dxl.writeControlTableItem(GOAL_POSITION,DXL_ID[4], 2049);
}


void read_position(){
 
  for(int count = 0; count < 1; count++)
  {
    Serial.print(count + 1);
    Serial.println( "번째 순서");
    int pos1 = dxl.getPresentPosition(DXL_ID[0]);  // 현재 위치 읽기
    Serial.print("현재 위치1: ");
    Serial.println(pos1);  // 시리얼 모니터에 출력

    delay(1000);  // 1초마다 읽기


    int pos2 = dxl.getPresentPosition(DXL_ID[1]);  // 현재 위치 읽기
    Serial.print("현재 위치2: ");
    Serial.println(pos2);  // 시리얼 모니터에 출력

    delay(1000);  // 1초마다 읽기
     
    int pos3 = dxl.getPresentPosition(DXL_ID[2]);  // 현재 위치 읽기
    Serial.print("현재 위치3: ");
    Serial.println(pos3);  // 시리얼 모니터에 출력

    delay(1000);  // 1초마다 읽기
   

    int pos4 = dxl.getPresentPosition(DXL_ID[3]);  // 현재 위치 읽기
    Serial.print("현재 위치4: ");
    Serial.println(pos4);  // 시리얼 모니터에 출력

    delay(1000);  // 1초마다 읽기
   

    int pos5 = dxl.getPresentPosition(DXL_ID[4]);  // 현재 위치 읽기
    Serial.print("현재 위치5: ");
    Serial.println(pos5);  // 시리얼 모니터에 출력

    int pos6 = dxl.getPresentPosition(DXL_ID[5]);  // 현재 위치 읽기
    Serial.print("현재 위치5: ");
    Serial.println(pos6);  // 시리얼 모니터에 출력

    delay(5000);  // 1초마다 읽기
   
    count++;
  }
}