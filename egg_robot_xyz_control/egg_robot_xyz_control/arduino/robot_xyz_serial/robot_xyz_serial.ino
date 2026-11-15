#include <AccelStepper.h>
// RAMPS X=shoulder, Y=elbow, Z=base. Gripper intentionally unused.
AccelStepper sx(AccelStepper::DRIVER,54,55), sy(AccelStepper::DRIVER,60,61), sz(AccelStepper::DRIVER,46,48);
const byte EX=38,EY=56,EZ=62;
const long LIMIT=110; // +/-110 microsteps from manually established reference
bool armed=false;
String line="";
void power(bool on){digitalWrite(EX,on?LOW:HIGH);digitalWrite(EY,on?LOW:HIGH);digitalWrite(EZ,on?LOW:HIGH);}
bool moving(){return sx.distanceToGo()!=0 || sy.distanceToGo()!=0 || sz.distanceToGo()!=0;}
void status(){Serial.print("STATUS armed=");Serial.print(armed);Serial.print(" busy=");Serial.print(moving());Serial.print(" X=");Serial.print(sx.currentPosition());Serial.print(" Y=");Serial.print(sy.currentPosition());Serial.print(" Z=");Serial.println(sz.currentPosition());}
void handle(String cmd){cmd.trim();cmd.toUpperCase();
 if(cmd=="STATUS"){status();return;}
 if(cmd=="ARM"){
   if(armed||moving()){Serial.println("ERR already armed or moving");return;}
   sx.setCurrentPosition(0);sy.setCurrentPosition(0);sz.setCurrentPosition(0);
   power(true);armed=true;Serial.println("OK ARMED reference set");return;
 }
 if(cmd=="STOP"){sx.stop();sy.stop();sz.stop();Serial.println("OK STOPPING (controlled deceleration)");return;}
 if(cmd=="DISARM"){
   if(moving()){Serial.println("ERR moving");return;}
   power(false);armed=false;Serial.println("OK DISARMED (no holding torque)");return;
 }
 if(cmd.startsWith("MOVE3 ")){
   if(!armed||moving()){Serial.println("ERR not armed or busy");return;}
   long x,y,z;char extra;
   if(sscanf(cmd.c_str(),"MOVE3 %ld %ld %ld %c",&x,&y,&z,&extra)!=3){Serial.println("ERR format MOVE3 X Y Z");return;}
   if(labs(x)>LIMIT||labs(y)>LIMIT||labs(z)>LIMIT){Serial.println("ERR outside +/-110 step reference envelope");return;}
   // Targets are absolute software step coordinates. All motors run each loop.
   sx.moveTo(x);sy.moveTo(y);sz.moveTo(z);Serial.println("OK MOVING3");return;
 }
 Serial.println("ERR unknown command");
}
void setup(){Serial.begin(115200);pinMode(EX,OUTPUT);pinMode(EY,OUTPUT);pinMode(EZ,OUTPUT);power(false);
 sx.setMaxSpeed(150);sy.setMaxSpeed(150);sz.setMaxSpeed(150);
 sx.setAcceleration(80);sy.setAcceleration(80);sz.setAcceleration(80);
 Serial.println("READY unarmed");}
void loop(){sx.run();sy.run();sz.run();while(Serial.available()){
 char c=(char)Serial.read();if(c=='\n'||c=='\r'){if(line.length()){handle(line);line="";}}
 else if(line.length()<70){line+=c;}
}}
