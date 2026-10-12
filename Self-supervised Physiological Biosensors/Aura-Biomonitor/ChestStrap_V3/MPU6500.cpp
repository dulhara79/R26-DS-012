#include "MPU6500.h"

/************************************************
 * Registers
 ************************************************/

#define WHO_AM_I       0x75

#define PWR_MGMT_1     0x6B
#define PWR_MGMT_2     0x6C

#define ACCEL_CONFIG   0x1C
#define GYRO_CONFIG    0x1B

#define CONFIG         0x1A
#define SMPLRT_DIV     0x19

#define ACCEL_XOUT_H   0x3B
#define TEMP_OUT_H     0x41
#define GYRO_XOUT_H    0x43

/************************************************/

MPU6500::MPU6500()
{
    initialized = false;
}

/************************************************/

bool MPU6500::begin()
{
    Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN);

    Wire.setClock(IMU_I2C_SPEED);

    delay(100);

    if(!selfTest())
        return false;

    // Wake device
    writeRegister(PWR_MGMT_1,0x00);

    delay(100);

    // DLPF = 41Hz

    writeRegister(CONFIG,0x03);

    // ±2g

    writeRegister(ACCEL_CONFIG,0x00);

    // ±250°/s

    writeRegister(GYRO_CONFIG,0x00);

    // 1kHz

    writeRegister(SMPLRT_DIV,0x00);

    initialized = true;

    return true;
}

/************************************************/

bool MPU6500::selfTest()
{
    return (readRegister(WHO_AM_I)==0x70);
}

/************************************************/

bool MPU6500::available()
{
    return initialized;
}

/************************************************/

void MPU6500::update()
{
    if(!initialized)
        return;

    readSensor();
}

/************************************************/

bool MPU6500::read(IMUData &data)
{
    if(!initialized)
        return false;

    data = imuData;

    imuData.newData = false;

    return true;
}

/************************************************/

void MPU6500::readSensor()
{
    imuData.ax = read16(ACCEL_XOUT_H);
    imuData.ay = read16(ACCEL_XOUT_H+2);
    imuData.az = read16(ACCEL_XOUT_H+4);

    imuData.temperature = read16(TEMP_OUT_H);

    imuData.gx = read16(GYRO_XOUT_H);
    imuData.gy = read16(GYRO_XOUT_H+2);
    imuData.gz = read16(GYRO_XOUT_H+4);

    imuData.newData = true;
}

/************************************************/

void MPU6500::writeRegister(uint8_t reg,uint8_t value)
{
    Wire.beginTransmission(MPU6500_ADDRESS);

    Wire.write(reg);

    Wire.write(value);

    Wire.endTransmission();
}

/************************************************/

uint8_t MPU6500::readRegister(uint8_t reg)
{
    Wire.beginTransmission(MPU6500_ADDRESS);

    Wire.write(reg);

    Wire.endTransmission(false);

    Wire.requestFrom(MPU6500_ADDRESS,(uint8_t)1);

    if(Wire.available())
        return Wire.read();

    return 0xFF;
}

/************************************************/

int16_t MPU6500::read16(uint8_t reg)
{
    Wire.beginTransmission(MPU6500_ADDRESS);

    Wire.write(reg);

    Wire.endTransmission(false);

    Wire.requestFrom(MPU6500_ADDRESS,(uint8_t)2);

    int16_t value = ((int16_t)Wire.read()<<8);

    value |= Wire.read();

    return value;
}