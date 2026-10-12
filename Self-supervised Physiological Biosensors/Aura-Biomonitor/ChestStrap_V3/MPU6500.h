#ifndef MPU6500_H
#define MPU6500_H

#include <Arduino.h>
#include <Wire.h>

#include "Config.h"
#include "DataTypes.h"

class MPU6500
{
public:

    MPU6500();

    bool begin();

    void update();

    bool read(IMUData &data);

    bool available();

    bool selfTest();

private:

    bool initialized;

    IMUData imuData;

    void writeRegister(uint8_t reg, uint8_t value);

    uint8_t readRegister(uint8_t reg);

    int16_t read16(uint8_t reg);

    void readSensor();
};

#endif