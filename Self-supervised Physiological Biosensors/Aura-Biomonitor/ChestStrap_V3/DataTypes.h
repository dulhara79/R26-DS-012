#ifndef DATATYPES_H
#define DATATYPES_H

#include <Arduino.h>

/**********************************************************
 * ECG Data
 **********************************************************/

struct ECGData
{
    int32_t ch1;
    int32_t ch2;

    uint8_t status[3];

    bool leadOff;       // true = electrodes not in contact with skin

    bool newData;
};

/**********************************************************
 * IMU Data
 **********************************************************/

struct IMUData
{
    int16_t ax;
    int16_t ay;
    int16_t az;

    int16_t gx;
    int16_t gy;
    int16_t gz;

    int16_t temperature;

    bool newData;
};

/**********************************************************
 * Body Temperature
 **********************************************************/

struct TemperatureData
{
    float bodyTemperature;

    bool newData;
};

/**********************************************************
 * Complete Sensor Packet
 **********************************************************/

struct SensorPacket
{
    ECGData ecg;

    IMUData imu;

    TemperatureData temperature;

    uint32_t timestamp;
};

#endif