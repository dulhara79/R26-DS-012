#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <Arduino.h>

class Scheduler
{
public:

    static void begin();

    static void update();

    static bool ECG_Task();

    static bool IMU_Task();

    static bool TEMP_Task();

    static bool BLE_Task();

private:

    static uint32_t previousECG;

    static uint32_t previousIMU;

    static uint32_t previousTEMP;

    static uint32_t previousBLE;
};

#endif