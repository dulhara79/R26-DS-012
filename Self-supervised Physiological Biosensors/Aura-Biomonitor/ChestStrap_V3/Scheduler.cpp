#include "Scheduler.h"
#include "Config.h"

/**********************************************************
 * Timing (microseconds)
 **********************************************************/

#define ECG_PERIOD_US     (1000000UL / ECG_SAMPLE_RATE)
#define IMU_PERIOD_US     (1000000UL / IMU_SAMPLE_RATE)

#define TEMP_PERIOD_US    (1000000UL / TEMP_SAMPLE_RATE)

#define BLE_PERIOD_US     (1000000UL / BLE_SAMPLE_RATE)

/**********************************************************
 * Static Variables
 **********************************************************/

uint32_t Scheduler::previousECG = 0;
uint32_t Scheduler::previousIMU = 0;
uint32_t Scheduler::previousTEMP = 0;
uint32_t Scheduler::previousBLE = 0;

/**********************************************************
 * Begin
 **********************************************************/

void Scheduler::begin()
{
    uint32_t now = micros();

    previousECG = now;
    previousIMU = now;
    previousTEMP = now;
    previousBLE = now;
}

/**********************************************************
 * Update
 **********************************************************/

void Scheduler::update()
{
    // Reserved for future use
}

/**********************************************************
 * ECG Scheduler
 **********************************************************/

bool Scheduler::ECG_Task()
{
    uint32_t now = micros();

    if (now - previousECG >= ECG_PERIOD_US)
    {
        previousECG += ECG_PERIOD_US;
        return true;
    }

    return false;
}

/**********************************************************
 * IMU Scheduler
 **********************************************************/

bool Scheduler::IMU_Task()
{
    uint32_t now = micros();

    if (now - previousIMU >= IMU_PERIOD_US)
    {
        previousIMU += IMU_PERIOD_US;
        return true;
    }

    return false;
}

/**********************************************************
 * Temperature Scheduler
 **********************************************************/

bool Scheduler::TEMP_Task()
{
    uint32_t now = micros();

    if (now - previousTEMP >= TEMP_PERIOD_US)
    {
        previousTEMP += TEMP_PERIOD_US;
        return true;
    }

    return false;
}

/**********************************************************
 * BLE Scheduler
 **********************************************************/

bool Scheduler::BLE_Task()
{
    uint32_t now = micros();

    if (now - previousBLE >= BLE_PERIOD_US)
    {
        previousBLE += BLE_PERIOD_US;
        return true;
    }

    return false;
}