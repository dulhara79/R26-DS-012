#ifndef UTILS_H
#define UTILS_H

#include <Arduino.h>
#include <math.h>

namespace Utils
{

/**********************************************************
 * Clamp Value
 **********************************************************/

template<typename T>
inline T clamp(T value, T minimum, T maximum)
{
    if(value < minimum)
        return minimum;

    if(value > maximum)
        return maximum;

    return value;
}

/**********************************************************
 * Map Float
 **********************************************************/

inline float mapFloat(float x,
                      float in_min,
                      float in_max,
                      float out_min,
                      float out_max)
{
    return (x - in_min) *
           (out_max - out_min) /
           (in_max - in_min)
           + out_min;
}

/**********************************************************
 * 24-bit Sign Extension
 **********************************************************/

inline int32_t int24ToInt32(uint32_t value)
{
    value &= 0x00FFFFFF;

    if(value & 0x00800000)
        value |= 0xFF000000;

    return (int32_t)value;
}

/**********************************************************
 * Combine 3 Bytes
 **********************************************************/

inline int32_t bytesToInt24(uint8_t msb,
                            uint8_t mid,
                            uint8_t lsb)
{
    uint32_t value =
            ((uint32_t)msb << 16) |
            ((uint32_t)mid << 8 ) |
             (uint32_t)lsb;

    return int24ToInt32(value);
}

/**********************************************************
 * Combine 2 Bytes
 **********************************************************/

inline int16_t bytesToInt16(uint8_t msb,
                            uint8_t lsb)
{
    return (int16_t)((msb << 8) | lsb);
}

/**********************************************************
 * MPU6500 Temperature
 *
 * Datasheet:
 * Temp (°C) = (RAW / 333.87) + 21
 **********************************************************/

inline float MPU6500Temperature(int16_t raw)
{
    return ((float)raw / 333.87f) + 21.0f;
}

/**********************************************************
 * Degrees ↔ Radians
 **********************************************************/

inline float degreesToRadians(float deg)
{
    return deg * DEG_TO_RAD;
}

inline float radiansToDegrees(float rad)
{
    return rad * RAD_TO_DEG;
}

/**********************************************************
 * Milliseconds Since
 **********************************************************/

inline bool elapsedMillis(uint32_t start,
                          uint32_t period)
{
    return (millis() - start) >= period;
}

/**********************************************************
 * Microseconds Since
 **********************************************************/

inline bool elapsedMicros(uint32_t start,
                          uint32_t period)
{
    return (micros() - start) >= period;
}

/**********************************************************
 * Check Bit
**********************************************************/

inline bool isBitSet(uint32_t value, uint8_t bit)
{
    return (value & (1UL << bit)) != 0;
}

/**********************************************************
 * Set Bit
 **********************************************************/

inline void setBitValue(uint32_t &value, uint8_t bit)
{
    value |= (1UL << bit);
}

/**********************************************************
 * Clear Bit
 **********************************************************/

inline void clearBitValue(uint32_t &value, uint8_t bit)
{
    value &= ~(1UL << bit);
}

/**********************************************************
 * Toggle Bit
 **********************************************************/

inline void toggleBitValue(uint32_t &value, uint8_t bit)
{
    value ^= (1UL << bit);
}

/**********************************************************
 * Absolute Difference
 **********************************************************/

template<typename T>
inline T absDiff(T a,
                 T b)
{
    return (a > b) ? (a - b) : (b - a);
}

/**********************************************************
 * Moving Average (Simple)
 **********************************************************/

inline float average(float a,
                     float b)
{
    return (a + b) * 0.5f;
}

}

#endif