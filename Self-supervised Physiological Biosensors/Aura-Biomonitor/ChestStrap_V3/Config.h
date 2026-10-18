#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

/**********************************************************
 *                  Firmware Information
 **********************************************************/

#define FW_NAME        "Chest Strap V3"
#define FW_VERSION     "1.0.2"

/**********************************************************
 *                     Debug
 **********************************************************/

#define DEBUG_ENABLED true

#if DEBUG_ENABLED
    #define DBG_BEGIN(x)        Serial.begin(x)
    #define DBG_PRINT(x)        Serial.print(x)
    #define DBG_PRINTLN(x)      Serial.println(x)
    #define DBG_PRINTF(...)     Serial.printf(__VA_ARGS__)
#else
    #define DBG_BEGIN(x)
    #define DBG_PRINT(x)
    #define DBG_PRINTLN(x)
    #define DBG_PRINTF(...)
#endif

/**********************************************************
 *                    ADS1292R
 **********************************************************/

#define ECG_CS_PIN          10
#define ECG_START_PIN        0
#define ECG_RESET_PIN        1
#define ECG_DRDY_PIN         3

#define ECG_SCLK_PIN         4
#define ECG_MISO_PIN         5
#define ECG_MOSI_PIN         6

#define ECG_SPI_SPEED    100000

/**********************************************************
 *                    MPU6500
 **********************************************************/

#define IMU_SDA_PIN          9
#define IMU_SCL_PIN          8

#define MPU6500_ADDRESS   0x69
#define IMU_I2C_SPEED   100000

/**********************************************************
 *                    DS18B20
 **********************************************************/

#define TEMP_PIN             2

/**********************************************************
 *                Off-Body Detection
 **********************************************************/

#define SKIN_TEMP_THRESHOLD  30.0f   // °C - diagnostic threshold

/**********************************************************
 *                 Sampling Frequencies
 **********************************************************/

#define ECG_SAMPLE_RATE      500
#define IMU_SAMPLE_RATE      100
#define TEMP_SAMPLE_RATE       1
#define BLE_SAMPLE_RATE        1

#endif
