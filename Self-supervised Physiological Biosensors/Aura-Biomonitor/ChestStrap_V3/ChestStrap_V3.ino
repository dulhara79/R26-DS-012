/*********************************************************************
 *  Chest Strap V3
 *
 *  Hardware
 *  ----------
 *  ESP32-C3 Super Mini
 *  ADS1292R
 *  MPU6500
 *  DS18B20
 *
 *********************************************************************/

#include "Config.h"
#include "DataTypes.h"

#include "Scheduler.h"

#include "ADS1292R.h"
#include "DS18B20.h"
#include "MPU6500.h"

#include "BLE_UART.h"
#include "HealthEngine.h"

#include "CRC.h"
#include "Utils.h"

/**********************************************************
 * Objects
 **********************************************************/

ADS1292R ecg;

MPU6500 imu;

DS18B20 temperature;

BLE_UART ble;

HealthEngine health;

/**********************************************************
 * Global Packet
 **********************************************************/

SensorPacket sensorPacket;

/**********************************************************
 * Device Status
 **********************************************************/

bool ecgOK = false;

bool imuOK = false;

bool tempOK = false;

bool bleOK = false;

/**********************************************************
 * Setup
 **********************************************************/

void setup() {
  DBG_BEGIN(115200);

  delay(1500);

  DBG_PRINTLN();
  DBG_PRINTLN("=================================");
  DBG_PRINTLN(FW_NAME);
  DBG_PRINT("Firmware : ");
  DBG_PRINTLN(FW_VERSION);
  DBG_PRINTLN("=================================");
  DBG_PRINTLN();

  Scheduler::begin();

  /************************************************
   * ADS1292R
   ************************************************/

  DBG_PRINT("Initializing ADS1292R... ");

  ecgOK = ecg.begin();

  if (ecgOK)
    DBG_PRINTLN("OK");
  else
    DBG_PRINTLN("FAILED");

  /************************************************
   * MPU6500
   ************************************************/

  DBG_PRINT("Initializing MPU6500... ");

  imuOK = imu.begin();

  if (imuOK)
    DBG_PRINTLN("OK");
  else
    DBG_PRINTLN("FAILED");

  /************************************************
   * DS18B20
   ************************************************/

  DBG_PRINT("Initializing DS18B20... ");

  tempOK = temperature.begin();

  if (tempOK)
    DBG_PRINTLN("OK");
  else
    DBG_PRINTLN("FAILED");

  /************************************************
   * BLE
   ************************************************/

  DBG_PRINT("Initializing BLE... ");

  bleOK = ble.begin("ChestStrap_V3");

  if (bleOK)
    DBG_PRINTLN("OK");
  else
    DBG_PRINTLN("FAILED");

  DBG_PRINTLN();
  DBG_PRINTLN("System Ready");
  DBG_PRINTLN();
}

/**********************************************************
 * Loop
 **********************************************************/

void loop() {
  Scheduler::update();

  sensorPacket.timestamp = millis();

  /************************************************
   * ECG
   ************************************************/

  if (ecgOK) {
    if (ecg.available()) {
      ecg.read(sensorPacket.ecg);
    }
  } else {
    // ECG hardware not available - treat as lead-off.
    sensorPacket.ecg.leadOff = true;
  }

  /************************************************
   * IMU
   ************************************************/

  if (imuOK) {
    if (Scheduler::IMU_Task()) {
      imu.update();

      imu.read(sensorPacket.imu);
    }
  }

  /************************************************
   * Temperature
   ************************************************/

  if (tempOK) {
    if (Scheduler::TEMP_Task()) {
      temperature.update();

      temperature.read(sensorPacket.temperature);
    }
  }

  /************************************************
   * Packet
   ************************************************/

  health.update(sensorPacket.ecg, sensorPacket.imu, sensorPacket.temperature,
                sensorPacket.timestamp);

  sensorPacket.ecg.newData = false;
  sensorPacket.imu.newData = false;
  sensorPacket.temperature.newData = false;
  ble.update();

  /************************************************
   * Serial Debug
   ************************************************/

  if (Scheduler::BLE_Task()) {
    // ADS1292R status format:
    // 1100 + LOFF_STAT[4:0] + GPIO[1:0] + 13 zero bits.
    const uint8_t loffStat =
        (uint8_t)(((sensorPacket.ecg.status[0] & 0x0F) << 1) |
                  ((sensorPacket.ecg.status[1] >> 7) & 0x01));

    DBG_PRINTF(
        "ST: %02X %02X %02X | LOFFSTAT:%02X | LOFF:%d | CH1:%ld | CH2:%ld\r\n",
        sensorPacket.ecg.status[0], sensorPacket.ecg.status[1],
        sensorPacket.ecg.status[2], loffStat, sensorPacket.ecg.leadOff ? 1 : 0,
        (long)sensorPacket.ecg.ch1, (long)sensorPacket.ecg.ch2);

    String csv = health.toCSV();

    // The model payload is longer. Debug output must not stall ECG reads
    // while the serial transmit buffer drains.
#if DEBUG_ENABLED
    if (Serial.availableForWrite() >= (int)csv.length()) DBG_PRINT(csv);
#endif

    if (ble.connected()) {
      ble.send(csv);
    }
  }
}
