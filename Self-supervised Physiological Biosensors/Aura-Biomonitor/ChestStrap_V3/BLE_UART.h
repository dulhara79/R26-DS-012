#ifndef BLE_UART_H
#define BLE_UART_H

#include <Arduino.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

class BLE_UART
{
public:

    BLE_UART();

    bool begin(const String& deviceName = "ChestStrap_V3");

    void update();

    bool connected();

    void send(const String& message);

private:

    BLEServer* server;

    BLEService* service;

    BLECharacteristic* txCharacteristic;

    BLECharacteristic* rxCharacteristic;

    bool deviceConnected;
    String pendingMessage;
    size_t pendingOffset = 0;
    uint32_t lastNotifyMs = 0;

    class ServerCallbacks : public BLEServerCallbacks
    {
    public:

        ServerCallbacks(BLE_UART* ble);

        void onConnect(BLEServer* server);

        void onDisconnect(BLEServer* server);

    private:

        BLE_UART* parent;
    };
};

#endif