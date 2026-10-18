#include "BLE_UART.h"

#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_RX_UUID "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_TX_UUID "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

// Keep notifications <= 20 bytes so the firmware remains compatible with
// the minimum BLE ATT MTU (23 bytes, 20 bytes payload), even if MTU
// negotiation is unavailable or fails on the phone.
static const size_t BLE_CHUNK_SIZE = 20;

BLE_UART::BLE_UART()
{
    server = nullptr;
    service = nullptr;
    txCharacteristic = nullptr;
    rxCharacteristic = nullptr;
    deviceConnected = false;
}

BLE_UART::ServerCallbacks::ServerCallbacks(BLE_UART* ble)
{
    parent = ble;
}

void BLE_UART::ServerCallbacks::onConnect(BLEServer* server)
{
    parent->deviceConnected = true;
}

void BLE_UART::ServerCallbacks::onDisconnect(BLEServer* server)
{
    parent->deviceConnected = false;
    BLEDevice::startAdvertising();
}

bool BLE_UART::begin(const String& deviceName)
{
    BLEDevice::init(deviceName.c_str());
    server = BLEDevice::createServer();
    server->setCallbacks(new ServerCallbacks(this));
    service = server->createService(SERVICE_UUID);

    txCharacteristic = service->createCharacteristic(
        CHARACTERISTIC_TX_UUID,
        BLECharacteristic::PROPERTY_NOTIFY);
    txCharacteristic->addDescriptor(new BLE2902());

    rxCharacteristic = service->createCharacteristic(
        CHARACTERISTIC_RX_UUID,
        BLECharacteristic::PROPERTY_WRITE);

    service->start();
    BLEAdvertising* advertising = BLEDevice::getAdvertising();
    advertising->addServiceUUID(SERVICE_UUID);
    advertising->start();
    return true;
}

void BLE_UART::update()
{
    if (!deviceConnected) {
        pendingMessage = "";
        pendingOffset = 0;
        return;
    }
    if (txCharacteristic == nullptr || pendingOffset >= pendingMessage.length() ||
        uint32_t(millis() - lastNotifyMs) < 2) return;

    // Send one chunk per loop, interleaving ECG acquisition. The old delay(2)
    // loop blocked acquisition for the entire CSV, especially with model fields.
    const size_t length = min(BLE_CHUNK_SIZE, pendingMessage.length() - pendingOffset);
    String chunk = pendingMessage.substring(pendingOffset, pendingOffset + length);
    txCharacteristic->setValue(chunk.c_str());
    txCharacteristic->notify();
    pendingOffset += length;
    lastNotifyMs = millis();
    if (pendingOffset == pendingMessage.length()) {
        pendingMessage = "";
        pendingOffset = 0;
    }
}

bool BLE_UART::connected()
{
    return deviceConnected;
}

void BLE_UART::send(const String& message)
{
    if(!deviceConnected || txCharacteristic == nullptr)
        return;

    // Never interrupt an in-flight CSV line with another line.
    if (pendingMessage.length() != 0) return;
    pendingMessage = message;
    if (!pendingMessage.endsWith("\n")) pendingMessage += "\n";
    pendingOffset = 0;
}
