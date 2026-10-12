#include "Packet.h"

Packet::Packet()
{
    memset(&currentPacket, 0, sizeof(currentPacket));
}

void Packet::update(const SensorPacket &packet)
{
    currentPacket = packet;
}

const SensorPacket& Packet::getPacket() const
{
    return currentPacket;
}

String Packet::toCSV()
{
    String csv;

    csv.reserve(150);

    csv += String(currentPacket.timestamp);
    csv += ",";

    csv += String(currentPacket.ecg.ch1);
    csv += ",";
    csv += String(currentPacket.ecg.ch2);
    csv += ",";

    csv += String(currentPacket.imu.ax);
    csv += ",";
    csv += String(currentPacket.imu.ay);
    csv += ",";
    csv += String(currentPacket.imu.az);
    csv += ",";

    csv += String(currentPacket.imu.gx);
    csv += ",";
    csv += String(currentPacket.imu.gy);
    csv += ",";
    csv += String(currentPacket.imu.gz);
    csv += ",";

    csv += String(currentPacket.temperature.bodyTemperature, 2);
    csv += "\n";

    return csv;
}