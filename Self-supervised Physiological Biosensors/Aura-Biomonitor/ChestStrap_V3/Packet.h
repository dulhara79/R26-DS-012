#ifndef PACKET_H
#define PACKET_H

#include <Arduino.h>
#include "DataTypes.h"

class Packet
{
public:
    Packet();

    void update(const SensorPacket &packet);

    String toCSV();

    const SensorPacket& getPacket() const;

private:
    SensorPacket currentPacket;
};

#endif