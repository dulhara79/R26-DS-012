#ifndef CRC_H
#define CRC_H

#include <Arduino.h>

class CRC
{
public:

    // CRC-16/CCITT-FALSE
    static uint16_t CRC16(const uint8_t* data, size_t length);

    // CRC-8
    static uint8_t CRC8(const uint8_t* data, size_t length);
};

#endif