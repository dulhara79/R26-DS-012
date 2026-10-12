#include "CRC.h"

/**********************************************************
 * CRC16-CCITT-FALSE
 *
 * Polynomial : 0x1021
 * Initial    : 0xFFFF
 * XOR Out    : 0x0000
 **********************************************************/

uint16_t CRC::CRC16(const uint8_t* data, size_t length)
{
    uint16_t crc = 0xFFFF;

    while (length--)
    {
        crc ^= ((uint16_t)(*data++)) << 8;

        for (uint8_t i = 0; i < 8; i++)
        {
            if (crc & 0x8000)
                crc = (crc << 1) ^ 0x1021;
            else
                crc <<= 1;
        }
    }

    return crc;
}

/**********************************************************
 * CRC8
 *
 * Polynomial : 0x07
 **********************************************************/

uint8_t CRC::CRC8(const uint8_t* data, size_t length)
{
    uint8_t crc = 0x00;

    while (length--)
    {
        crc ^= *data++;

        for (uint8_t i = 0; i < 8; i++)
        {
            if (crc & 0x80)
                crc = (crc << 1) ^ 0x07;
            else
                crc <<= 1;
        }
    }

    return crc;
}