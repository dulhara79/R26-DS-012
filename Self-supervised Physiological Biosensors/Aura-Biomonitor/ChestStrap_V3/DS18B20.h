#ifndef DS18B20_H
#define DS18B20_H

#include <Arduino.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#include "Config.h"
#include "DataTypes.h"

class DS18B20
{
public:

    DS18B20();

    bool begin();

    void update();

    bool read(TemperatureData &data);

    bool available();

private:

    OneWire oneWire;

    DallasTemperature sensors;

    TemperatureData tempData;

    bool initialized;

    uint32_t lastRequestTime;

    bool conversionRequested;
};

#endif