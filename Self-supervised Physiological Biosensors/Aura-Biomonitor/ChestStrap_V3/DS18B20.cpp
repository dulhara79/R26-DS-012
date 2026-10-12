#include "DS18B20.h"

DS18B20::DS18B20() :
    oneWire(TEMP_PIN),
    sensors(&oneWire)
{
    initialized = false;
    conversionRequested = false;
    lastRequestTime = 0;
}

/************************************************/

bool DS18B20::begin()
{
    sensors.begin();

    if (sensors.getDeviceCount() == 0)
    {
        return false;
    }

    // Non-blocking temperature conversion
    sensors.setWaitForConversion(false);

    initialized = true;

    return true;
}

/************************************************/

bool DS18B20::available()
{
    return initialized;
}

/************************************************/

void DS18B20::update()
{
    if (!initialized)
        return;

    if (!conversionRequested)
    {
        sensors.requestTemperatures();

        lastRequestTime = millis();

        conversionRequested = true;

        return;
    }

    // Wait for conversion (~750ms max for 12-bit)

    if (millis() - lastRequestTime >= 800)
    {
        tempData.bodyTemperature = sensors.getTempCByIndex(0);

        tempData.newData = true;

        conversionRequested = false;
    }
}

/************************************************/

bool DS18B20::read(TemperatureData &data)
{
    if (!initialized)
        return false;

    if (!tempData.newData)
        return false;

    data = tempData;

    tempData.newData = false;

    return true;
}