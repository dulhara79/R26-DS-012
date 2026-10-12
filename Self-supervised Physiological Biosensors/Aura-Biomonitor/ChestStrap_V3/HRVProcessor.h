#ifndef HRVPROCESSOR_H
#define HRVPROCESSOR_H

#include <Arduino.h>
#include <math.h>

#define HRV_WINDOW_SIZE 20

class HRVProcessor {
public:
    HRVProcessor();
    
    void addRRInterval(float rr_ms);
    
    float getMeanHR() const;
    float getMeanRR() const;
    float getSDNN() const;
    float getRMSSD() const;

    void reset();
    
private:
    float rr_buffer[HRV_WINDOW_SIZE];
    int head;
    int count;
    
    float meanRR;
    float sdnn;
    float rmssd;
    float meanHR;

    void recalculate();
};

#endif
