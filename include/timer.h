#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsedMs() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time
        );
        return duration.count() / 1000.0;
    }
    
    double elapsedSeconds() const {
        return elapsedMs() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

#endif // TIMER_H

