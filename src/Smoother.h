#ifndef ARMORDETECTION_SMOOTHER_H
#define ARMORDETECTION_SMOOTHER_H

#include <opencv2/opencv.hpp>
#include "FixedQueue.h"

namespace ArmorDetection {

    class Smoother {
    public:
        explicit Smoother(cv::Size shape);

        bool smooth(bool hasValue, cv::Point &newTarget);

    private:
        class RecentPointsQueue : public FixedQueue<cv::Point> {
        public:
            explicit RecentPointsQueue(size_t maxSize)
                    : FixedQueue<cv::Point>(maxSize) {}

            cv::Point weightedAvg() const;
        };

        const float farEnough;
        RecentPointsQueue recentPoints;
        static const int CNT_THRESHOLD = 5;
    };

}

#endif //ARMORDETECTION_SMOOTHER_H
