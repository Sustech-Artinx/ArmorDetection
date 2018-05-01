#include "Smoother.h"
#include "utils.h"

using namespace ArmorDetection;
using utils::sqr;


Smoother::Smoother(cv::Size shape)
        : farEnough(std::sqrt(float(sqr(shape.width) + sqr(shape.height))) / 12),  // diagonal / 12
          recentPoints(CNT_THRESHOLD) {}


bool Smoother::smooth(bool hasValue, cv::Point &target) {
    if (recentPoints.isEmpty()) {
        if (hasValue) recentPoints.enQueue(target);
        return hasValue;  // no recent values, adopt new target
    }

    // recentPoints not empty
    cv::Point recent = recentPoints.weightedAvg();
    if (hasValue && cv::norm(recent - target) < farEnough) {
        recentPoints.enQueue(target);
        target = recentPoints.weightedAvg();
    } else {
        target = recent;
        recentPoints.deQueue();
    }
    return true;
}


cv::Point Smoother::RecentPointsQueue::weightedAvg() const {
    // assert(!isEmpty());
    size_t p;
    int cnt, sum;
    cv::Point point(0, 0);
    for (p = head, cnt = 1, sum = 0; cnt <= size; p = (p + 1) % MAX_SIZE, sum += cnt++)
        point += data[p] * cnt;
    return point / sum;
}