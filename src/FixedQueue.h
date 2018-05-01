#ifndef ARMORDETECTION_FIXEDQUEUE_H
#define ARMORDETECTION_FIXEDQUEUE_H

#include <cassert>
#include <cstddef>

template<typename T>
class FixedQueue {
public:
    explicit FixedQueue(size_t maxSize) : MAX_SIZE(maxSize) {
        head = rear = size = 0;
        data = new T[MAX_SIZE];
    }

    virtual ~FixedQueue() {
        delete[] data;
    }

    void enQueue(const T &e) {
        data[rear] = e;
        if (isFull())
            head = (head + 1) % MAX_SIZE;
        else
            ++size;
        rear = (rear + 1) % MAX_SIZE;
    }

    void deQueue() {
        // assert(!isEmpty());
        head = (head + 1) % MAX_SIZE;
        --size;
    }

    bool isEmpty() const {
        return size == 0;
    }

    bool isFull() const {
        return size == MAX_SIZE;
    }

protected:
    T *data;
    size_t head, rear, size;

    const size_t MAX_SIZE;
};

#endif //ARMORDETECTION_FIXEDQUEUE_H
