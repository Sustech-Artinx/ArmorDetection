#ifndef ARMORDETECTION_DETECTOR_H
#define ARMORDETECTION_DETECTOR_H

#include <forward_list>
#include <functional>
#include <iostream>
#include <tuple>
#include <opencv2/opencv.hpp>


namespace ArmorDetection {

    using std::string;
    using cv::Mat;
    using cv::Point;
    using cv::Scalar;
    using cv::Size;
    using cv::RotatedRect;

    enum BarColor {
        BLUE, RED
    };

    namespace MarkerBgrColor {
        const Scalar BLUE(255, 0, 0);
        const Scalar GREEN(0, 255, 0);
        const Scalar RED(0, 0, 255);
        const Scalar YELLOW(0, 255, 255);
        const Scalar PURPLE(255, 0, 255);
    }

    class Detector final {
    public:

        Detector(BarColor color, bool debug) : color(color), debug(debug), currentFrame(nullptr) {}

        void hsv(const Mat &frame, Mat &lightArea, Mat &rightArea);

        bool target(const Mat &frame, Point &targetCenter);

        std::forward_list<RotatedRect> getLights(Mat &binaryImg);

        std::tuple<Point, float> haloCircle(Mat &binaryImg);

        std::forward_list<RotatedRect> selectLightsInHalo(
                std::forward_list<RotatedRect> &lights, const Point &haloCenter, float haloRadius);

        std::forward_list<RotatedRect> selectVerticalLights(std::forward_list<RotatedRect> &lights);

        bool getArmor(const std::forward_list<RotatedRect> &lights, Point &target);

        const std::forward_list<std::tuple<string, Mat>> &getDebugImgs() const {
            return debugImgs;
        }

    private:
        BarColor color;
        bool debug;
        const Mat *currentFrame;
        std::forward_list<std::tuple<string, Mat>> debugImgs;

        void debugPrint(const string &msg) const {
            if (debug) std::cerr << "Debug>>> " << msg << std::endl;
        }

        void addDebugImg(const string &title, const Mat &img) {
            if (debug) debugImgs.push_front(std::make_tuple(title, img));
        }

        void addDebugImg(const string &title, const std::function<Mat()> &getImg) {
            if (debug) debugImgs.push_front(std::make_tuple(title, getImg()));
        }
    };

}


#endif //ARMORDETECTION_DETECTOR_H
