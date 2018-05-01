#ifndef ARMORDETECTION_APP_H
#define ARMORDETECTION_APP_H

#include "Detector.h"
#include "Smoother.h"

namespace ArmorDetection {

    class ArmorDetectionApp {
    public:
        explicit ArmorDetectionApp(BarColor color, bool debug = false)
                : detector(color, debug),
                  color(color),
                  debug(debug) {}

        virtual ~ArmorDetectionApp() {}

        virtual void run() = 0;

    protected:
        Detector detector;
        BarColor color;
        bool debug;

        void drawTarget(Mat &img, const Point &target);
    };


    class ImgArmorDetectionApp final : public ArmorDetectionApp {
    public:
        ImgArmorDetectionApp(BarColor color, const Size &frameSize, const string &folder = ".",
                             const string &ext = "jpg", bool debug = false);

        void run() override;

    private:
        Size frameSize;
        std::forward_list<string> files;
    };


    class VideoArmorDetectionApp final : public ArmorDetectionApp {
    public:
        VideoArmorDetectionApp(BarColor color, const Size &frameSize, const string &file, bool debug = false);

        void run() override;

    private:
        Size frameSize;
        string file;
        Smoother smoother;
    };

} // namespace ArmorDetection

using ArmorDetection::ArmorDetectionApp;
using ArmorDetection::ImgArmorDetectionApp;
using ArmorDetection::VideoArmorDetectionApp;
using ArmorDetection::BarColor;

#endif //ARMORDETECTION_APP_H
