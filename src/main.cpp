#include <iostream>
#include <memory>
#include "ArmorDetectionApp.h"

typedef std::unique_ptr<ArmorDetectionApp> Ptr;

int main(int argc, char **argv) {
    Ptr app;

    try {
//        app = Ptr(new ImgArmorDetectionApp(
//                BarColor::BLUE,
//                cv::Size(640, 480),
//                "/home/jeeken/Pictures/blue",
//                "jpg",
//                true
//        ));
        app = Ptr(new VideoArmorDetectionApp(
                BarColor::BLUE,
                cv::Size(640, 480),
                "/home/jeeken/Videos/live_blue.avi",
                false
        ));

        app->run();
    } catch (std::exception &e) {
        std::cerr << "Sorry: " << e.what() << std::endl;
    }

    return 0;
}