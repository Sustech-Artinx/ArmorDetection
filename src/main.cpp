#include <iostream>
#include "ArmorDetectionApp.h"

int main(int argc, char **argv) {

//    ImgArmorDetectionApp app(
//            BarColor::BLUE,
//            cv::Size(640, 480),
//            "/home/jeeken/Pictures/blue_dark",
//            "jpg",
//            false
//    );

    VideoArmorDetectionApp app(
            BarColor::BLUE,
            cv::Size(640, 480),
            "/home/jeeken/Videos/live_blue.avi",
            false
    );
    app.run();
    return 0;
}