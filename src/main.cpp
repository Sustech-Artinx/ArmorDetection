#include <iostream>
#include <memory>
#include "ArmorDetectionApp.h"
#include <tclap/CmdLine.h>

typedef std::unique_ptr<ArmorDetectionApp> Ptr;
using namespace TCLAP;

int main(int argc, const char **argv) {
    CmdLine cmd("Armor Detection, Artinx CV 2018", ' ', " C++ v0.2");
    ValueArg<std::string> imgPathArg("i", "img", "Directory of images folder", false,
                                     "./img/", "/path/to/images/folder");
    ValueArg<std::string> videoPathArg("v", "video", "Path of video file", false,
                                       "armor.mp4", "/path/to/video");
    cmd.xorAdd(imgPathArg, videoPathArg);
    ValueArg<char> colorArg("c", "color", "Color of armor, 'b' for blue, 'r' for red", true, 'b', "color", cmd);
    SwitchArg debugArg("d", "debug", "Provide debug information", cmd, false);

    Ptr app;

    try {
        cmd.parse(argc, argv);

        char color = colorArg.getValue();
        BarColor armorColor;
        if (color == 'r')
            armorColor = BarColor::RED;
        else if (color == 'b')
            armorColor = BarColor::BLUE;
        else
            throw ArgException("invalid color", "c");

        if (videoPathArg.isSet()) {
            app = Ptr(new VideoArmorDetectionApp(
                    armorColor,
                    cv::Size(640, 480),
                    videoPathArg.getValue(),
                    debugArg.getValue()
            ));
        } else if (imgPathArg.isSet()) {
            app = Ptr(new ImgArmorDetectionApp(
                    armorColor,
                    cv::Size(640, 480),
                    imgPathArg.getValue(),
                    "jpg",  // TODO
                    debugArg.getValue()
            ));
        }

        app->run();

    } catch (ArgException &e) {
        std::cerr << "error: " << e.error() << " for " << e.argId() << std::endl;
    } catch (std::invalid_argument &e) {
        std::cerr << "invalid path: " << e.what() << std::endl;
    } catch (std::exception &e) {
        std::cerr << "runtime error: " << e.what() << std::endl;
    }

    return 0;
}