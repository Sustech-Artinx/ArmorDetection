#include "ArmorDetectionApp.h"
#include <iostream>
#include <memory>
#include <tclap/CmdLine.h>

typedef std::unique_ptr<ArmorDetectionApp> Ptr;

using namespace TCLAP;
using std::string;
using std::vector;

int main(int argc, const char **argv) {
    CmdLine cmd("Armor Detection, Artinx CV 2018", ' ', " C++ v0.2");

    ValueArg<string> colorArg("c", "color", "Color of armor", true, "b", "color", cmd);
    SwitchArg debugArg("d", "debug", "Provide debug information", cmd, false);

    ValueArg<string> imgPathArg("i", "img", "Directory of images folder", true, "./img/", "/path/to/images/folder");
    ValueArg<string> videoPathArg("v", "video", "Path of video file", true, "armor.mp4", "/path/to/video");
    ValueArg<int> cameraIdxArg("l", "live", "Camera index", true, 0, "index");
    vector<Arg*> modeArgs = { &imgPathArg, &videoPathArg, &cameraIdxArg };
    cmd.xorAdd(modeArgs);

    Ptr app;
    try {
        cmd.parse(argc, argv);

        string& color = colorArg.getValue();
        BarColor armorColor;
        if (color == "r" || color == "red" || color == "RED")
            armorColor = BarColor::RED;
        else if (color == "b" || color == "blue" || color == "BLUE")
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
        } else if (cameraIdxArg.isSet()) {
            app = Ptr(new LiveArmorDetectionApp(
                    armorColor,
                    cv::Size(640, 480),
                    cameraIdxArg.getValue(),
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