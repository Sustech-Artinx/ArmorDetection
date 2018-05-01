#include "ArmorDetectionApp.h"
#include "utils.h"

using namespace ArmorDetection;

void ArmorDetectionApp::drawTarget(Mat &img, const Point &target) {
    Scalar aimColor;
    if (img.channels() == 1)
        aimColor = 255;
    else
        aimColor = color == BarColor::BLUE ? MarkerBgrColor::RED : MarkerBgrColor::GREEN;
    cv::circle(img, target, 20, aimColor, 2);
    cv::drawMarker(img, target, aimColor, cv::MARKER_CROSS, 80, 2);
}


ImgArmorDetectionApp::ImgArmorDetectionApp(BarColor color, const Size &frameSize,
                                           const string &folder, const string &ext, bool debug)
        : ArmorDetectionApp(color, debug),
          frameSize(frameSize),
          files(utils::getFilesFromFolder(folder, ext)) {}


void ImgArmorDetectionApp::run() {
    for (string &filename: files) {
        std::cout << "Image: " << filename << std::endl;
        Mat img;
        cv::resize(cv::imread(filename), img, frameSize);

        cv::imshow("Origin", img);

        Point target;
        if (detector.target(img, target)) ArmorDetectionApp::drawTarget(img, target);

        if (debug)
            for (auto &imgAndTitle: detector.getDebugImgs())
                cv::imshow(std::get<0>(imgAndTitle), std::get<1>(imgAndTitle));

        cv::imshow("Aimed", img);

        if (cv::waitKey() == 'q') {
            cv::destroyAllWindows();
            break;
        }
    }
}


VideoArmorDetectionApp::VideoArmorDetectionApp(BarColor color, const Size &frameSize, const string &file, bool debug)
        : ArmorDetectionApp(color, debug),
          frameSize(frameSize),
          file(file),
          smoother(frameSize) {
    if (!utils::fileExists(file))
        throw std::invalid_argument("The file does not exists!");
}


void VideoArmorDetectionApp::run() {
    auto cap = cv::VideoCapture(file);
    while (cap.isOpened()) {
        Mat frame;
        if (!cap.read(frame)) break;
        cv::resize(frame, frame, frameSize);
        cv::imshow("Original", frame);

        Point target;
        bool hasTarget = detector.target(frame, target);

        if (debug)
            for (auto &debugInfo: detector.getDebugImgs())
                cv::imshow(std::get<0>(debugInfo), std::get<1>(debugInfo));

        if (hasTarget) {
            std::cout << "Target:   " << target << std::endl;
            cv::circle(frame, target, 4, MarkerBgrColor::GREEN, -1);
        } else {
            std::cout << "No Target" << std::endl;
        }

        bool hasValidTarget = smoother.smooth(hasTarget, target);
        if (hasValidTarget) {
            ArmorDetectionApp::drawTarget(frame, target);
            std::cout << "Smoothed: " << target << std::endl;
        } else {
            std::cout << "No Smoothed" << std::endl;
        }
        std::cout << "--------------------" << std::endl;

        cv::imshow("Aimed", frame);

        if (cv::waitKey(1) == 'q') break;
    }
}