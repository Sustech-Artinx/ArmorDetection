#include "Detector.h"
#include "utils.h"

using namespace ArmorDetection;

bool Detector::target(const Mat &frame, Point &targetCenter) {
    debugImgs.clear();
    currentFrame = &frame;

    // FIXME
    Mat tmp;
    cv::pyrDown(frame, tmp);
    cv::pyrUp(tmp, frame);

    Mat light, halo;
    hsv(frame, light, halo);

    Point haloCenter;
    float haloRadius;
    std::tie(haloCenter, haloRadius) = haloCircle(halo);
    auto lightsInfo = getLights(light);

    // debug
    addDebugImg("Halo", [&halo, &haloCenter, haloRadius]() {
        cv::circle(halo, haloCenter, utils::round(haloRadius), Scalar(255), 2);
        return halo;
    });
    addDebugImg("Light", [&light, &haloCenter, haloRadius]() {
        cv::circle(light, haloCenter, utils::round(haloRadius), Scalar(255), 2);
        return light;
    });

    auto lightsInside = selectLightsInHalo(lightsInfo, haloCenter, haloRadius);
    auto verticalLights = selectVerticalLights(lightsInside);
    return getArmor(verticalLights, targetCenter);
}


void Detector::hsv(const Mat &frame, Mat &lightArea, Mat &haloArea) {
    Mat hsvFrame;
    if (color == BarColor::BLUE) {
        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
        Scalar lowerBlueLight(90, 0, 200), upperBlueLight(120, 160, 255);
        Scalar lowerBlueHalo(90, 120, 185), upperBlueHalo(120, 255, 255);
        cv::inRange(hsvFrame, lowerBlueLight, upperBlueLight, lightArea);
        cv::inRange(hsvFrame, lowerBlueHalo, upperBlueHalo, haloArea);
    } else { // For BarColor::RED
        // cheat openCV: swap Red and Blue channels implicitly
        cv::cvtColor(frame, hsvFrame, cv::COLOR_RGB2HSV);
        Scalar lowerBlueLight(100, 0, 200), upperBlueLight(130, 150, 255);
        Scalar lowerBlueHalo(100, 120, 185), upperBlueHalo(130, 255, 255);
        cv::inRange(hsvFrame, lowerBlueLight, upperBlueLight, lightArea);
        cv::inRange(hsvFrame, lowerBlueHalo, upperBlueHalo, haloArea);
    }
}


std::forward_list<RotatedRect> Detector::getLights(Mat &binaryImg) {
    std::vector<Mat> contours;
    cv::findContours(binaryImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float verySmallArea = binaryImg.size().width / 40.0f;
    auto notVerySmall = [verySmallArea](RotatedRect rect) {
        return rect.size.area() > verySmallArea;
    };
    std::function<RotatedRect(Mat)> minAreaRect = cv::minAreaRect;  // for type matching
    return utils::filter(utils::map(contours, minAreaRect), notVerySmall);
}


std::tuple<Point, float> Detector::haloCircle(Mat &binaryImg) {
    int width = binaryImg.size().width, height = binaryImg.size().height;

    std::vector<Mat> contours;
    cv::findContours(binaryImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float verySmallArea = width / 40.0f;
    auto selectedContours = utils::filter(contours, [verySmallArea](Mat contour) {
        Mat xy[2];
        cv::split(contour, xy);
        double xMin, xMax, yMin, yMax;
        cv::minMaxLoc(xy[0], &xMin, &xMax);
        cv::minMaxLoc(xy[1], &yMin, &yMax);
        return (xMax - xMin) * (yMax - yMin) > verySmallArea;
    });

    Point center;
    float radius;
    if (!selectedContours.empty()) {
        cv::Point2f originalCenter;
        Mat combinedHaloPoints;
        for (const Mat &contour: selectedContours)
            for (auto p = contour.begin<cv::Vec2i>(); p != contour.end<cv::Vec2i>(); ++p)
                combinedHaloPoints.push_back(*p);
        cv::minEnclosingCircle(combinedHaloPoints, originalCenter, radius);

        center = originalCenter;
        if (radius > width * 0.4f) {
            radius = width * 0.25f;  // FIXME in python
        } else if (width / 15.0f < radius && radius < width / 4.0f) {
            radius *= 1.8f;
        } else if (radius <= width / 15.0f) {
            radius = width / 15.0f;
        }
    } else {
        center = Point(width / 2, height / 2);
        radius = width / 3.0f;
    }
    return std::make_tuple(center, radius);
};


std::forward_list<RotatedRect> Detector::selectLightsInHalo(
        std::forward_list<RotatedRect> &lights, const Point &haloCenter, float haloRadius) {
    int x = haloCenter.x, y = haloCenter.y;

    auto lightsInside = utils::filter(lights, [x, y, haloRadius](const RotatedRect &light) {
        return utils::sqr(light.center.x - x) + utils::sqr(light.center.y - y) < utils::sqr(haloRadius);
    });

    addDebugImg("Lights Inside", [this, &lightsInside]() {
        Mat show = this->currentFrame->clone();
        for (auto &light: lightsInside)
            cv::ellipse(show, light, MarkerBgrColor::GREEN, 2);
        return show;
    });

    return lightsInside;
}


std::forward_list<RotatedRect> Detector::selectVerticalLights(std::forward_list<RotatedRect> &lights) {
    std::function<RotatedRect(RotatedRect)> normalizeAngle = [](const RotatedRect &light) {
        return light.size.width > light.size.height
               ? RotatedRect(light.center, cv::Point2f(light.size.height, light.size.width), light.angle + 90)
               : light;
    };
    auto angleNormalizedLights = utils::map(lights, normalizeAngle);
    auto verticalLights = utils::filter(angleNormalizedLights, [](const RotatedRect &light) {
        return -25 < light.angle && light.angle < 25;
    });

    addDebugImg("Vertical Lights", [this, &verticalLights]() {
        Mat show = this->currentFrame->clone();
        // FIXME in python, useless rand_color
        for (auto &light: verticalLights)
            cv::ellipse(show, light, MarkerBgrColor::GREEN, 2);
        return show;
    });

    return verticalLights;
}


float parallel(RotatedRect light1, RotatedRect light2) {
    return std::fabs(light1.angle - light2.angle) / 15.0f;
};

float shapeSimilarity(RotatedRect light1, RotatedRect light2) {
    float w1 = light1.size.width, h1 = light1.size.height;
    float w2 = light2.size.width, h2 = light2.size.height;
    float minWidth = std::min(w1, w2), minHeight = std::min(h1, h2);
    return std::fabs(w1 - w2) / minWidth + 0.33333f * std::fabs(h1 - h2) / minHeight;  // FIXME in python
};

float squareRatio(RotatedRect light1, RotatedRect light2) {
    float x1 = light1.center.x, y1 = light1.center.y;
    float x2 = light2.center.x, y2 = light2.center.y;
    float armorWidth = std::sqrt(utils::sqr(x1 - x2) + utils::sqr(y1 - y2));
    float armorHeight = 0.5f * (light1.size.height + light2.size.height);
    float ratio = armorWidth / armorHeight;
    // FIXME in python: useless abs()
    return ratio > 0.85 ? utils::sqr(ratio - 2.5f) : 1e6f;
};

float yDis(RotatedRect light1, RotatedRect light2) {
    float y1 = light1.center.y, y2 = light2.center.y;
    // FIXME in python: y coordinates may be negetive
    return std::fabs((y1 - y2) / std::min(light1.size.height, light2.size.height));
};

bool Detector::getArmor(const std::forward_list<RotatedRect> &lights, Point &target) {
    std::vector<RotatedRect> candidates;
    for (auto &light: lights)
        candidates.push_back(light);

    if (candidates.size() < 2) return false;

    typedef std::tuple<float, size_t, size_t> Result;  // score, index1, index2
    size_t iEnd = candidates.size() - 1, jEnd = candidates.size();
    std::vector<Result> results(iEnd * jEnd / 2);
    auto p = results.begin();
    for (size_t i = 0; i < iEnd; ++i) {
        for (size_t j = i + 1; j < jEnd; ++j) {
            auto &l1 = candidates[i], &l2 = candidates[j];
            float score = squareRatio(l1, l2) * 5.0f
                          + yDis(l1, l2) * 8.0f
                          + shapeSimilarity(l1, l2) * 3.0f
                          + parallel(l1, l2) * 1.2f;
            *(p++) = std::make_tuple(score, i, j);
        }
    }
    float winnerScore;
    size_t index1, index2;
    std::tie(winnerScore, index1, index2) = *std::min_element(results.begin(), results.end(),
                                                              [](const Result &a, const Result &b) {
                                                                  return std::get<0>(a) < std::get<0>(b);
                                                              });
    debugPrint("score: " + std::to_string(winnerScore));
    if (winnerScore > 7) return false;

    cv::Point2f p1 = candidates[index1].center, p2 = candidates[index2].center;
    addDebugImg("Selected Pair", [this, &p1, &p2]() {
        Mat show = this->currentFrame->clone();
        cv::drawMarker(show, Point(p1), MarkerBgrColor::GREEN, cv::MARKER_CROSS, 15, 2);
        cv::drawMarker(show, Point(p2), MarkerBgrColor::GREEN, cv::MARKER_CROSS, 15, 2);
        return show;
    });
    target = (p1 + p2) / 2;
    return true;
}