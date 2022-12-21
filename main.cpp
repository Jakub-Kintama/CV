#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main() {
    using namespace cv;
    using namespace std;
    VideoCapture vc(0);
    if (!vc.isOpened()) return 1;
    Mat frame, src, dst, detected, dilated, lined;
    std::vector<int> lower = {130, 60, 20};
    std::vector<int> upper = {174, 180, 80};

    namedWindow("first", WINDOW_AUTOSIZE);
    namedWindow("detected", WINDOW_AUTOSIZE);
    createTrackbar("lh", "detected", &lower[0], 255);
    createTrackbar("ls", "detected", &lower[1], 255);
    createTrackbar("lv", "detected", &lower[2], 255);
    createTrackbar("hh", "detected", &upper[0], 255);
    createTrackbar("hs", "detected", &upper[1], 255);
    createTrackbar("hv", "detected", &upper[2], 255);

    while (waitKey(10) != 27) {
        vc >> frame;
        src = frame;
        flip(src, dst, 1);
        imshow("Original", dst);
        inRange(dst, lower, upper, detected);
        imshow("detected", detected);
        auto kernel = getStructuringElement(MORPH_ELLIPSE, Size{5, 5});
        dilate(detected, dilated, kernel);
        erode(dilated, dilated, kernel);
        imshow("dilated", dilated);
        vector<vector<Size>> contours, restOfContours;
        vector<Vec4i> hierarchy;


        findContours(dilated, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);


        std::sort(contours.begin(), contours.end(), [](auto first, auto second) {
            return contourArea(first) > contourArea(second);
        });

        auto Moment = cv::moments(contours[0]);
        auto contourX1 = int(Moment.m10 / Moment.m00);
        auto contourY1 = int(Moment.m01 / Moment.m00);

        auto Moment2 = cv::moments(contours[1]);
        auto contourX2 = int(Moment2.m10 / Moment2.m00);
        auto contourY2 = int(Moment2.m01 / Moment2.m00);

        lined = dst;

        int distanceYDifference = contourY1 - contourY2;


        if (distanceYDifference < 100 && distanceYDifference > -100) {
            line(lined, {contourX1, contourY1}, {contourX2, contourY2}, Scalar(0, 0, 255), 3, LINE_AA);
            line(lined, {contourX1, contourY1 - 10}, {contourX2, contourY2 - 10}, Scalar(0, 255, 255), 2, LINE_AA);
            line(lined, {contourX1, contourY1 + 10}, {contourX2, contourY2 + 10}, Scalar(255, 0, 255), 4, LINE_AA);
        }

        imshow("lined", lined);
    }
    return 0;
}