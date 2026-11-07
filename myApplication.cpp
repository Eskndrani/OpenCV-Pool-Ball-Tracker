#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

void MyApplication()
{
    // Load the input image and define the table corners in the image.
    string inputImagePath = "data/images/Table1.jpeg";
    cv::Mat inputImage = cv::imread(inputImagePath);
    if (inputImage.empty()) {
        std::cout << "Error loading image\n";
        return;
    }
    // Define the table corners in the image (top-left, top-right, bottom-right, bottom-left).
    vector<cv::Point2f> tableCorners = {
        cv::Point2f(337, 35), 
        cv::Point2f(1570, 21), 
        cv::Point2f(1570, 451), 
        cv::Point2f(38, 472)    
    };
    // Define destination corners for the perspective transform (top-left, top-right, bottom-right, bottom-left).
    vector<cv::Point2f> destCorners = {
        cv::Point2f(0, 0),
        cv::Point2f(1434, 0),
        cv::Point2f(1434, 450),
        cv::Point2f(0, 450)
    };

    cv::Mat perspectiveTransform = cv::getPerspectiveTransform(tableCorners, destCorners);

    cv::Mat warpedImage;
    cv::warpPerspective(inputImage, warpedImage, perspectiveTransform, cv::Size(1434,  683));

    cv::imwrite("results/warped_Table1.jpeg", warpedImage);

    // TODO: Stage 1 - Isolate the green playing surface and remove rails,
    // pockets, markings, and background from further processing.

    // TODO: Stage 1 - Detect ball candidates, estimate their centres, and map
    // the centres into transformed table coordinates.

    // TODO: Stage 2 - Recognise the detected balls (labels 1-15 and cue ball)
    // using the reference image and the method taught in the module.

    // TODO: Evaluate against ground truth and report detection, localisation,
    // and recognition metrics.

    std::cout << "Pool ball assignment template: implementation pending." << std::endl;
}

#ifndef MYAPPLICATION_NO_MAIN
int main()
{
    MyApplication();
    return 0;
}
#endif
