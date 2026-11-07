#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

void MyApplication()
{
    const std::string inputImagePath = "data/images/Table1.jpeg";
    const std::string warpedImagePath = "results/warped_Table1.jpeg";

    const cv::Mat inputImage = cv::imread(inputImagePath);
    if (inputImage.empty())
    {
        std::cerr << "Failed to load image: " << inputImagePath << std::endl;
        return;
    }

    // Table corners in the original image, ordered top-left, top-right,
    // bottom-right, bottom-left. Coordinates are (column, row).
    const std::vector<cv::Point2f> tableCorners = {
        cv::Point2f(337.0F, 35.0F),
        cv::Point2f(1570.0F, 21.0F),
        cv::Point2f(1888.0F, 451.0F),
        cv::Point2f(38.0F, 472.0F)
    };

    // Map the table to a 1434 x 683 image, matching the physical
    // cushion-to-cushion dimensions in millimetres.
    const std::vector<cv::Point2f> destCorners = {
        cv::Point2f(0.0F, 0.0F),
        cv::Point2f(1434.0F, 0.0F),
        cv::Point2f(1434.0F, 683.0F),
        cv::Point2f(0.0F, 683.0F)
    };

    const cv::Mat perspectiveTransform = cv::getPerspectiveTransform(tableCorners, destCorners);

    cv::Mat warpedImage;
    cv::warpPerspective(inputImage, warpedImage, perspectiveTransform, cv::Size(1434, 683));

    if (!cv::imwrite(warpedImagePath, warpedImage))
    {
        std::cerr << "Failed to save image: " << warpedImagePath << std::endl;
        return;
    }

    std::cout << "Saved rectified table image to " << warpedImagePath << std::endl;

    // TODO: Stage 1 - Isolate the green playing surface and remove rails,
    // pockets, markings, and background from further processing.

    // TODO: Stage 1 - Detect ball candidates, estimate their centres, and map
    // the centres into transformed table coordinates.

    // TODO: Stage 2 - Recognise the detected balls (labels 1-15 and cue ball)
    // using the reference image and the method taught in the module.

    // TODO: Evaluate against ground truth and report detection, localisation,
    // and recognition metrics.
}

#ifndef MYAPPLICATION_NO_MAIN
int main()
{
    MyApplication();
    return 0;
}
#endif
