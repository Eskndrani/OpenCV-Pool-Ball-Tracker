#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace
{
struct DetectionMetrics
{
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    int correctLabels = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double recognitionAccuracy = 0.0;
};

struct GroundTruthBall
{
    cv::Point2f center;
    int label = -1;
};

struct ReferenceBall
{
    int label = -1;
    cv::Mat histogram;
};

struct ClassificationResult
{
    int label = -1;
    double distance = 1e9;
};

std::vector<GroundTruthBall> loadGroundTruth(const std::string& csvPath, const std::string& imageName)
{
    std::vector<GroundTruthBall> groundTruth;
    std::ifstream file(csvPath);

    if (!file.is_open())
    {
        std::cerr << "Could not open ground-truth file: " << csvPath << std::endl;
        return groundTruth;
    }

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream lineStream(line);
        std::string image;
        std::string labelText;
        std::string xText;
        std::string yText;

        std::getline(lineStream, image, ',');
        std::getline(lineStream, labelText, ',');
        std::getline(lineStream, xText, ',');
        std::getline(lineStream, yText, ',');

        if (image != imageName)
        {
            continue;
        }

        try
        {
            const int label = (labelText == "White") ? 0 : std::stoi(labelText);
            groundTruth.push_back({cv::Point2f(std::stof(xText), std::stof(yText)), label});
        }
        catch (const std::exception&)
        {
            std::cerr << "Skipping invalid ground-truth row: " << line << std::endl;
        }
    }

    return groundTruth;
}

cv::Mat calculateBallHistogram(const cv::Mat& hsvImage, const cv::Point2f& center, float radius)
{
    const int left = std::max(0, cvRound(center.x - radius));
    const int top = std::max(0, cvRound(center.y - radius));
    const int right = std::min(hsvImage.cols - 1, cvRound(center.x + radius));
    const int bottom = std::min(hsvImage.rows - 1, cvRound(center.y + radius));

    const cv::Rect roiRect(left, top, right - left + 1, bottom - top + 1);
    const cv::Mat roi = hsvImage(roiRect);

    cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8U);
    cv::circle(
        mask,
        cv::Point(cvRound(center.x) - left, cvRound(center.y) - top),
        cvRound(radius * 0.9F),
        cv::Scalar(255),
        cv::FILLED
    );

    int histogramSize[] = {30, 32};
    float hueRange[] = {0, 180};
    float saturationRange[] = {0, 256};
    const float* histogramRanges[] = {hueRange, saturationRange};
    int histogramChannels[] = {0, 1};

    cv::Mat histogram;
    cv::calcHist(
        &roi,
        1,
        histogramChannels,
        mask,
        histogram,
        2,
        histogramSize,
        histogramRanges,
        true,
        false
    );
    cv::normalize(histogram, histogram, 0, 255, cv::NORM_MINMAX);
    return histogram;
}

std::vector<ReferenceBall> buildReferenceBalls(const std::string& referenceImagePath)
{
    std::vector<ReferenceBall> references;
    const cv::Mat referenceImage = cv::imread(referenceImagePath);

    if (referenceImage.empty())
    {
        std::cerr << "Could not load reference image: " << referenceImagePath << std::endl;
        return references;
    }

    cv::Mat grayImage;
    cv::cvtColor(referenceImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 1.5);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(
        blurredImage,
        circles,
        cv::HOUGH_GRADIENT,
        1.0,
        40.0,
        80.0,
        14.0,
        20,
        35
    );

    std::vector<cv::Vec3f> referenceCircles;
    for (const cv::Vec3f& circle : circles)
    {
        if (circle[0] > 1000.0F && circle[1] > 110.0F && circle[1] < 230.0F)
        {
            referenceCircles.push_back(circle);
        }
    }

    if (referenceCircles.size() != 16)
    {
        std::cerr << "Warning: expected 16 reference balls, found "
                  << referenceCircles.size() << std::endl;
    }

    size_t cueIndex = 0;
    for (size_t i = 1; i < referenceCircles.size(); ++i)
    {
        if (referenceCircles[i][0] < referenceCircles[cueIndex][0])
        {
            cueIndex = i;
        }
    }

    std::vector<cv::Vec3f> topRow;
    std::vector<cv::Vec3f> bottomRow;
    for (size_t i = 0; i < referenceCircles.size(); ++i)
    {
        if (i == cueIndex)
        {
            continue;
        }

        if (referenceCircles[i][1] < 170.0F)
        {
            topRow.push_back(referenceCircles[i]);
        }
        else
        {
            bottomRow.push_back(referenceCircles[i]);
        }
    }

    const auto sortByX = [](const cv::Vec3f& first, const cv::Vec3f& second)
    {
        return first[0] < second[0];
    };
    std::sort(topRow.begin(), topRow.end(), sortByX);
    std::sort(bottomRow.begin(), bottomRow.end(), sortByX);

    cv::Mat hsvImage;
    cv::cvtColor(referenceImage, hsvImage, cv::COLOR_BGR2HSV);

    const auto addReference = [&](int label, const cv::Vec3f& circle)
    {
        const cv::Mat histogram = calculateBallHistogram(
            hsvImage,
            cv::Point2f(circle[0], circle[1]),
            circle[2]
        );
        references.push_back({label, histogram});
    };

    addReference(0, referenceCircles[cueIndex]);

    for (size_t i = 0; i < topRow.size(); ++i)
    {
        addReference(static_cast<int>(i) + 1, topRow[i]);
    }

    for (size_t i = 0; i < bottomRow.size(); ++i)
    {
        addReference(static_cast<int>(i) + 9, bottomRow[i]);
    }

    return references;
}

ClassificationResult classifyBall(
    const cv::Mat& hsvImage,
    const cv::Point2f& center,
    float radius,
    const std::vector<ReferenceBall>& references)
{
    ClassificationResult result;
    const cv::Mat testHistogram = calculateBallHistogram(hsvImage, center, radius);

    for (const ReferenceBall& reference : references)
    {
        const double distance = cv::compareHist(
            testHistogram,
            reference.histogram,
            cv::HISTCMP_BHATTACHARYYA
        );

        if (distance < result.distance)
        {
            result.distance = distance;
            result.label = reference.label;
        }
    }

    return result;
}

DetectionMetrics evaluateDetections(
    const std::vector<cv::Vec3f>& detections,
    const std::vector<int>& predictedLabels,
    const std::vector<GroundTruthBall>& groundTruth,
    float tolerance)
{
    DetectionMetrics metrics;
    std::vector<bool> matched(groundTruth.size(), false);

    for (size_t detectionIndex = 0; detectionIndex < detections.size(); ++detectionIndex)
    {
        const cv::Vec3f& detection = detections[detectionIndex];
        const cv::Point2f center(detection[0], detection[1]);
        int bestIndex = -1;
        float bestDistance = 0.0F;

        for (size_t i = 0; i < groundTruth.size(); ++i)
        {
            if (matched[i])
            {
                continue;
            }

            const float distance = static_cast<float>(cv::norm(center - groundTruth[i].center));
            if (bestIndex < 0 || distance < bestDistance)
            {
                bestIndex = static_cast<int>(i);
                bestDistance = distance;
            }
        }

        if (bestIndex >= 0 && bestDistance <= tolerance)
        {
            matched[bestIndex] = true;
            ++metrics.truePositives;

            if (predictedLabels[detectionIndex] == groundTruth[bestIndex].label)
            {
                ++metrics.correctLabels;
            }
        }
    }

    metrics.falsePositives = static_cast<int>(detections.size()) - metrics.truePositives;
    metrics.falseNegatives = static_cast<int>(groundTruth.size()) - metrics.truePositives;

    metrics.precision = detections.empty()
        ? 0.0
        : static_cast<double>(metrics.truePositives) / detections.size();
    metrics.recall = groundTruth.empty()
        ? 0.0
        : static_cast<double>(metrics.truePositives) / groundTruth.size();
    metrics.f1 = (metrics.precision + metrics.recall) > 0.0
        ? 2.0 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        : 0.0;
    metrics.recognitionAccuracy = (metrics.truePositives > 0)
        ? static_cast<double>(metrics.correctLabels) / metrics.truePositives
        : 0.0;

    return metrics;
}

void printDetectionMetrics(const std::string& imageName, const DetectionMetrics& metrics)
{
    std::cout << imageName << std::endl;
    std::cout << "  True positives: " << metrics.truePositives << std::endl;
    std::cout << "  False positives: " << metrics.falsePositives << std::endl;
    std::cout << "  False negatives: " << metrics.falseNegatives << std::endl;
    std::cout << "  Precision: " << metrics.precision << std::endl;
    std::cout << "  Recall: " << metrics.recall << std::endl;
    std::cout << "  F1-score: " << metrics.f1 << std::endl;
    std::cout << "  Recognition accuracy: " << metrics.recognitionAccuracy << std::endl;
}
}

void MyApplication()
{
    const std::string groundTruthPath = "data/ground_truth/ball_locations.csv";
    const std::string referenceImagePath = "data/images/AllBalls.jpeg";
    const std::vector<std::string> imageNames = {
        "Table1.jpeg",
        "Table2.jpeg",
        "Table3.jpeg",
        "Table4.jpeg",
        "Table5.jpeg",
        "Table6.jpeg"
    };

    const std::vector<ReferenceBall> referenceBalls = buildReferenceBalls(referenceImagePath);
    if (referenceBalls.empty())
    {
        std::cerr << "Reference-ball histograms could not be built." << std::endl;
        return;
    }

    // Table corners in the original image, ordered top-left, top-right,
    // bottom-right, bottom-left. Coordinates are (column, row).
    const vector<cv::Point2f> tableCorners = {
        cv::Point2f(337, 35),
        cv::Point2f(1570, 21),
        cv::Point2f(1888, 451),
        cv::Point2f(38, 472)
    };
    // The 1434 x 683 frame matches the physical cushion-to-cushion size in millimetres.
    const vector<cv::Point2f> destCorners = {
        cv::Point2f(0, 0),
        cv::Point2f(1434, 0),
        cv::Point2f(1434, 683),
        cv::Point2f(0, 683)
    };

    const cv::Mat perspectiveTransform = cv::getPerspectiveTransform(tableCorners, destCorners);

    const std::vector<cv::Point2f> pocketCenters = {
        cv::Point2f(0.0F, 0.0F),
        cv::Point2f(717.0F, 0.0F),
        cv::Point2f(1434.0F, 0.0F),
        cv::Point2f(0.0F, 683.0F),
        cv::Point2f(717.0F, 683.0F),
        cv::Point2f(1434.0F, 683.0F)
    };

    std::cout << std::fixed << std::setprecision(3);

    double precisionSum = 0.0;
    double recallSum = 0.0;
    double f1Sum = 0.0;
    int totalTruePositives = 0;
    int totalFalsePositives = 0;
    int totalFalseNegatives = 0;
    int totalCorrectLabels = 0;
    int imageCount = 0;

    for (const std::string& imageName : imageNames)
    {
        const std::string inputImagePath = "data/images/" + imageName;
        cv::Mat inputImage = cv::imread(inputImagePath);
        if (inputImage.empty())
        {
            std::cout << "Error loading image: " << inputImagePath << std::endl;
            continue;
        }

        cv::Mat warpedImage;
        cv::warpPerspective(inputImage, warpedImage, perspectiveTransform, cv::Size(1434, 683));
        cv::imwrite("results/warped_" + imageName, warpedImage);

        const std::string imageBase = imageName.substr(0, imageName.find_last_of('.'));

        cv::Mat grayImage;
        cv::cvtColor(warpedImage, grayImage, cv::COLOR_BGR2GRAY);

        cv::Mat cannyEdges;
        cv::Canny(grayImage, cannyEdges, 100, 200);

        cv::Mat hsvImage;
        cv::cvtColor(warpedImage, hsvImage, cv::COLOR_BGR2HSV);

        const cv::Rect feltRoiRect(200, 200, 100, 100);
        const cv::Mat feltRoi = hsvImage(feltRoiRect);

        int histogramSize[] = {30, 32};
        float hueRange[] = {0, 180};
        float saturationRange[] = {0, 256};
        const float* histogramRanges[] = {hueRange, saturationRange};
        int histogramChannels[] = {0, 1};

        cv::Mat feltHistogram;
        cv::calcHist(
            &feltRoi,
            1,
            histogramChannels,
            cv::Mat(),
            feltHistogram,
            2,
            histogramSize,
            histogramRanges,
            true,
            false
        );
        cv::normalize(feltHistogram, feltHistogram, 0, 255, cv::NORM_MINMAX);

        cv::Mat backProjection;
        cv::calcBackProject(
            &hsvImage,
            1,
            histogramChannels,
            feltHistogram,
            backProjection,
            histogramRanges,
            1.0,
            true
        );

        cv::Mat feltMask;
        cv::threshold(backProjection, feltMask, 50, 255, cv::THRESH_BINARY);

        cv::Mat tableMask;
        cv::bitwise_not(feltMask, tableMask);

        const cv::Mat morphologyKernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25));
        cv::morphologyEx(tableMask, tableMask, cv::MORPH_CLOSE, morphologyKernel);
        cv::morphologyEx(tableMask, tableMask, cv::MORPH_OPEN, morphologyKernel);
        cv::bitwise_not(tableMask, feltMask);

        cv::Mat maskedGray;
        cv::bitwise_and(grayImage, tableMask, maskedGray);
        const double meanFeltValue = cv::mean(grayImage(feltRoiRect))[0];
        maskedGray.setTo(cv::Scalar(meanFeltValue), feltMask);

        cv::Mat blurredImage;
        cv::GaussianBlur(maskedGray, blurredImage, cv::Size(5, 5), 1.5);

        std::vector<cv::Vec3f> houghCircles;
        cv::HoughCircles(
            blurredImage,
            houghCircles,
            cv::HOUGH_GRADIENT,
            1.0,
            38.0,
            80.0,
            9.0,
            19,
            24
        );

        std::vector<cv::Vec3f> filteredCircles;
        filteredCircles.reserve(houghCircles.size());

        for (const cv::Vec3f& circle : houghCircles)
        {
            const cv::Point2f center(circle[0], circle[1]);
            if (center.x < 22 || center.x > 1434 - 22 ||
                center.y < 22 || center.y > 683 - 22)
            {
                continue;
            }

            bool nearPocket = false;
            for (const cv::Point2f& pocket : pocketCenters)
            {
                if (cv::norm(center - pocket) < 95.0)
                {
                    nearPocket = true;
                    break;
                }
            }

            if (nearPocket)
            {
                continue;
            }

            filteredCircles.push_back(circle);
        }

        std::vector<cv::Vec3f> recognizedBalls;
        std::vector<int> recognizedLabels;
        recognizedBalls.reserve(filteredCircles.size());
        recognizedLabels.reserve(filteredCircles.size());

        for (const cv::Vec3f& circle : filteredCircles)
        {
            const cv::Point2f center(circle[0], circle[1]);
            const ClassificationResult classification =
                classifyBall(hsvImage, center, circle[2], referenceBalls);

            if (classification.label < 0 || classification.distance > 0.65)
            {
                continue;
            }

            recognizedBalls.push_back(circle);
            recognizedLabels.push_back(classification.label);
        }

        cv::Mat detectionImage = warpedImage.clone();
        for (size_t i = 0; i < recognizedBalls.size(); ++i)
        {
            const cv::Point centerPoint(
                static_cast<int>(std::lround(recognizedBalls[i][0])),
                static_cast<int>(std::lround(recognizedBalls[i][1]))
            );
            const int radius = static_cast<int>(std::lround(recognizedBalls[i][2]));
            const std::string labelText = (recognizedLabels[i] == 0)
                ? "Cue"
                : std::to_string(recognizedLabels[i]);

            cv::circle(detectionImage, centerPoint, radius, cv::Scalar(0, 0, 255), 2);
            cv::circle(detectionImage, centerPoint, 2, cv::Scalar(0, 255, 0), -1);
            cv::putText(
                detectionImage,
                labelText,
                centerPoint + cv::Point(-8, 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA
            );
        }

        std::cout << "Detected " << filteredCircles.size() << " candidates in " << imageName
                  << "; " << recognizedBalls.size() << " passed colour verification." << std::endl;

        const std::vector<GroundTruthBall> groundTruth = loadGroundTruth(groundTruthPath, imageName);
        const DetectionMetrics metrics =
            evaluateDetections(recognizedBalls, recognizedLabels, groundTruth, 21.5F);
        printDetectionMetrics(imageName, metrics);

        precisionSum += metrics.precision;
        recallSum += metrics.recall;
        f1Sum += metrics.f1;
        totalTruePositives += metrics.truePositives;
        totalFalsePositives += metrics.falsePositives;
        totalFalseNegatives += metrics.falseNegatives;
        totalCorrectLabels += metrics.correctLabels;
        ++imageCount;

        cv::imwrite("results/" + imageBase + "_mask.jpeg", feltMask);
        cv::imwrite("results/" + imageBase + "_edges.jpeg", blurredImage);
        cv::imwrite("results/" + imageBase + "_detections.jpeg", detectionImage);
    }

    const int divisor = (imageCount > 0) ? imageCount : 1;
    const double recognitionAccuracy = (totalTruePositives > 0)
        ? static_cast<double>(totalCorrectLabels) / totalTruePositives
        : 0.0;

    std::cout << "Overall average metrics across " << imageCount << " images:" << std::endl;
    std::cout << "  Average precision: " << precisionSum / divisor << std::endl;
    std::cout << "  Average recall: " << recallSum / divisor << std::endl;
    std::cout << "  Average F1-score: " << f1Sum / divisor << std::endl;
    std::cout << "  Total true positives: " << totalTruePositives << std::endl;
    std::cout << "  Total false positives: " << totalFalsePositives << std::endl;
    std::cout << "  Total false negatives: " << totalFalseNegatives << std::endl;
    std::cout << "  Stage 2 recognition accuracy: " << recognitionAccuracy << std::endl;
}

#ifndef MYAPPLICATION_NO_MAIN
int main()
{
    MyApplication();
    return 0;
}
#endif
