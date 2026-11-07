// Pool-table ball detection and recognition assignment.
// All assignment implementation must remain in this file, with MyApplication()
// exposed as the entry point expected by the assessment harness.

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

void MyApplication()
{
    // TODO: Stage 1 - Load the input image and apply the perspective transform
    // into the assignment's 1434 x 683 table coordinate frame.

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
