# AI and resource acknowledgement

This file records the AI assistance used to create the starter repository. Update it if additional AI assistance or external resources are used, and follow the exact acknowledgement format required by the module.

## AI use in this scaffold

- ChatGPT/Codex was used to create the initial folder structure, this README, `.gitignore`, `CMakeLists.txt`, and the minimal `myApplication.cpp` entry-point stub.
- ChatGPT/Codex provided environment-check findings and general project-planning suggestions in the README.
- No ball detection, coordinate transformation, ball recognition, training, metric computation, experiment result, or report text was implemented or generated for this initial scaffold.
- The only source data supplied by the student at this point is `Ball Locations In Transformed Image(1).xlsx`; the assignment description and screenshots were used to derive the setup notes.

## Perspective transform assistance

- **Date:** 7 November 2026
- **Tool:** opencode AI assistant
- **Purpose:** Provided the C++ code structure for the perspective-transform stage and helped correct the source-corner correspondences.
- **Retained in code:** `myApplication.cpp` uses `cv::getPerspectiveTransform` to compute the planar homography from the four supplied source coordinates to the four destination coordinates of the 1434 x 683 rectified table frame, and `cv::warpPerspective` to produce and save `results/warped_Table1.jpeg`.
- **Student review:** The student remains responsible for checking the corner measurements, the resulting rectified image, and all later detection and recognition work.

Record any later AI use here with the tool, date, purpose, and exactly what was retained in code or the report. Also acknowledge non-AI resources such as course material, documentation, libraries, and example code as required by the assignment.

