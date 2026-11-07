# Pool Table Ball Vision Assignment

Starter repository for the individual C++/OpenCV pool-ball assignment. The assignment requires all implementation code to be in `myApplication.cpp` and the entry point to be callable as `MyApplication()`.

## Current contents

- `myApplication.cpp` — minimal entry-point stub; no detection, recognition, or evaluation solution is included.
- `CMakeLists.txt` — C++17 build setup using OpenCV components for image processing and display.
- `data/ground_truth/` — place a copy of the supplied spreadsheet here if desired.
- `data/images/` — the supplied assignment images are stored here and are no longer excluded by Git.
- `results/` — reserve for automatically generated evaluation outputs and selected visual examples.
- `report/` — reserve for the final report and working figures.
- `AI_USAGE.md` — disclosure of the AI assistance used to create this starter scaffold.

The supplied workbook is currently in the repository root. The image files are included in Git's normal file tracking; consider repository visibility carefully and follow the course rule not to make your work accessible to others.

## Finding the course images

The assignment sheet says its six table frames (`Table1.jpeg` to `Table6.jpeg`) and `AllBalls.jpeg` are included in the TIPS package. The introductory-lab page you found links to the full TIPS code-samples ZIP; its page says that archive includes the sample application images and videos. This is the first archive to check: download it, extract it, and search the extracted folder for the exact filenames. The [TIPS code-samples page](https://publications.scss.tcd.ie/book-supplements/A-Practical-Introduction-to-Computer-Vision-with-OpenCV/Code/) links the Windows Visual Studio 2022 / OpenCV 4.12.0 package.

There is also a separate textbook [Playing Pool application-problem ZIP](https://publications.scss.tcd.ie/book-supplements/A-Practical-Introduction-to-Computer-Vision-with-OpenCV/Problems/), described on that page as one 38-second pool video. It may be a different teaching example; the public page does not identify it as the six still images or `AllBalls.jpeg` from this assignment. If those names are absent from the extracted TIPS archive, check the assignment's Blackboard content / TIPS package location or ask the module lecturer where the assignment dataset is distributed. Do not substitute the public pool video for the required evaluation images.

Once found, keep the six evaluation frames and the recognition reference image in separate folders so it is easy to enforce the assignment's training restriction. The `.gitignore` excludes `data/images/` to avoid accidentally publishing the dataset.

## Software check on this computer

Checked on 1 November 2026:

- CMake, MinGW `g++`, Git, and Python 3.12 are available.
- Python `cv2` 4.13.0 is available, but that does not by itself provide the C++ OpenCV headers and link libraries needed by this project.
- `pkg-config opencv4` did not find a C++ OpenCV development installation, and common header locations were not found.
- Python `openpyxl` 3.1.5 is available for inspecting the supplied workbook if useful.

The C++ build is therefore scaffolded but is expected to stop at OpenCV discovery until its C++ development package is installed and visible to CMake. On Windows, install an OpenCV distribution that matches the chosen compiler, or install the matching MSYS2 UCRT64 OpenCV package. Then configure and build from this folder:

```powershell
cmake -S . -B build -DOpenCV_DIR="C:/path/to/opencv/build"
cmake --build build --config Release
```

If the OpenCV package is discoverable without `OpenCV_DIR`, omit that option. Do not mix MSVC-built OpenCV binaries with MinGW `g++`; compiler ABI compatibility matters. The supplied code is only an entry-point stub and may need to be linked with the assignment harness that calls `MyApplication()`.

## Suggested work plan

1. **Understand and preserve the data.** Keep the six table images and `AllBalls.jpeg` as separate inputs. Use `AllBalls.jpeg` only for recognition training/examples as the assignment allows. Keep a clear record of the workbook's coordinate convention (row, column) and missing/blank labels.
2. **Build a reproducible evaluation path early.** Create a fixed list of six test images and load the provided locations as ground truth. Use the same parameters on all six images; do not tune on test images. Record all settings and save machine-readable per-image results.
3. **Stage 1: ball locations.** The course clarification says the table corner positions are provided, so use the supplied four image-space corners rather than trying to detect them. Rectify the table into its stated 1434-column by 683-row coordinate frame. Restrict candidate detection to the playing surface, estimate ball centres, and map centres into table coordinates. Decide and document a matching tolerance before evaluating.
4. **Choose and explain methods.** The course clarification says to mainly use techniques taught in the module. For any additional technique, explain how it works and why it was needed; a suggestion from generative AI is not a justification. Keep all assignment code in `myApplication.cpp`, with `MyApplication()` as the callable entry point.
5. **Measure detection.** Match predictions to ground-truth centres one-to-one within the chosen tolerance. Report precision, recall, and F1, plus localization error (for matched detections, in table pixels or millimetres). Explain how false positives, missed balls, and unmatched locations are counted. Include results per image and overall.
6. **Stage 2: recognition.** Build examples only from `AllBalls.jpeg`; define how each candidate crop is normalized and how labels 1–15 and cue ball are assigned. Keep detection and recognition scores separate, then report end-to-end recognition accuracy for correctly localized balls and explain the denominator.
7. **Inspect failures.** Save examples of intermediate stages and overlays showing ground truth, predictions, and matches. Examine overlaps, shadows, reflections, pockets/rails, numbered markings, and partly occluded balls. Use these examples to support the error analysis.
8. **Write the report.** Keep it within eight pages, with roughly equal space for theory and results. Describe processing theory and parameter choices without referring to code or library calls. Include an illustration of each processing stage, full metrics and calculation definitions, successes and failures, robustness limits, and specific improvements.
9. **Re-run everything from a clean build.** Confirm the same fixed settings generate all reported metrics and figures. Include an accurate acknowledgement of all resources, including this AI-generated project scaffold and any further assistance you use.

## Coordinate and metric reminders

The assignment defines table coordinates as a 1434 by 683 transformed image for a 1434 mm by 683 mm table, with ball diameter 43 mm. This implies approximately one transformed pixel per millimetre, so an unobstructed ball should span about 43 transformed pixels in diameter. Confirm whether workbook entries refer to pixel centres and whether rows/columns are zero-based before computing errors. Justify a matching tolerance in physical or pixel terms rather than choosing it after seeing the scores.

## Suggested repository layout

```text
.
├── CMakeLists.txt
├── myApplication.cpp
├── AI_USAGE.md
├── data/
│   ├── ground_truth/
│   └── images/
├── results/
│   ├── metrics/
│   └── figures/
└── report/
```

