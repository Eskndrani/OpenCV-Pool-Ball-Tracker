# Pool Table Ball Detection and Recognition

Individual C++/OpenCV implementation of the pool-ball vision assignment. All implementation code lives in `myApplication.cpp`, and the callable entry point is `MyApplication()`.

Repository: <https://github.com/Eskndrani/OpenCV-Pool-Ball-Tracker>

## Project status

The assignment pipeline is implemented end to end:

- **Stage 1 (detection):** precision **0.911**, recall **0.911**, F1 **0.911**.
- **Stage 2 (recognition):** **76.8%** accuracy on correctly localised balls (`63 / 82`).
- **Report:** `report/Report.pdf`, compiled from `report/Report.tex` at exactly eight pages.

## Pipeline overview

1. **Rectification.** Each table image is warped through the supplied corner points into the required `1434 x 683` table coordinate frame.
2. **Edge fusion.** Multi-channel CIELAB Canny maps are combined: the lightness channel captures the outer silhouette, while the `a` and `b` channels preserve colour boundaries under shadows and reflections.
3. **Dynamic pocket masking.** Dark pocket holes, bright metallic brackets, and edge-dense netting are masked locally around the six pockets instead of applying a blind radius.
4. **Hough detection and fusion.** Circular Hough detection is followed by a 40-pixel physical NMS and a 22-pixel cushion margin. A raw-frame secondary Hough pass is projected through the homography and fused only when it is more than 30 pixels from every rectified detection.
5. **Recognition.** One-dimensional hue (30 bins) and saturation (32 bins) histograms are sampled from a 16-pixel-radius mask and compared with the sixteen `AllBalls.jpeg` references using a 30/70 hue-saturation Bhattacharyya distance.
6. **Cue gating.** A split-mask cue gate checks centre brightness/whiteness and full-ball saturation, then applies the Highlander rule (lowest full saturation) to select the cue ball. A weighted distance above `0.55` rejects unreliable observations.

## Repository layout

```text
.
├── CMakeLists.txt
├── myApplication.cpp          # full pipeline and evaluation
├── AI_USAGE.md
├── data/
│   ├── images/                # AllBalls.jpeg and Table1-6.jpeg
│   └── ground_truth/          # ball_locations.csv
├── results/                   # generated overlays, edge maps, and figures
├── report/
│   ├── Report.tex
│   └── Report.pdf             # final eight-page report
└── README.md
```

## Requirements

- CMake 3.20 or newer
- A C++17 compiler
- OpenCV **4.12.0** with `core`, `imgproc`, `imgcodecs`, and `highgui`

The build was verified on Windows with MSVC/NMake and OpenCV installed through vcpkg. `CMakeLists.txt` searches:

- `OpenCV_DIR` passed on the command line or set in the environment,
- `../OpenCVExample/OpenCVConfig.cmake`,
- `C:/vcpkg/installed/x64-windows/share/opencv4`.

Do not mix MSVC-built OpenCV binaries with a MinGW compiler; the ABI must match.

## Build

```powershell
cmake -S . -B build -DOpenCV_DIR="C:/path/to/opencv/build"
cmake --build build --config Release
```

If CMake already finds OpenCV, omit `-DOpenCV_DIR`.

## Run

Run from the repository root because the program uses relative paths:

```powershell
.\build\pool_ball_vision.exe
```

With a multi-config Visual Studio generator the executable is usually under `build\Release\`.

The program reads:

- `data/images/AllBalls.jpeg`
- `data/images/Table1.jpeg` through `Table6.jpeg`
- `data/ground_truth/ball_locations.csv`

It prints per-image and overall metrics to the console and writes diagnostic overlays and edge maps into `results/`.

## Evaluation

Stage 1 matches detections to ground-truth centres one-to-one within a `21.5 px` radius, which is half the `43 px` ball diameter. Precision, recall, and F1 are reported per table and overall. Stage 2 recognition accuracy uses only Stage 1 true positives as its denominator.

The main remaining failure modes are:

- duplicate Hough responses on a single perimeter ball,
- missed interior balls where touching balls merge Canny arcs in a dense cluster,
- rotation and specular-reflection effects that shift 1D hue/saturation histogram bins toward the wrong reference or the `0.55` rejection threshold.

The report contains the full calculation definitions, parameter discussion, visual evidence, and error analysis.

## AI assistance

See `AI_USAGE.md` and the AI Usage Statement in `report/Report.tex` for the disclosure of tools used and the work completed directly by the author.

## Dataset note

The assignment images are tracked in this repository. Follow the course rules on repository visibility and do not redistribute the dataset or solution outside the permitted submission channels.
