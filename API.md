# API

## Files
| File Name      | Inform                                   |
| -------------- | :--------------------------------------- |
| car_target.py  | Armour Detection                         |
| circle_test.py | A testing program for circle detection   |
| tool.py        | Supporting file for armour detection program |
| communicate.py | Module for communication  with car       |

## Tutorial

### For Users:

#### Parameter Settings

**MODE:** Running mode, `D` for debug mode, `R` for running mode.

**SRC:** Image source, `PIC` for reading from picture, `CAM` for camera, `VID` for video

**TARGET:** Color of target, `BLUE` for detecting blue light bar, `RED` for detecting red light bar

### For Developers:

#### Fixing Log

1. Return target coordinate from `main(mat)` and write to microcomputer from serial port
2. Resizing the output image to a formal value
3. Match target from camera coordinate to gun coordinate
4. Convert (x,y) to (pitch, yaw)

#### Improvements:

1. Coordinate conversion
2. Targeting Mode
3. Robustness: in current algorithm, final target selection is based on pixel numbers. Whichever methods we used for tracking, we cannot keep tracking a car in real competition. The gun will keep switching between different cars because of the disturbance. This problem must be fixed!
4. Computation Speed: try to eliminate for loop and use matrix operation instead)


## Complete Documentation for Car Target Detection

### API for functions in the program

#### func_undistort

Recover the undistorted image obtained from the camera.

##### parameters:

- `mat` - image matrix

#### track

1. color threshold
2. segment image by connected component and find the bar
3. match the bar and get the bar pairs with similar angle and length
4. select the bar pairs

##### parameters:

- `mat` - image matrix
- `color_threshold` - the value of threshold
- `square_ratio` - the threshold ratio of height/width (any smaler than the threshold will be deleted)
- `angle_rate` - angle factor for matching
- `length_rate` - length factor for matching
- `matrix threshold` - match those with score greater than threshold
- `DIST` - distance of the target

##### return:

`[]` when no target, `[x, y, pixel_count]` for target

#### func_get_delete_strict_list

Get a strict list of deleted component. Used when target is near

##### parameters:

- `connect_data` - the information list of connected component
- `square_ratio` - standard ratio for selecting the square

#### func_get_delete_loose_list

Get a strict list of deleted component. Used when target is far

##### parameters:

- `connect_data` - the information list of connected component
- `square_ratio` - standard ratio for selecting the square

#### main

Main function of detection.

##### parameters:
- `mat` - image matrix

#### cv2.connectedComponentsWithStats

connect_output is a tuple :

- int component count,
- ndarray label map : use int to label connect  component, same int value means one component, size equal to "b"
- ndarray connect component info : a n * 5 ndarray to show connect component info, [left_top_x, left_top_y, width, height, pixel number])
- ndarray unused not clear


### Detail description on Armour Detection

#### Setup

Set up undistorted matrix and some parameters

#### Program

##### 1. Setup shooting range

##### 2. Judge image source and run corresponding program

###### 2.1 Image source is picture

Give a file list

Read pictures in the list one by one.

Store a read picture in `mat`, and run `main(mat)`

Quit the program and close all windows when "q" was pressed

###### 2.2 Image source is camera

Capture a picture with camera

If camera is open, read the captured picture, stored it in `mat`, run `func_undistort(mat)`` to undistort the captured picture, and run `main(mat)`

Stop capturing, and close all windows

###### 2.3 Image source is video

Run a video capturer to capture pictures from a video

While capturer is running, read the image captured, store it in `mat` and run `main(mat)`

Wait for some seconds (the exact number of seconds is decided by variable `wait`)

When "q" was pressed, break; when "p" was pressed, set `wait` to 0; when "c" was pressed, set `wait` to 10.

Stop capturing and close all windows.

#### 3. Main

1. Get global variable: target_last and MIN_STEP. Copy `mat` (input image matrix) to `origin`, set `tstart` to starting time.
2. Track the near by targets: `track`(detail information of this function is depicted below)
3. If no target is found, track the far targets.
4. Calculate the distance of the target
5. Draw a circle on the detected targets.

#### 4. Track

1. Relative threshold: set 255 if b - r < color_threshold. Show the processed image if it's in debug mode.
2. Label connected component: Use `cv2.connectedComponentsWithStats` (whose detail information is given in function part) to label. Draw rectangles on the selected connected components if in Debug mode.
3. Get delete list according to the ratio of length and height, and area of connected components ( use `func_get_delete_strict_list` in `NEAR` mode and with `func_get_delete_loose_list` in `MID` mode to get the list of connected components to be deleted ). If not connected components are remained after deletion, return.
4. Get y coordinate and x coordinate of mid point for both top line and bottom line of the light bar. Draw a circle on both points when in debug mode.
5. Calculate bar length, bar angle and get weighted sum.
6. Calculate bar angle difference and bar length difference  matrix
7. Use a weighted sum combining angle difference and length difference, and then threshold the sum to get a primitive match
8. Calculate x difference, y difference and z difference of each pair of light bars. Further distinguish difference between pairs by thresholding cross ratio and angles of crossing line between paris.
9. Times up all three match matrix to get the final match result. (All three matrices are 0-1 matrix and square matrix, it's just like combining 3 relation and keep only those available in all three matrices as real available pairs).
10. Calculate the sum of pixels of matching pairs, pick up pairs with most total pixel to be the target. Calculate and return its x and y coordinate.
