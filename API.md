# API

## Files
|File Name| Inform|
|------------|:----|
|car_target.py| Armour Detection |
|circle_test.py| A testing program for circle detection|
|tool.py| Supporting file for armour detection program |

## Complete Documentation for Car Target Detection

### API for functions in the program
#### func_undistort
Recover the undistorted image obtained from the camera. </br>

<b>paramter :</b>
* <b>mat</b> - image matrix

#### track
1. color threshold
2. segment image by connected component and find the bar
3. match the bar and get the bar pairs with similar angle and length
4. select the bar pairs</br>

<b>paramter :</b>
* <b>mat</b> - image matrix
* <b>color_threshold</b> - the value of threshold
* <b>square_ratio</b> - the threshold ratio of height/width (any smaler than the threshold will be deleted)
* <b>angle_rate</b> - angle factor for matching
* <b>length_rate</b> - length factor for matching
* <b>matrix threshold</b> - match those with score greater than threshold
* <b>DIST</b> - distance of the target

<b> return :</b>
* return: [] when no target, [x, y, pixel count] for target

#### func_get_delete_strict_list
Get a strict list of deleted component. Used when target is near

<b>paramter :</b>
* <b>connect_data</b> - the information list of connected component
* <b>square_ratio</b> - standard ratio for selecting the square

#### func_get_delete_loose_list

Get a strict list of deleted component. Used when target is far

<b>paramter :</b>
* <b>connect_data</b> - the information list of connected component
* <b>square_ratio</b> - standard ratio for selecting the square

#### main
Main function of detection.

<b>paramter :</b>
* <b>mat</b> - image matrix

#### cv2.connectedComponentsWithStats

connect_output is a tuple :
- int component count,
- ndarray label map : use int to label connect  component, same int value means one component, size equal to "b"
- ndarray connect component info : a n * 5 ndarray to show connect component info, [left_top_x, left_top_y, width, height, pixel number])
- ndarray unused not clear


### Detail description on Armour Detection

#### Setup
Set up undisorted matrix and some paramters

#### Program

##### 1. Setup shooting range

##### 2. Judge image souce and run corresponding program

###### 2.1 Image source is picture<br>

Give a file list<br>
Read pictures in the list one by one.<br>
Store a read picture in <b>mat</b>, and run <b>main(mat)</b> <br>
Quit the program and close all windows when press "q"<br>
###### 2.2 Image source is camera

Capture a picture with camera<br>
If camera is open, read the captured picture, stored it in <b>mat</b>, run <b>func_undistort(mat)</b> to undisort the captured picture, and run <b>main(mat)</b><br>
Stop capturing, and close all windows

###### 2.3 Image source is video

Run a video capturer to capture pictures from a video<br>
While capturer is running, read the image captured, store it in <b>mat</b> and run <b>main(mat)</b><br>
Wait for some seconds (the exact number of seconds is decided by variable <b>wait</b>)<br>
When press "q", break; when press "p", set <b>wait</b> to 0; when press "c",set <b>wait</b> to 10.<br>
Stop capturing and close all windows.

#### 3. Main
1. Get global variable : target_last and MIN_STEP. Copy <b>mat</b> (input image matrix) to <b>origin</b>, set <b>tstart</b> to starting time.
2. Track the near by targets :<b>track</b>(detail information of this function is dipicted below)
3. If no target is found, track the far targets.
4. Calculate the distance of the target
5. Draw a circle on the detected targets.

#### 4.Track
1. Relative threshold: set 255 if b - r < color_threshold. Show the processed image if it's in debug mode.<br>
2. Label connected component: Use <b>cv2.connectedComponentsWithStats</b> (whose detail information is given in function part) to label. Draw rectangles on the selected connected components if in Debug mode.
3. Get delete list according to the ratio of length and height, and area of connected components ( use <b>func_get_delete_strict_list</b> in <b>NEAR</b> mode and with <b>func_get_delete_loose_list</b> in <b>MID</b> mode to get the list of connecetd components to be deleted ). If not connected components are remained after deletion, reutun.
4. Get y coordinate and x coordinate of mid point for both top line and bottom line of the light bar. Draw a cirlce on both points when in debug mode.
5. Calculate bar length, bar angle and get weighted sum.
6. Calculate bar angle dfference and bar length difference  matrix
7. Use a weighted sum combining angle difference and length difference, and then threshold the sum to get a primitive match
8. Calculate x difference, y difference and z difference of each pair of light bars, use the threshold to further distinguish difference in cross ratio and angles.
9. Times up all three match matrix to get the final match result. (All three matricies are 0-1 matrix and square matrix, it's just like combining 3 relation and keep only those available in all thre matrix as real available pairs).
9. Calculate the sum of pixels of matching pairs, pick up pairs with most total pixel to be the target. Calculate and return its x and y coordinate.
