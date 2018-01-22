# API

## Files
|File Name| Inform|
|------------|:----|
|car_target.py| Armour Detection |
|circle_test.py| A testing program for circle detection|
|tool.py|Unclear|

## Complete Documentation for Car Target Detection

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
* <b>square_ratio</b> - the standard ratio of length and width of a given square
* <b>angle_rate</b> - the value of standard angle between the lengh and width of a given square_ratio
* <b>length_rate</b> -
* <b>matrix threshold</b> -
* <b>DIST</b> - distance of the target

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
