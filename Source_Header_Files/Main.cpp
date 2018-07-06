/**
Source.cpp
Created by: Eldon Lin
Last Edited by: Eldon Lin
Contributers: Eldon Lin
Created on 2018-06-30 10:00pm by Eldon Lin
Last Edited on 2018-07-06 1:49am by Eldon Lin
References
K. Hong. "Filters A - Average and GaussianBlur." http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_imgproc_gausian_median_blur_bilateral_filter_image_smoothing.php. [Accessed: July 2, 2018]
K. Hong. "Filters B - MedianBlur and Bilateral." http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_imgproc_gausian_median_blur_bilateral_filter_image_smoothing_B.php. [Accesed: July 2, 2018]
"Getting Started with Images." https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html. Nov 10,2014. [Accessed July 2, 2018]
*/

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;


Mat src, erosion_dst, dilation_dst;



#include "Filter.h"


#pragma region main
int main(int argc, char** argv)
{
	//The first four lines below is how you would call an image. The output would show the before image and the binarized image
	
	/// Load an image
	CommandLineParser parser(argc, argv, "{@input | ../data/lena.jpg | input image}");
	src = imread(parser.get<String>("@input"),IMREAD_COLOR);
	Filter filter; 
	filter.smoothingByBilateral(src);
	system("pause");
	

}
#pragma endregion

