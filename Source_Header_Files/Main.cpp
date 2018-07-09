/**
Source.cpp
Created by: Eldon Lin
Last Edited by: Eldon Lin
Contributers: Eldon Lin
Created on 2018-06-30 10:00pm by Eldon Lin
Last Edited on 2018-07-09 12:33am by Eldon Lin
References
K. Hong. "Filters A - Average and GaussianBlur." http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_imgproc_gausian_median_blur_bilateral_filter_image_smoothing.php. [Accessed: July 2, 2018]
K. Hong. "Filters B - MedianBlur and Bilateral." http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_imgproc_gausian_median_blur_bilateral_filter_image_smoothing_B.php. [Accesed: July 2, 2018]
"Getting Started with Images." https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html. Nov 10,2014. [Accessed July 2, 2018]
"Finding contours in your image." https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html. [Accessed July 8, 2018]
"Learn OpenCV by Examples." http://opencvexamples.blogspot.com/2013/09/find-contour.html. [Accessed July 8, 2018]
A. Tiger. "opencv project Digital Image Processing" https://opencvproject.wordpress.com/projects-files/detection-shape/. [Accessed July 8,2018].
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

//for (int i = 0; i < contours.size(); i++)
//{
//	// Approximate contour with accuracy proportional
//	// to the contour perimeter
//	cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
//
//	// Skip small or non-convex objects
//	if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
//		continue;
//
//	if (approx.size() == 3)
//	{
//		setLabel(dst, "TRI", contours[i]);    // Triangles
//	}
//	else if (approx.size() >= 4 && approx.size() <= 6)
//	{
//		// Number of vertices of polygonal curve
//		int vtc = approx.size();
//
//		// Get the cosines of all corners
//		std::vector cos;
//		for (int j = 2; j < vtc + 1; j++)
//			cos.push_back(angle(approx[j%vtc], approx[j - 2], approx[j - 1]));
//
//		// Sort ascending the cosine values
//		std::sort(cos.begin(), cos.end());
//
//		// Get the lowest and the highest cosine
//		double mincos = cos.front();
//		double maxcos = cos.back();
//
//		// Use the degrees obtained above and the number of vertices
//		// to determine the shape of the contour
//		if (vtc == 4)
//			setLabel(dst, "RECT", contours[i]);
//		else if (vtc == 5)
//			setLabel(dst, "PENTA", contours[i]);
//		else if (vtc == 6)
//			setLabel(dst, "HEXA", contours[i]);
//	}
//	else
//	{
//		// Detect and label circles
//		double area = cv::contourArea(contours[i]);
//		cv::Rect r = cv::boundingRect(contours[i]);
//		int radius = r.width / 2;
//
//		if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
//			std::abs(1 - (area / (CV_PI * (radius*radius)))) <= 0.2)
//			setLabel(dst, "CIR", contours[i]);
//	}
//}
//cv::imshow("src", src);
//cv::imshow("dst", dst);