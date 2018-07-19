/**
//Source.cpp
//Created by: Eldon Lin
//Last Edited by: Eldon Lin
//Contributers: Eldon Lin, Jehaan Joseph
//Created on 2018-06-30 10:00pm by Eldon Lin
//Last Edited on 2018-07-19 11:45pm by Eldon Lin
//References
//K. Hong. "Filters A - Average and GaussianBlur." http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_imgproc_gausian_median_blur_bilateral_filter_image_smoothing.php. [Accessed: July 2, 2018]
//K. Hong. "Filters B - MedianBlur and Bilateral." http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_imgproc_gausian_median_blur_bilateral_filter_image_smoothing_B.php. [Accesed: July 2, 2018]
//"Getting Started with Images." https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html. Nov 10,2014. [Accessed July 2, 2018]
//"Finding contours in your image." https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html. [Accessed July 8, 2018]
//"Learn OpenCV by Examples." http://opencvexamples.blogspot.com/2013/09/find-contour.html. [Accessed July 8, 2018]
//A. Tiger. "opencv project Digital Image Processing." https://opencvproject.wordpress.com/projects-files/detection-shape/. [Accessed July 8,2018].
// "Canny Edge Detection." https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html [Accessed July 19,2018]
//*/

/**
* Simple shape detector program.
* It loads an image and tries to find simple shapes (rectangle, triangle, circle, etc) in it.
* This program is a modified version of `squares.cpp` found in the OpenCV sample dir.
*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>


#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include "Filter.h"

using namespace cv;
using namespace std;

Mat src, erosion_dst, dilation_dst;

/**
* Helper function to find a cosine of angle between vectors
* from pt0->pt1 and pt0->pt2
*/
static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

/**
* Helper function to display text in the center of a contour
*/
void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
	cv::Rect r = cv::boundingRect(contour);

	cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255, 255, 255), CV_FILLED);
	cv::putText(im, label, pt, fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
}

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, "{@input | ../data/lena.jpg | input image}");
	src = imread(parser.get<String>("@input"),IMREAD_COLOR);
	/*if (src.empty())
		return -1;*/
	/*Filter filter;
	filter.smoothingByBilateral(src);*/
	

	// Convert to grayscale
	cv::Mat gray;
	cv::cvtColor(src, gray, CV_BGR2GRAY);

	// Use Canny instead of threshold to catch squares with gradient shading
	//Canny is a popular edge detection algorithm that does:
	//noise reduction
	//Finding intensity gradient of the image
	//Non-maximum suppression
	//Hysteresis Thresholding
	//Canny does the binarization
	//last parameter must be between 3 to 7, it is the sobel kernel size. Larger, the more blurry.
	//Original code was 5 change to 7
	cv::Mat bw;
	cv::Canny(gray, bw, 0, 500, 7);
	

	// Find contours
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	std::vector<cv::Point> approx;
	cv::Mat dst = src.clone();

	for (int i = 0; i < contours.size(); i++)
	{
		// Approximate contour with accuracy proportional
		// to the contour perimeter
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);

		// Skip small or non-convex objects 
		if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
			continue;

		if (approx.size() == 3)
		{
			setLabel(dst, "TRI", contours[i]);    // Triangles
		}
		else if (approx.size() >= 4 && approx.size() <= 6)
		{
			// Number of vertices of polygonal curve
			int vtc = approx.size();

			// Get the cosines of all corners
			std::vector<double> cos;
			for (int j = 2; j < vtc + 1; j++)
				cos.push_back(angle(approx[j%vtc], approx[j - 2], approx[j - 1]));

			// Sort ascending the cosine values
			std::sort(cos.begin(), cos.end());

			// Get the lowest and the highest cosine
			double mincos = cos.front();
			double maxcos = cos.back();

			// Use the degrees obtained above and the number of vertices
			// to determine the shape of the contour
			if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
				setLabel(dst, "RECT", contours[i]);
			else if (vtc == 5 && mincos >= -0.44 && maxcos <= -0.17)
				setLabel(dst, "PENTA", contours[i]);
			//else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45)
			else if (vtc == 6 /*&& mincos >= -0.55 && maxcos <= -0.45*/)
				setLabel(dst, "HEXA", contours[i]);
		}
		else
		{
			// Detect and label circles or ovals
			double area = cv::contourArea(contours[i]);
			cv::Rect r = cv::boundingRect(contours[i]);
			int radius = r.width / 2;

			if (std::abs(1 - ((double)r.width / r.height)) <= 0.035 &&
				std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.035)
				setLabel(dst, "CIR", contours[i]);
			else if (std::abs(1 - ((double)r.width / r.height)) >= 0.2 &&
				std::abs(1 - (area / (CV_PI * (double)r.width * (double)r.height)) >= 0.2))
				setLabel(dst, "OVL", contours[i]);
		}
	}
	cv::imshow("bw", bw);
	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::waitKey(0);
	return 0;
}

//
//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//
//Mat src, erosion_dst, dilation_dst;
//
//
//
//#include "Filter.h"
//
//
//#pragma region main
//int main(int argc, char** argv)
//{
//	//The first four lines below is how you would call an image. The output would show the before image and the binarized image
//	
//	/// Load an image
//	CommandLineParser parser(argc, argv, "{@input | ../data/lena.jpg | input image}");
//	src = imread(parser.get<String>("@input"),IMREAD_COLOR);
//	Filter filter; 
//	filter.smoothingByBilateral(src);
//	
//
//	system("pause");
//	
//
//}
//#pragma endregion
//
////for (int i = 0; i < contours.size(); i++)
////{
////	// Approximate contour with accuracy proportional
////	// to the contour perimeter
////	cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
////
////	// Skip small or non-convex objects
////	if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
////		continue;
////
////	if (approx.size() == 3)
////	{
////		setLabel(dst, "TRI", contours[i]);    // Triangles
////	}
////	else if (approx.size() >= 4 && approx.size() <= 6)
////	{
////		// Number of vertices of polygonal curve
////		int vtc = approx.size();
////
////		// Get the cosines of all corners
////		std::vector cos;
////		for (int j = 2; j < vtc + 1; j++)
////			cos.push_back(angle(approx[j%vtc], approx[j - 2], approx[j - 1]));
////
////		// Sort ascending the cosine values
////		std::sort(cos.begin(), cos.end());
////
////		// Get the lowest and the highest cosine
////		double mincos = cos.front();
////		double maxcos = cos.back();
////
////		// Use the degrees obtained above and the number of vertices
////		// to determine the shape of the contour
////		if (vtc == 4)
////			setLabel(dst, "RECT", contours[i]);
////		else if (vtc == 5)
////			setLabel(dst, "PENTA", contours[i]);
////		else if (vtc == 6)
////			setLabel(dst, "HEXA", contours[i]);
////	}
////	else
////	{
////		// Detect and label circles
////		double area = cv::contourArea(contours[i]);
////		cv::Rect r = cv::boundingRect(contours[i]);
////		int radius = r.width / 2;
////
////		if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
////			std::abs(1 - (area / (CV_PI * (radius*radius)))) <= 0.2)
////			setLabel(dst, "CIR", contours[i]);
////	}
////}
////cv::imshow("src", src);
////cv::imshow("dst", dst);