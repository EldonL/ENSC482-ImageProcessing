/*
//Source.cpp
//Created by: Eldon Lin
//Last Edited by: Eldon Lin
//Contributers: Eldon Lin, Jehaan Joseph
//Created on 2018-06-30 10:00pm by Eldon Lin
//Last Edited on 2018-07-30 3:56pm by Eldon Lin
//References
//K. Hong. "Filters A - Average and GaussianBlur." http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_imgproc_gausian_median_blur_bilateral_filter_image_smoothing.php. [Accessed: July 2, 2018]
//K. Hong. "Filters B - MedianBlur and Bilateral." http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_imgproc_gausian_median_blur_bilateral_filter_image_smoothing_B.php. [Accesed: July 2, 2018]
//"Getting Started with Images." https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html. Nov 10,2014. [Accessed July 2, 2018]
//"Finding contours in your image." https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html. [Accessed July 8, 2018]
//"Learn OpenCV by Examples." http://opencvexamples.blogspot.com/2013/09/find-contour.html. [Accessed July 8, 2018]
//A. Tiger. "opencv project Digital Image Processing." https://opencvproject.wordpress.com/projects-files/detection-shape/. [Accessed July 8,2018].
// "Canny Edge Detection." https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html [Accessed July 19,2018]
//"Reading and Writing Video" https://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html [Accessed July 30, 2018]
// "Ball Tracking / Detection using OpenCV " http://anikettatipamula.blogspot.com/2012/12/ball-tracking-detection-using-opencv.html [Accessed July 30, 2018]
// "How to Detect and Track Object With OpenCV" https://www.intorobotics.com/how-to-detect-and-track-object-with-opencv/ [Accessed July 19,2018]
// "OpenCV Basics - 12 - Webcam & Video Capture" https://www.youtube.com/watch?v=zhEqiW3qnos [Accessed: July 30,2018]
H. Kyle "Tutorial: Real-Time Object Tracking Using OpenCV" https://www.youtube.com/watch?v=bSeFrPrqZ2A [Accessed July 30,2018]
*/


/*
Possible Logic for camera capture with shape recognition
1. First I have converted the image from the camera from RGB to HSV
2. Then I have converted it into binary image using inRange() function
3. After that I have used erode() and dilate() functions to reduce the noise
4. After getting a binary image with a reduced noise, I have used cvFindContours() function to find all the contours and then used cvApproxPoly() function to count the number of the edges.
5.The number of edges should tell you about the type of few simple shapes.
References:
"Shape Detection & Tracking using Contours" https://www.opencv-srf.com/2011/09/object-detection-tracking-using-contours.html [Accessed July 30,2018]
"Shape Recognition using OpenCV" https://www.youtube.com/watch?v=_LYGuOmq0c0 [Accessed July 30,2018]

*/


/**
* Simple shape detector program.
* It loads an image and tries to find simple shapes (rectangle, triangle, circle, etc) in it.
* This program is a modified version of `squares.cpp` found in the OpenCV sample dir.
*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include "opencv2/opencv.hpp"


#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include "Filter.h"

using namespace cv;
using namespace std;

#pragma region JC codes and Camera link 

Mat src, erosion_dst, dilation_dst;
bool isCircle = false;

//Helper function to find a cosine of angle between vectors
//from pt0->pt1 and pt0->pt2

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


//Helper function to display text in the center of a contour

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

void FindShapeContours(Mat src)
{
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
	//Original code was 5. Changed to 7
	cv::Mat bw;
	cv::Canny(gray, bw, 0, 500, 5);

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
		else if (approx.size() >= 4 && approx.size() <= 7)
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
			else if (vtc == 7)
				setLabel(dst, "hepta", contours[i]);
			/*else if (vtc == 8 && mincos >= -0.55 && maxcos <= -0.45)
			setLabel(dst, "Octa", contours[i]);*/
			else
				setLabel(dst, "unknown", contours[i]);

		}
		else
		{
			// Detect and label circles or ovals
			double area = cv::contourArea(contours[i]);
			cv::Rect r = cv::boundingRect(contours[i]);
			int radius = r.width / 2;
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
			vector<Vec3f> circles;
			//reading from gray scale
			HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows / 8, 200, 100, 0, 0);
			for (size_t i = 0; i < circles.size(); i++)
			{
				Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
				int radius = cvRound(circles[i][2]);
				// circle center
				//circle(dst, center, 3, Scalar(0, 255, 0), -1, 8, 0);
				// circle outline
				circle(dst, center, radius, Scalar(0, 0, 255), 3, 8, 0);
				isCircle = true;
			}

			if (isCircle)
			{
				setLabel(dst, "cir", contours[i]);
				isCircle = false;
			}
			else if (vtc == 8)
			{
				setLabel(dst, "octagon", contours[i]);
			}
			else if (std::abs(1 - ((double)r.width / r.height)) >= 0.2 &&
				std::abs(1 - (area / (CV_PI * (double)r.width * (double)r.height)) >= 0.2))
				setLabel(dst, "OVL", contours[i]);
			else
				setLabel(dst, "round obj", contours[i]);
			/*if (vtc == 8 && mincos >= -0.75 && maxcos <= -0.7)
			setLabel(dst, "Octagon", contours[i]);
			else if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
			std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
			setLabel(dst, "CIR", contours[i]);
			else if (std::abs(1 - ((double)r.width / r.height)) >= 0.2 &&
			std::abs(1 - (area / (CV_PI * (double)r.width * (double)r.height)) >= 0.2))
			setLabel(dst, "OVL", contours[i]);
			else
			setLabel(dst, "round obj", contours[i]);*/
		}
	}
	cv::imshow("gray", gray);
	cv::imshow("bw", bw);
	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::waitKey(0);
}

int CameraLink()
{
	//Video stuff
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges;
	namedWindow("edges", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, edges, COLOR_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
		imshow("edges", edges);
		if (waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;

}
#pragma endregion 


#pragma region TrackBarCodes from online



//objectTrackingTutorial.cpp

//Written by  Kyle Hounslow 2013

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software")
//, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//IN THE SOFTWARE.


//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";
void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed





}
string intToString(int number) {


	std::stringstream ss;
	ss << number;
	return ss.str();
}
/*void createTrackbars() {
//create window for trackbars


namedWindow(trackbarWindowName, 0);
//create memory to store trackbar name on window
char TrackbarName[50];
sprintf(TrackbarName, "H_MIN", H_MIN);
sprintf(TrackbarName, "H_MAX", H_MAX);
sprintf(TrackbarName, "S_MIN", S_MIN);
sprintf(TrackbarName, "S_MAX", S_MAX);
sprintf(TrackbarName, "V_MIN", V_MIN);
sprintf(TrackbarName, "V_MAX", V_MAX);
//create trackbars and insert them into window
//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
//the max value the trackbar can move (eg. H_HIGH),
//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
//                                  ---->    ---->     ---->
createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);


}*/
void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25>0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25<FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25>0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25<FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}
void morphOps(Mat &thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);



}
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects<MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				drawObject(x, y, cameraFeed);
			}

		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
}

/*void TrackBarCodes()
{
//some boolean variables for different functionality within this
//program
bool trackObjects = true;
bool useMorphOps = true;
//Matrix to store each frame of the webcam feed
Mat cameraFeed;
//matrix storage for HSV image
Mat HSV;
//matrix storage for binary threshold image
Mat threshold;
//x and y values for the location of the object
int x = 0, y = 0;
//create slider bars for HSV filtering
createTrackbars();
//video capture object to acquire webcam feed
VideoCapture capture;
//open capture object at location zero (default location for webcam)
capture.open(0);
//set height and width of capture frame
capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
//start an infinite loop where webcam feed is copied to cameraFeed matrix
//all of our operations will be performed within this loop
while (1) {
//store image to matrix
capture.read(cameraFeed);
//convert frame from BGR to HSV colorspace
cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
//filter HSV image between values and store filtered image to
//threshold matrix
inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);
//perform morphological operations on thresholded image to eliminate noise
//and emphasize the filtered object(s)
if (useMorphOps)
morphOps(threshold);
//pass in thresholded frame to our object tracking function
//this function will return the x and y coordinates of the
//filtered object
if (trackObjects)
trackFilteredObject(x, y, threshold, cameraFeed);

//show frames
imshow(windowName2, threshold);
imshow(windowName, cameraFeed);
imshow(windowName1, HSV);


//delay 30ms so that screen can refresh.
//image will not appear without this waitKey() command
waitKey(30);
}
}*/
#pragma endregion

int main(int argc, char* argv[])
{

	/*INSTRUCTION: Comment out either "JC and Camera Codes" (4 lines)  or TrackBarCodes (1 line)*/

	//JC and Camera Codes

	VideoCapture cap;
	// open the default camera, use something different from 0 otherwise;
	// Check VideoCapture documentation.
	if (!cap.open(0))
		return 0;
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) break; // end of video stream
		imshow("this is you, smile! :)", frame);
		imwrite("../data/lena.jpg", frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}

	CommandLineParser parser(argc, argv, "{@input | ../data/lena.jpg | input image}");
	src = imread(parser.get<String>("@input"), IMREAD_COLOR);
	FindShapeContours(src);
	//CameraLink();

	//Online Codes
	//Converts RGB video to HSV. Tracks a specific object. Found on "Tutorial: Real-Time Object Tracking Using OpenCV" https://www.youtube.com/watch?v=bSeFrPrqZ2A
	//TrackBarCodes();

	return 0;
}