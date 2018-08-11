/*
//Main.cpp
//Created by: Eldon Lin
//Last Edited by: Eldon Lin
//Contributers: Eldon Lin, Jehaan Joseph
//Created on 2018-06-30 10:00pm by Eldon Lin
//Last Edited on 2018-08-10 10:50pm by Eldon Lin
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
H. Kyle "OpenCV Tutorial: Multiple Object Tracking in Real Time (1/3)" https://www.youtube.com/watch?v=RS_uQGOQIdg [Accessed Aug 6, 2018]
H. Kyle "OpenCV Tutorial: Multiple Object Tracking in Real Time (2/3)" https://www.youtube.com/watch?v=ASCi7J5W1FM [Accessed Aug 6, 2018]
H. Kyle "OpenCV Tutorial: Multiple Object Tracking in Real Time (3/3)" https://www.youtube.com/watch?v=4KYlHgQQAts [Accessed Aug 6, 2018]
H. Kyle "OpenCV Tutorial: Real-Time Object Tracking Without Colour" https://www.youtube.com/watch?v=X6rPdRZzgjg [Accessed Aug 6, 2018]
*/



#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <GL/glut.h>
#include <GL/gl.h>
#include "Filter.h"

using namespace cv;
using namespace std;

Mat frame1, frame2;
GLfloat angles = 0.0;
GLuint texture;


//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 50;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
int theObject[2] = { 0,0 };
//bounding rectangle of the object, we will use the center of this as its position.
Rect objectBoundingRectangle = Rect(0, 0, 0, 0);

//threshold for corner detection
int thresh = 150;

struct myclass {
	bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.x < pt2.x); }
} myobject;

struct myclass2 {
	bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.x > pt2.x); }
} myobject2;

//int to string helper function
string intToString(int number) {

	//this function has a number input and string output
	std::stringstream ss;
	ss << number;
	return ss.str();
}


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

void searchForMovement(Mat thresholdImage, Mat &cameraFeed) {
	//notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
	//to take the values passed into the function and manipulate them, rather than just working with a copy.
	//eg. we draw to the cameraFeed to be displayed in the main() function.
	string name;
	bool objectDetected = false;
	Mat temp, temp2;
	//for corner detection
	Mat dst, gray;
	Mat dst_norm, dst_norm_scaled;
	//contains corner points
	vector<Point> points;
	thresholdImage.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	//findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
	findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);// retrieves external contours

	for (int i = 0; i<contours.size(); i++)
	{
		//storing 
		std::vector<cv::Point> shapeToShow;
		cv::approxPolyDP(cv::Mat(contours[i]), shapeToShow, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
		//largestContourVec.push_back(contours.at(contours.size() - 1));
		objectDetected = false;

		//make a bounding rectangle around the largest contour then find its centroid
		//this will be the object's final estimated position.
		objectBoundingRectangle = boundingRect(shapeToShow);
		int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width / 2;
		int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height / 2;

		//update the objects positions by changing the 'theObject' array values
		theObject[0] = xpos, theObject[1] = ypos;

		if (shapeToShow.size() == 3)
		{
			name = "TRI";
			objectDetected = true;
		}
		else if (shapeToShow.size() >= 4 && shapeToShow.size() <= 7)
		{
			// Number of vertices of polygonal curve
			int vtc = shapeToShow.size();
			// Get the cosines of all corners
			std::vector<double> cos;
			for (int j = 2; j < vtc + 1; j++)
				cos.push_back(angle(shapeToShow[j%vtc], shapeToShow[j - 2], shapeToShow[j - 1]));
			// Sort ascending the cosine values
			std::sort(cos.begin(), cos.end());

			// Get the lowest and the highest cosine
			double mincos = cos.front();
			double maxcos = cos.back();
			// Use the degrees obtained above and the number of vertices
			// to determine the shape of the contour
			if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
				name = "RECT";
			else if (vtc == 5 && mincos >= -0.44 && maxcos <= -0.17)
				name = "PENTA";
			else if (vtc == 6 /*&& mincos >= -0.55 && maxcos <= -0.45*/)
				name = "HEXA";
			else if (vtc == 7)
				name = "HEPTA";
			objectDetected = true;
		}
		else
		{
			// Detect and label circles or ovals
			double area = cv::contourArea(contours[i]);
			cv::Rect r = cv::boundingRect(contours[i]);
			int radius = r.width / 2;
			// Number of vertices of polygonal curve
			int vtc = shapeToShow.size();

			if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 && std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
			{
				name = "CIRC";
				objectDetected = true;
			}

			else if (std::abs(1 - ((double)r.width / r.height)) >= 0.2 &&
				std::abs(1 - (area / (CV_PI * (double)r.width * (double)r.height)) >= 0.2))
			{
				name = "OVL";
				objectDetected = true;
			}
			//do not put objectDetected outside of if or else if otherwise, objectDetected will always be true
		}

		//debugging purposes
		//cout << shapeToShow.size() << endl;

		//make some temp x and y variables so we dont have to type out so much
		int x = theObject[0];
		int y = theObject[1];
		if (objectDetected)
		{

			////draw some crosshairs around the object
			//circle(cameraFeed, Point(x, y), 20, Scalar(0, 255, 0), 2);
			//line(cameraFeed, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
			//line(cameraFeed, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
			//line(cameraFeed, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
			//line(cameraFeed, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);

			//write the position of the object to the screen
			//putText(cameraFeed, "Tracking object at (" + intToString(x) + "," + intToString(y) + ")", Point(x, y), 1, 1, Scalar(255, 0, 0), 2);

			//sort(shapeToShow.begin(), shapeToShow.begin() + (shapeToShow.size()) / 2, myobject);
			//sort(shapeToShow.begin() + (shapeToShow.size()) / 2, shapeToShow.end(), myobject2);

			for (int i = 0; i < shapeToShow.size(); i++) {

				if (i + 1 == shapeToShow.size())
				{
					line(cameraFeed, shapeToShow.at(i), shapeToShow.at(0), Scalar(0, 255, 0), 2, 8);
					line(cameraFeed, shapeToShow.at(i)-Point(35,35), shapeToShow.at(0)- Point(35, 35), Scalar(0, 255, 0), 2, 8);
					line(cameraFeed, shapeToShow.at(i), shapeToShow.at(i) - Point(35, 35), Scalar(0, 255, 0), 2, 8);
					line(cameraFeed, shapeToShow.at(0), shapeToShow.at(0) - Point(35, 35), Scalar(0, 255, 0), 2, 8);
				}
				else
				{
					line(cameraFeed, shapeToShow.at(i), shapeToShow.at(i+1), Scalar(0, 255, 0), 2, 8);
					line(cameraFeed, shapeToShow.at(i) - Point(35, 35), shapeToShow.at(i + 1) - Point(35, 35), Scalar(0, 255, 0), 2, 8);
					line(cameraFeed, shapeToShow.at(i), shapeToShow.at(i) - Point(35, 35), Scalar(0, 255, 0), 2, 8);
					line(cameraFeed, shapeToShow.at(i + 1), shapeToShow.at(i + 1) - Point(35, 35), Scalar(0, 255, 0), 2, 8);
				}
					

			}

			putText(cameraFeed, name, Point(x, y), 1, 1, Scalar(0, 255, 255), 2);
		}

	}

}

int main(int argc, char **argv) {
	

	//some boolean variables for added functionality
	bool objectDetected = false;
	//these two can be toggled by pressing 'd' or 't'
	bool debugMode = false;
	bool trackingEnabled = true;
	//pause and resume code
	bool pause = false;
	//set up the matrices that we will need
	//the two frames we will be comparing

	//their grayscale images (needed for absdiff() function)
	Mat grayImage1, grayImage2;
	//resulting difference image
	Mat differenceImage;
	//thresholded difference image (for use in findContours() function)
	Mat thresholdImage;
	//video capture object.
	VideoCapture capture;


	capture.open(0);

	if (!capture.isOpened()) {
		cout << "ERROR ACQUIRING VIDEO FEED\n";
		getchar();
		return -1;
	}
	while (1) {

		//we can loop the video by re-opening the capture every time the video reaches its last frame


		//check if the video has reach its last frame.
		//we add '-1' because we are reading two frames from the video at a time.
		//if this is not included, we get a memory error!

		//read first frame
		capture.read(frame1);
		
		//convert frame1 to gray scale for frame differencing
		cv::cvtColor(frame1, grayImage1, COLOR_BGR2GRAY);
	
		//threshold intensity image at a given sensitivity value
		cv::threshold(grayImage1, thresholdImage, SENSITIVITY_VALUE, 255, THRESH_BINARY);
		GaussianBlur(thresholdImage, thresholdImage, Size(7, 7), 1.5, 1.5);
		Canny(thresholdImage, thresholdImage, 0, 3000, 5);



		if (debugMode == true) {
			//show the difference image and threshold image
			
			cv::imshow("Threshold Image", thresholdImage);

			// Showing the result
			//namedWindow("corners_window", CV_WINDOW_AUTOSIZE);
			//imshow("corners_window", dst_norm_scaled);
		}
		else {
			//if not in debug mode, destroy the windows so we don't see them anymore
			cv::destroyWindow("Difference Image");
			cv::destroyWindow("Threshold Image");
		}
		//blur the image to get rid of the noise. This will output an intensity image
		//cv::blur(thresholdImage, thresholdImage, cv::Size(BLUR_SIZE, BLUR_SIZE));
		//threshold again to obtain binary image from blur output
		//cv::threshold(thresholdImage, thresholdImage, SENSITIVITY_VALUE, 255, THRESH_BINARY);


		if (debugMode == true) {
			//show the threshold image after it's been "blurred"

			imshow("Final Threshold Image", thresholdImage);

		}
		else {
			//if not in debug mode, destroy the windows so we don't see them anymore
			cv::destroyWindow("Final Threshold Image");
		}

		//if tracking enabled, search for contours in our thresholded image
		if (trackingEnabled) {

			searchForMovement(thresholdImage, frame1);
		}

		////create new windows
		//namedWindow("OPENGL CAMERA", WINDOW_OPENGL);
		//// enable texture
		//glEnable(GL_TEXTURE_2D);
		//setOpenGlDrawCallback("OPENGL CAMERA", on_opengl);
		////while (waitKey(30) != 'q')
		//{
		//	capture>> frame1;
		//	//create first texture
		//	loadTexture();
		//	updateWindow("OPENGL CAMERA");
		//	angles = angles + 4;
		//}
		//show our captured frame
		imshow("Frame1", frame1);
		//check to see if a button has been pressed.
		//this 10ms delay is necessary for proper operation of this program
		//if removed, frames will not have enough time to referesh and a blank 
		//image will appear.
		switch (waitKey(10)) {

		case 27: //'esc' key has been pressed, exit program.
			return 0;
		case 116: //'t' has been pressed. this will toggle tracking
			trackingEnabled = !trackingEnabled;
			if (trackingEnabled == false) cout << "Tracking disabled." << endl;
			else cout << "Tracking enabled." << endl;
			break;
		case 100: //'d' has been pressed. this will debug mode
			debugMode = !debugMode;
			if (debugMode == false) cout << "Debug mode disabled." << endl;
			else cout << "Debug mode enabled." << endl;
			break;
		case 112: //'p' has been pressed. this will pause/resume the code.
			pause = !pause;
			if (pause == true) {
				cout << "Code paused, press 'p' again to resume" << endl;
				while (pause == true) {
					//stay in this loop until 
					switch (waitKey()) {
						//a switch statement inside a switch statement? Mind blown.
					case 112:
						//change pause back to false
						pause = false;
						cout << "Code ResumedT" << endl;
						break;
					}
				}
			}
		}
	}

	return 0;

}