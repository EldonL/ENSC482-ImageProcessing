/*Filter.cpp
Created by: Eldon Lin
Last Edited by: Eldon Lin
Contributers: Eldon Lin
Created on 2018-07-06 12:36am
Last Edited on 2018-07-15 10:29 by Eldon Lin
Class Description: filters, grey and binarizes images. Can check if file can open
Class invariant: a source must be passed through the function. 
*/


#include "Filter.h"

//Check if file can open
bool Filter::canOpenFile(Mat aSrc)
{
	if (aSrc.empty())
	{
		cout << "Could not open or find the image!\n" << endl;

		return false;
	}
	return true;
}

/*Quote from refernce: Bilateral filter can reduce noise of the image while preserving the edges.
The drawback of this type of filter is that it takes longer to filter the input image*/
//Uses bilateral filtering to filter
void Filter:: smoothingByBilateral(Mat aSrc)
{
	cout << "loading..." << endl;
	//if file cannot open
	if (!canOpenFile(aSrc))
	{
		return;
	}
	// Create a destination Mat object
	Mat otsu;

	// display the source image
	imshow("Original", aSrc);

	for (int i = 1; i<50; i+= 10)
	{
		// Bilateral

		bilateralFilter(aSrc, otsu, i,i,i, 0);

		//show the blurred image with the text
		//imshow("Bilateral filter", dst);
		BinarizeByOtsu(otsu);
		FindContours(bin);
		//cout << "Kernal Size: " << i << " x " << i << endl; //debugging purposes
															//wait for some number of seconds. 1000=1second
		waitKey(1000);
	}
	Label(drawing);
	cout << "done" << endl;
}

//Uses Otsu method to binarize. Also makes picture gray
void Filter::BinarizeByOtsu(Mat aSrc)
{
	
	cvtColor(aSrc, aSrc, COLOR_RGB2GRAY, 0);
	//imshow("after gray", aSrc);
	threshold(aSrc, bin, 0, 255, THRESH_OTSU);
	//imshow("Binarized Image", bin);
	

}//end of Filter.cpp


 //retrieves contours from the binary image and draws the image
 //Function invariance: must pass in a Binary image 
void Filter::FindContours(Mat aBin)
{
    
	RNG rng(1234);
	findContours(bin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	drawing = Mat::zeros(bin.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(255, 255), rng.uniform(255, 255), rng.uniform(255, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	/// Show in a window
	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//imshow("Contours", drawing);


}

void Filter::Label(Mat aSrc)
{
	dst = aSrc.clone();

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
			else if (vtc == 5 /*&& mincos >= -0.34 && maxcos <= -0.27*/)
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

			if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
				std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
				setLabel(dst, "CIR", contours[i]);
			else if (std::abs(1 - ((double)r.width / r.height)) >= 0.2 &&
				std::abs(1 - (area / (CV_PI * (double)r.width * (double)r.height)) >= 0.2))
				setLabel(dst, "OVL", contours[i]);
		}
	}

	//cv::imshow("src", aSrc);
	cv::imshow("dst", dst);
	cv::waitKey(0);
	
}

void Filter::setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
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

/**
* Helper function to find a cosine of angle between vectors
* from pt0->pt1 and pt0->pt2
*/
double Filter:: angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
//returns binary image
Mat Filter:: getBin()
{
	return bin;
}