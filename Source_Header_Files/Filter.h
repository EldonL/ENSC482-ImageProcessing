#pragma once
/*Filter.h
Created by Eldon Lin
Last Edited by: Eldon Lin
Contributers: Eldon Lin
Created on 2018-07-06 12:36am
Last Edited on 2018-07-09 12:33am by Eldon Lin
Class Description: filters, grey and binarizes images. Can check if file can open
Class invariant: a source must be passed through the function.
*/

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

class Filter
{
private: 
	int argc;
	char** argv;
	//Uses Otsu method to binarize. Also makes picture gray
	void BinarizeByOtsu(Mat aSrc);
	Mat bin;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
public: 
	
	//Check if file can open
	bool canOpenFile(Mat aSrc);
	/*Quote from refernce: Bilateral filter can reduce noise of the image while preserving the edges.
	The drawback of this type of filter is that it takes longer to filter the input image*/
	//Uses bilateral filtering to filter
	void smoothingByBilateral(Mat aSrc);

	//retrieves contours from the binary image and draws the image
	//Function invariance: must pass in a Binary image 
	void FindContours(Mat aBin);

	//returns binary image
	Mat getBin();


	
};
//end of Filter.h
