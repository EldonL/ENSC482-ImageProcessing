/*Filter.cpp
Created by: Eldon Lin
Last Edited by: Eldon Lin
Contributers: Eldon Lin
Created on 2018-07-06 12:36am
Last edited on 2018-07-06 1:50am
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
	//if file cannot open
	if (!canOpenFile(aSrc))
	{
		return;
	}
	// Create a destination Mat object
	Mat dst;

	// display the source image
	imshow("Before", aSrc);

	for (int i = 1; i<25; i+= 10)
	{
		// Bilateral

		bilateralFilter(aSrc, dst, i,i,i, 0);

		//show the blurred image with the text
		//imshow("Bilateral filter", dst);
		BinarizeByOtsu(dst);
		//cout << "Kernal Size: " << i << " x " << i << endl; //debugging purposes
															//wait for some number of seconds. 1000=1second
		waitKey(1000);
	}

}

//Uses Otsu method to binarize. Also makes picture gray
void Filter::BinarizeByOtsu(Mat aSrc)
{
	Mat bin;
	cvtColor(aSrc, aSrc, COLOR_RGB2GRAY, 0);
	//imshow("after gray", aSrc);
	threshold(aSrc, bin, 0, 255, THRESH_OTSU);
	imshow("Binarized Image", bin);
}//end of Filter.cpp