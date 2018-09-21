// ------------------------------------------------------------------------------------------------
// SimplerBlobDetector.cpp
// Joost van Stuijvenberg
// Avans Hogeschool Breda
// August 2018
//
// Simplified version of OpenCV's SimpleBlobDetector, for educational purposes. The code obtained
// from https://github.com/opencv/opencv/blob/master/modules/features2d/src/blobdetector.cpp has
// been stripped from all unnecessary details and augmented with comments. Furthermore, it was
// converted into a standalone application.
// ------------------------------------------------------------------------------------------------

#include <iostream>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

// ------------------------------------------------------------------------------------------------
// Center struct, as used in findBlobs().
// ------------------------------------------------------------------------------------------------
struct Center
{
	Point2d location;
	double radius;
	double confidence;
};

// ------------------------------------------------------------------------------------------------
// Globals.
// ------------------------------------------------------------------------------------------------
SimpleBlobDetector::Params params;
int window = 0;

// ------------------------------------------------------------------------------------------------
// onMouseClick() - Mouse click callback handler.
// ------------------------------------------------------------------------------------------------
// PRE  : event contains the mouse event type
// PRE  : x and y contain the click location
// PRE  : userdata may be a reference to the Mat object that caught the mouse click
// POST : grayvalue of the clicked pixel is written to console output
// ------------------------------------------------------------------------------------------------
void onMouseClick(int event, int x, int y, int flags, void* userdata)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
	    assert(userdata != nullptr);
	    assert(x > 0 && y > 0 && x < ((cv::Mat *)userdata)->size().width && y < ((cv::Mat *)userdata)->size().height);

		cv::Mat image(*(cv::Mat *)userdata);
		int gr = image.at<uchar>(y, x);
		cout << "Grayvalue = " << gr << endl;
	}
}

// ------------------------------------------------------------------------------------------------
// findBlobs() - Uses the findContours()-function to find contours in the binary image.
// ------------------------------------------------------------------------------------------------
// PRE  : _binaryImage contains a valid binary image
// PRE  : centers is a reference to a vector of Center instances
// POST : centers contains Center objects for each blob that was found
// ------------------------------------------------------------------------------------------------
void findBlobs(InputArray _image, InputArray _binaryImage, vector<Center> &centers, int threshold)
{
	Scalar contourColor = Scalar(255, 0, 0);				 // Blue
	Scalar filteredByAreaColor = Scalar(0, 255, 0);          // Green
	Scalar filteredByCircularityColor = Scalar(255, 255, 0); // Cyan
	Scalar filteredByInertiaColor = Scalar(0, 255, 255);     // Yellow
	Scalar filteredByConvexityColor = Scalar(255, 0, 255);
	Scalar filteredByColorColor = Scalar(255, 255, 255);     // White
	Scalar selectedColor = Scalar(0, 0, 255);                // Red

	Mat image = _image.getMat(), binaryImage = _binaryImage.getMat();
	centers.clear();

	// Find contours in the binary image using the findContours()-function. Let this function
	// return a list of contours only (no hierarchical data).
	vector<vector<Point> > contours;
	Mat tmpBinaryImage = binaryImage.clone();
	findContours(tmpBinaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);

	// Now process all the contours that were found.
	Mat inter;
	cvtColor(binaryImage, inter, COLOR_GRAY2BGR);
	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{
		Center center;
		center.confidence = 1;
		Moments moms = moments(Mat(contours[contourIdx]));

		// Show the current contour in blue, initially.
		drawContours(inter, contours, contourIdx, contourColor, 3);
		drawContours(inter, contours, contourIdx,  3);

		// If filtering by area is requested, see if the area of the current contour
		// is within the specified min and max limits, excluding min and including max.
		if (params.filterByArea)
		{
			double area = moms.m00;
			if (area < params.minArea || area >= params.maxArea)
			{
				drawContours(inter, contours, contourIdx, filteredByAreaColor, 3);
				continue;
			}
		}

		// If filtering by circularity is requested, see if the current contour has a
		// sufficient circularity.
		if (params.filterByCircularity)
		{
			double area = moms.m00;
			double perimeter = arcLength(Mat(contours[contourIdx]), true);
			double ratio = 4 * CV_PI * area / (perimeter * perimeter);
			if (ratio < params.minCircularity || ratio >= params.maxCircularity)
			{
				drawContours(inter, contours, contourIdx, filteredByCircularityColor, 3);
				continue;
			}
		}

		// If filtering by inertia is requested, 
		if (params.filterByInertia)
		{
			double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
			const double eps = 1e-2;
			double ratio;
			if (denominator > eps)
			{
				double cosmin = (moms.mu20 - moms.mu02) / denominator;
				double sinmin = 2 * moms.mu11 / denominator;
				double cosmax = -cosmin;
				double sinmax = -sinmin;

				double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
				double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
				ratio = imin / imax;
			}
			else
				ratio = 1;

			if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio)
			{
				drawContours(inter, contours, contourIdx, filteredByInertiaColor, 3);
				continue;
			}

			center.confidence = ratio * ratio;
		}

		// If filtering by convexity is requested, skip this contour if the ratio between the contour
		// area and the hull area is not within the specified limits.
		if (params.filterByConvexity)
		{
			std::vector<Point> hull;
			convexHull(Mat(contours[contourIdx]), hull);
			double area = contourArea(Mat(contours[contourIdx]));
			double hullArea = contourArea(Mat(hull));
			double ratio = area / hullArea;
			if (ratio < params.minConvexity || ratio >= params.maxConvexity)
			{
				drawContours(inter, contours, contourIdx, filteredByConvexityColor, 3);
				continue;
			}
		}

		// Prevent division by zero, should this contour have no area.
		if (moms.m00 == 0.0)
			continue;
		center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

		// If filtering by color was requested, skip this contour if the center pixel's color does 
		// not match the specified color. Note that we are processing gray scale images here...
		if (params.filterByColor)
			if (binaryImage.at<uchar>(cvRound(center.location.y), cvRound(center.location.x)) != params.blobColor)
			{
				drawContours(inter, contours, contourIdx, filteredByColorColor, 3);
				continue;
			}

		// By the time we reach here, the current contour apparently hasn't been filtered out,
		// so we compute the blob radius and store it as a Center in the centers vector.
		std::vector<double> dists;
		for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
		{
			Point2d pt = contours[contourIdx][pointIdx];
			dists.push_back(norm(center.location - pt));
		}
		std::sort(dists.begin(), dists.end());
		center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
		centers.push_back(center);

		// If a contour has not been filtered out, overwrite it in red.
		drawContours(inter, contours, contourIdx, selectedColor, 2);
	}

	// Let the window border reflect the number of found and filtered contours.
	ostringstream os;
	os << "Threshold: " << threshold << ", number of contours found: " << contours.size();
	os << ", remaining after filtering: " << centers.size();
	imshow(os.str(), inter);
	moveWindow(os.str(), 100 + window * 50, 100 + window++ * 50);
}

// ------------------------------------------------------------------------------------------------
// detect()- Detects blobs in an image.
// ------------------------------------------------------------------------------------------------
// PRE  : image contains an image
// PRE  : keypoints is a reference to a vector of KeyPoint instances
// PRE  : mask is an ROI matrix with nonzero values for pixels of interest
// POST : keypoints contains a KeyPoint instance for each blob that was found
// ------------------------------------------------------------------------------------------------
void detect(InputArray image, vector<KeyPoint>& keypoints, InputArray mask)
{
	keypoints.clear();

	// When necessary, convert the supplied image to a gray scale image.
	Mat grayscaleImage;
	if (image.channels() == 3 || image.channels() == 4)
		cvtColor(image, grayscaleImage, COLOR_BGR2GRAY);
	else
		grayscaleImage = image.getMat();

	// We only support 8 bit images.
	if (grayscaleImage.type() != CV_8UC1)
		CV_Error(Error::StsUnsupportedFormat, "Blob detector only supports 8-bit images!");

	vector<vector<Center> > centers;
	for (double thresh = params.minThreshold; thresh < params.maxThreshold; thresh += params.thresholdStep)
	{
		// For each threshold value from the specified minimum to the specified maximum using the
		// specified step size, generate a binary image and find the centers of any blobs in it.
		Mat binarizedImage;
		threshold(grayscaleImage, binarizedImage, thresh, 255, THRESH_BINARY);
		vector<Center> curCenters;
		findBlobs(grayscaleImage, binarizedImage, curCenters, thresh);

		// Now that we have the centers of any blobs in the image, cancel out all the centers that
		// are within the minimum distance between blobs.
		vector<vector<Center> > newCenters;
		for (size_t i = 0; i < curCenters.size(); i++)
		{
			bool isNew = true;
			for (size_t j = 0; j < centers.size(); j++)
			{
				double dist = norm(centers[j][centers[j].size() / 2].location - curCenters[i].location);
				isNew = dist >= params.minDistBetweenBlobs && 
				        dist >= centers[j][centers[j].size() / 2].radius && 
				        dist >= curCenters[i].radius;
				if (!isNew)
				{
					centers[j].push_back(curCenters[i]);
					size_t k = centers[j].size() - 1;
					while (k > 0 && centers[j][k].radius < centers[j][k - 1].radius)
					{
						centers[j][k] = centers[j][k - 1];
						k--;
					}
					centers[j][k] = curCenters[i];
					break;
				}
			}
			if (isNew)
				newCenters.emplace_back(std::vector<Center>(1, curCenters[i]));
		}
		copy(newCenters.begin(), newCenters.end(), back_inserter(centers));
	}

	// Now convert the centers that were found into KeyPoints. Skip contours with
	// a repeat count lesser than the minimum repeatability.
	for (size_t i = 0; i < centers.size(); i++)
	{
		if (centers[i].size() < params.minRepeatability)
			continue;
		Point2d sumPoint(0, 0);
		double normalizer = 0;
		for (size_t j = 0; j < centers[i].size(); j++)
		{
			sumPoint += centers[i][j].confidence * centers[i][j].location;
			normalizer += centers[i][j].confidence;
		}
		sumPoint *= (1. / normalizer);
		KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius) * 2.0f);
		keypoints.push_back(kpt);
	}

	// Filter the KeyPoints by the supplied mask, if any.
	if (!mask.empty())
		KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
}

// ------------------------------------------------------------------------------------------------
// main()
// ------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	/* Needed to generate the file (saved for reference only). Please be aware that there is a
	   bug in OpenCV that prevents the closing element </opencv_storage> from being written.
	   Make sure you add it to the generated XML file manually, or reading the file will crash. */
	//FileStorage storage("../BlobDetectionParameters.xml", FileStorage::WRITE);
	//params.write(storage);

	// Read the blob detection parameters from the designated file.
	FileStorage storage("../BlobDetectionParameters.xml", FileStorage::READ);
	FileNode node = storage["opencv_storage"];
	params.read(node);

	// This program requires a filename as the first command line parameter. Read the
	// image from the specified file, if possible.
	if (argc < 2)
	{
		cout << "Usage: SimpleBlobDetector {filename} [lightOnDark]" << endl;
		exit(-1);
	}

	Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!image.data)
	{
		cout << "Could not open file " << argv[1] << endl;
		exit(-1);
	}

	// An optional second parameter specifies whether or not the image contains bright
	// objects on a dark background (default) or not.
	bool lightObjectsOnDarkBackground = true;
	if (argc > 2)
	{
		string bright = argv[2];		
		if (bright.compare("true") != 0 && bright.compare("false") != 0)
		{
			cout << "Usage: SimpleBlobDetector {filename} (true|false)" << endl;
			exit(-1);
		}
		if (bright.compare("false") == 0)
			lightObjectsOnDarkBackground = false;
	}	

	// Convert the image to a gray scale image and invert it (B/W) when necessary.
	// Then perform opening to facilitate detection of contours on adjacent objects.
	// Note that these steps are NOT part of SimpleBlobDetector's algorithms.
	cvtColor(image, image, CV_BGR2GRAY);
	if (!lightObjectsOnDarkBackground)
		bitwise_not(image, image);
	dilate(image, image, Mat::ones(5, 5, CV_32F));
	erode(image, image, Mat::ones(5, 5, CV_32F));

	// Show the original (converted to grayscale) and allow clicking. This shows the
	// actual gray value under the cursor, which may help in determining which threshold
	// values are appropriate.
	imshow("Original", image);
	setMouseCallback("Original", onMouseClick, &image);
	moveWindow("Original", 400, 0);

	// Find blobs in the image, using the blob detector parameters specified above.
	// Show the coordinates of all keypoints that were found.
	Mat mask, result;
	vector<KeyPoint> keypoints;
	detect(image, keypoints, mask);
	for (KeyPoint k : keypoints)
		cout << "(" << k.pt.x << "," << k.pt.y << ")" << endl;
	drawKeypoints(image, keypoints, result, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	ostringstream os;
	os << "Result: " << keypoints.size() << " keypoints.";
	imshow(os.str(), result);
	moveWindow(os.str(), 800, 0);

	// Wait for a key.
	waitKey(0);
	exit(0);
}

