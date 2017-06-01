/*
ading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

namespace {
const char* about = "Basic marker detection";
const char* keys  =
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{c        |       | Camera intrinsic parameters. Needed for camera pose }"
        "{l        | 0.1   | Marker side lenght (in meters). Needed for correct scale in camera pose }"
        "{dp       |       | File of marker detector parameters }"
        "{r        |       | show rejected candidates too }";
}

/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}



/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["doCornerRefinement"] >> params->doCornerRefinement;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}

int angle(cv::Mat& image, cv::Point p0, cv::Point p1)
{
	  //float angleValue = angle2(cv::Point(p0.first, p0.second), cv::Point(p1.first, p1.second), cv::Point(p2.first, p2.second));
	  //angleValue = angleValue *180/M_PI;
	  //int angleValue = int(std::atan2((p0.second-p1.second),(p0.first-p1.first))*180/M_PI);
	  float angleValue = atan2((p1.y-p0.y), (p1.x-p0.x)) * 180 / M_PI;
	  //printf("Angle %d\n", (int)std::round(angleValue));
	  return (int)std::round(angleValue);
}

Point2f GetCenterId( int id, vector< int > &ids, vector< vector< Point2f > > &corners)
{
	int iIndexFound = -1;
	for (int index = 0; index < ids.size(); index++)
	{
		if (ids[index]==id)
		{
			iIndexFound = index;
		}
	}
	if (iIndexFound >= 0)
	{
		Point p1 = Point(corners[iIndexFound][0].x, corners[iIndexFound][0].y);
		Point p2 = Point(corners[iIndexFound][1].x, corners[iIndexFound][1].y);
		Point p3 = Point(corners[iIndexFound][2].x, corners[iIndexFound][2].y);
		Point p4 = Point(corners[iIndexFound][3].x, corners[iIndexFound][3].y);
		return Point2f((p1+p2+p3+p4)/4);
	}
	return Point2f(0,0);
}

void overlayImage(const cv::Mat &background, const cv::Mat &foreground,
  cv::Mat &output, cv::Point2i location)
{
  background.copyTo(output);


  // start at the row indicated by location, or at row 0 if location.y is negative.
  for(int y = std::max(location.y , 0); y < background.rows; ++y)
  {
    int fY = y - location.y; // because of the translation

    // we are done of we have processed all rows of the foreground image.
    if(fY >= foreground.rows)
      break;

    // start at the column indicated by location,

    // or at column 0 if location.x is negative.
    for(int x = std::max(location.x, 0); x < background.cols; ++x)
    {
      int fX = x - location.x; // because of the translation.

      // we are done with this row if the column is outside of the foreground image.
      if(fX >= foreground.cols)
        break;

      // determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
      double opacity =
        ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

        / 255.;


      // and now combine the background and foreground pixel, using the opacity,

      // but only if opacity > 0.
      for(int c = 0; opacity > 0 && c < output.channels(); ++c)
      {
        unsigned char foregroundPx =
          foreground.data[fY * foreground.step + fX * foreground.channels() + c];
        unsigned char backgroundPx =
          background.data[y * background.step + x * background.channels() + c];
        output.data[y*output.step + output.channels()*x + c] =
          backgroundPx * (1.-opacity) + foregroundPx * opacity;
      }
    }
  }
}


void loadAndRotateMask(cv::Point p500, cv::Point p501, cv::Point p502, int angle, cv::Mat& imageMaskRotated, double& offset)
{
	  cv::Mat imageMaskLoad = cv::imread("/home/patric/phact.png", CV_LOAD_IMAGE_UNCHANGED);
	  double width = cv::norm(p500-p502);
	  double height = cv::norm(p500-p501);
	  double diagonal = cv::norm(p501-p502);
	  offset = diagonal/2;

	  // Resize image so scale matches
	  cv::Size size(cv::norm(p500-p502), cv::norm(p500-p501));//the dst image size,e.g.100x100
	  cv::Mat imageMaskResized;//dst image
	  resize(imageMaskLoad,imageMaskResized,size);//resize image

	  // Create image to contain rotation
	  cv::Mat imageMaskEmptyContainer(cv::norm(p501-p502), cv::norm(p501-p502), CV_8UC4, cv::Scalar(0,0,0,0));


	  // Copy resized image to middle of rotation image
	  cv::Mat imageMask;
	  cv::Point2f pointMaskOffset = cv::Point2f((diagonal-width)/2, (diagonal-height)/2);
	  overlayImage(imageMaskEmptyContainer, imageMaskResized, imageMask, pointMaskOffset);

	  cv::Point2f pc(imageMask.cols/2., imageMask.rows/2.);
      cv::Mat r = cv::getRotationMatrix2D(pc, angle, 1.0);
	  cv::warpAffine(imageMask, imageMaskRotated, r, imageMask.size()); // what size I should use?
}

/**
 */
int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 2) {
        parser.printMessage();
        return 0;
    }

    int dictionaryId = parser.get<int>("d");
    bool showRejected = parser.has("r");
    bool estimatePose = parser.has("c");
    float markerLength = parser.get<float>("l");

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }
    detectorParams->doCornerRefinement = true; // do corner refinement in markers

    int camId = parser.get<int>("ci");

    String video;
    if(parser.has("v")) {
        video = parser.get<String>("v");
    }

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Mat camMatrix, distCoeffs;
    if(estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }

    VideoCapture inputVideo;
    int waitTime;
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    } else {
        inputVideo.open(camId);
        waitTime = 10;
    }

    double totalTime = 0;
    int totalIterations = 0;

    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)getTickCount();

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;
        vector< Vec3d > rvecs, tvecs;

        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

        // draw results
        image.copyTo(imageCopy);



        if(ids.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);

        }
        if (ids.size() >= 3)
        {
        	Point2f p500 = GetCenterId(0, ids, corners);
        	Point2f p501 = GetCenterId(1, ids, corners);
        	Point2f p502 = GetCenterId(2, ids, corners);
        	Point2f pCenter = Point2f((p501+p502)/2);

			int angleValue = angle(imageCopy, p500, p502);
			angleValue = (180 - angleValue)%360;
			//printf("Angle %d\n", angleValue);

			//cv::circle(image, pCenter, 4, cv::Scalar(255,0,0,0), 2);

			cv::Mat imageMaskRotated;
			cv::Point2f pointRotated;
			double offset;
			loadAndRotateMask(p500, p501, p502, angleValue, imageMaskRotated, offset);
			cv::Point2f offsetNew = cv::Point2f(pCenter.x-offset, pCenter.y-offset);

		    cv::Mat dst;
 		    overlayImage(imageCopy, imageMaskRotated, dst, offsetNew);
 		    imshow("out", dst); // OpenCV call

        }
        else
        {
        	imshow("out", imageCopy);
        }
        char key = (char)waitKey(waitTime);
        if(key == 27) break;
    }

    return 0;
}

