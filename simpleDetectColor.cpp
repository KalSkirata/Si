#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv ){
	VideoCapture cap(0); //capture the video from web cam

	if(!cap.isOpened()){
		cout << "Cannot open the web cam" << endl;
		return -1;
	}

	int lowH = 11;
	int highH = 29;

	int lowS = 90; 
	int highS = 255;

	int lowV = 71;
	int highV = 255;

	Mat imgOriginal, imgHSV, imgThresholded;

	while(1){
		/* *cap >> src;
		cvtColor(src,hsv,CV_BGR2HSV);
		inRange(hsv,Scalar(lowH,lowS,lowV),Scalar(highH,highS,highV),threshold);
		imshow("thr",threshold);*/

      if (!cap.read(imgOriginal)){
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		inRange(imgHSV, Scalar(lowH, lowS, lowV), Scalar(highH, highS, highV), imgThresholded); //Threshold the image

		//imshow("Thresholded Image", imgThresholded);

		
		/*std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(imgThresholded, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
		std::vector<cv::Rect> boundRect( contours.size() );
		for( int i = 0; i < contours.size(); i++ ){ 
			cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
			boundRect[i] = cv::boundingRect( cv::Mat(contours_poly[i]) );
		}
		*/


		/*Mat threshold_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		int thresh = 100;
		RNG rng(12345);

		/// Detect edges using Threshold
		threshold( imgThresholded, threshold_output, thresh, 255, THRESH_BINARY );
		/// Find contours
		findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

		vector<vector<Point> > contours_poly( contours.size() );
 		vector<Rect> boundRect( contours.size() );

		for( int i = 0; i < contours.size(); i++ ){
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			boundRect[i] = boundingRect( Mat(contours_poly[i]) );
		}

		Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  		for( int i = 0; i< contours.size(); i++ ){
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
			rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
		}

		imshow( "Contours", drawing );*/


		imshow("Thresholded Image", imgThresholded);
		//imshow("HSV", imgHSV);
		//imshow("Original", imgOriginal);

		if(waitKey(30) == 27){
			cout << "esc key is pressed by user" << endl;
			break; 
		}
	}	

	return 0;
}
