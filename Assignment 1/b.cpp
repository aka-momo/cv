//
//  main.cpp
//  Sample
//
//  Created by Mohamed Diaa on 2015-06-09.
//  Copyright (c) 2015 swifters. All rights reserved.
//

#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void displayImage(Mat image, String windowName){
  namedWindow( windowName, WINDOW_AUTOSIZE );
  imshow(windowName, image );
}

Mat load_image(String imagePath){
  Mat image;
  image = imread(imagePath, CV_LOAD_IMAGE_COLOR);
  if(! image.data )
    cout <<  "Could not open or find the image" << std::endl ;
  return image;
}

int main(){
  Mat image;
  int histSize = 256, histH = 256, maxR = 0, maxB = 0, maxG = 0;
  std::map<int, int> redHash, greenHash, blueHash;
  Mat redHistImage( histH, histSize , CV_8UC3, Scalar( 255,255,255) );
  Mat blueHistImage( histH, histSize , CV_8UC3, Scalar( 255,255,255) );
  Mat greenHistImage( histH, histSize , CV_8UC3, Scalar( 255,255,255) );

  image = load_image("./lenna.png");

  displayImage(image, "Original Image");  

  for(int i = 0; i < image.rows; i++)
    for(int j = 0; j < image.cols; j++){
      Vec3b v = image.at<Vec3b>(i, j);
      redHash[v[2]]++;
      greenHash[v[1]]++;
      blueHash[v[0]]++;
    }
  
  for(int i = 0; i < histSize; i++ ){
    if(redHash[i] > maxR)
      maxR = redHash[i];
    if(greenHash[i] > maxG)
      maxG = greenHash[i];
    if(blueHash[i] > maxB)
      maxB = blueHash[i];
  }

  float consR = (float)histH/maxR;
  float consG = (float)histH/maxG;
  float consB = (float)histH/maxB;

  /// Draw for each channel
  for( int i = 0; i < histSize; i++ ){
    line(redHistImage, Point(i, histH), Point(i, histH - redHash[i]*consR), Scalar(0, 0, 255), 2, 8, 0);
    line(greenHistImage, Point( i, histH), Point(i, histH - greenHash[i]*consG), Scalar(0, 255, 0), 2, 8, 0);
    line(blueHistImage, Point( i, histH), Point( i, histH - blueHash[i]*consB), Scalar(255, 0, 0), 2, 8, 0);
  }

  displayImage(redHistImage, "Red Histogram");
  displayImage(blueHistImage, "Blue Histogram");
  displayImage(greenHistImage, "Green Histogram");

  waitKey(0);

  return 0;
}