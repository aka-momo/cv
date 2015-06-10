//
//  main.cpp
//  Sample
//
//  Created by Mohamed Diaa on 2015-06-09.
//  Copyright (c) 2015 swifters. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void displayImage(Mat image){
  namedWindow( "Display window", WINDOW_AUTOSIZE );
  imshow("Display window", image );
}

Mat load_image(String imagePath){
  Mat image;
  image = imread(imagePath, CV_LOAD_IMAGE_COLOR);
  if(! image.data )
    cout <<  "Could not open or find the image" << std::endl ;
  return image;
}

Mat grayScale(Mat image){
  Mat newImage(image.size(), CV_8U);
  for(int i = 0; i < image.rows; i++)
    for(int j = 0; j < image.cols; j++){
      Vec3b v = image.at<Vec3b>(i, j);
      newImage.at<uchar> (i, j) = (v[0] + v[1] + v[2])/3;
    }
  return newImage;
}

int main(){   
  Mat image = load_image("./lenna.png");
  Mat newImage = grayScale(image);
  displayImage(newImage);
  waitKey(0);
  return 0;
}