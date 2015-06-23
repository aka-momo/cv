#include <iostream>
#include "corner_harris.hpp"

using namespace std;

int main( int argc, char** argv )
{

    Mat image;
    image = imread("./test.png", CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    // OpenCV implementation to benchmark against 
    opencvHarris(image);
    
    Mat gray_image; 
    grayImage(image, gray_image);

    Mat dst, dst_norm, dst_norm_scaled;
    harris(gray_image, dst, 2, 0.04);

    /// Normalizing
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    
    int thresh = 200;

    /// Drawing a circle around corners
    for( int j = 0; j < dst_norm.rows ; j++ )
     { for( int i = 0; i < dst_norm.cols; i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > thresh )
              {
               circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              }
          }
     }


    /// Showing the result
    namedWindow( "Our Harris", CV_WINDOW_AUTOSIZE );
    imshow( "Our Harris", dst_norm_scaled );

    waitKey(0);

    return 0;
}
