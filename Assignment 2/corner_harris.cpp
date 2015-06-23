#include "corner_harris.hpp"

static int sobel_x_ar[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
static int sobel_y_ar[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

float gaussian(Point p, Point center, float sd){

    return ((1.0f / (2.0f * M_PI * pow(sd, 2))) * exp(-(pow(p.x - center.x, 2) + pow(p.y - center.y, 2)) / (2.0 * pow(sd, 2))));
}

void sobel(Mat &img, Mat& deriv, uint axis){

    // Initialize sobel operators
    Mat sobel_x = Mat(3, 3, CV_32S, &sobel_x_ar);
    Mat sobel_y = Mat(3, 3, CV_32S, &sobel_y_ar);

    // Convert to floating point
    sobel_x.convertTo(sobel_x, CV_32FC1);
    sobel_y.convertTo(sobel_y, CV_32FC1);

    // Ensure size & type
    deriv.create( img.size(), CV_32FC1);

    int nRows = img.rows;
    int nCols = img.cols;

    // Loop over pixels
    int i,j;
    for( i = 0; i < nRows; i++)
    {

        for ( j = 0; j < nCols; j++)
        {

            // Enforce matrices bounds 

            int slice_cols_end = j + 3;
            int slice_rows_end = i + 3;
            
            if(slice_cols_end > nCols){
                slice_cols_end = nCols;
            }
            
            if(slice_rows_end > nRows){
                slice_rows_end = nRows;
            }

            Mat sub_img = Mat::zeros(3, 3, CV_32FC1);

            // Prepare sub-matrix to be convolved
            Mat tmp = img.rowRange(i, slice_rows_end).colRange(j, slice_cols_end);

            int sub_img_rows = slice_rows_end - i;
            int sub_img_cols = slice_cols_end - j;

            tmp.copyTo(sub_img.rowRange(0, sub_img_rows).colRange(0, sub_img_cols));

            // When at image borders, we expand submatrix to (3 x 3) by copying the nearest pixel values
            for (int sub_img_row = sub_img_rows; sub_img_row < 3; sub_img_row++)
            {

                tmp = sub_img.rowRange(sub_img_row - 1, sub_img_row).colRange(Range::all());   
                tmp.copyTo(sub_img.rowRange(sub_img_row, sub_img_row + 1).colRange(Range::all()));
                
            }

            for (int sub_img_col = sub_img_cols; sub_img_col < 3; sub_img_col++)
            {

                tmp = sub_img.colRange(sub_img_col - 1, sub_img_col).rowRange(Range::all());
                tmp.copyTo(sub_img.colRange(sub_img_col, sub_img_col + 1).rowRange(Range::all()));
            }

            sub_img.convertTo(sub_img, CV_32FC1);

            // Convolve

            int conv;
            if(axis == X_AXIS){

                conv = sub_img.dot(sobel_x);
            }else if(axis == Y_AXIS){

                conv = sub_img.dot(sobel_y);
            }

            float *p = deriv.ptr<float>(i);
            p[j] = conv;
        }
    }

}

void harris(Mat &img, Mat& harris, int block_size, float k){
    
    // Ensure size & type
    harris.create(img.size(), CV_32FC1);

    // Calculate derivatives
    Mat x_deriv, y_deriv;
    sobel(img, x_deriv, X_AXIS);
    sobel(img, y_deriv, Y_AXIS);

    // Loop over pixels

    int nRows = img.rows;
    int nCols = img.cols;

    int i,j;
    for( i = 0; i < nRows; ++i)
    {

        for ( j = 0; j < nCols; ++j)
        {

            // initialize and ensure block bounds

            int block_cols_end = j + block_size;
            int block_rows_end = i + block_size;
            
            if(block_cols_end > nCols){
                block_cols_end = nCols;
            }
            
            if(block_rows_end > nRows){
                block_rows_end = nRows;
            }

            // Calculate Gaussian center

            float mu_x = j + (block_size / 2.0f);
            float mu_y = i + (block_size / 2.0f);

            // Calculate covariance matrix at this window 

            Mat m = Mat::zeros(2, 2, CV_32FC1);
            for (int block_row = i; block_row < block_rows_end; block_row++)
            {
                
                for (int block_column = j; block_column < block_cols_end; block_column++)
                {
                    
                    float w = gaussian(Point(block_column, block_row), Point(mu_x, mu_y), 1);
                    
                    Mat m_tmp = Mat::zeros(2, 2, CV_32FC1);

                    float img_x = x_deriv.ptr<float>(block_row)[block_column];
                    float img_y = y_deriv.ptr<float>(block_row)[block_column];

                    m_tmp.ptr<float>(0)[0] = pow(img_x, 2);
                    m_tmp.ptr<float>(0)[1] = img_x * img_y;
                    m_tmp.ptr<float>(1)[0] = img_x * img_y;
                    m_tmp.ptr<float>(1)[1] = pow(img_y, 2);

                    m_tmp *= w;

                    m += m_tmp;
                }
            }

            // R-Score

            float r = determinant(m) - (k * pow(sum(trace(m))[0], 2));

            harris.ptr<float>(i)[j] = r;

        }
    }

}

void opencvHarris(Mat &img){
    int thresh = 200;
    int max_thresh = 255;

    Mat src_gray;
    grayImage(img, src_gray);

    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( img.size(), CV_32FC1 );

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

    /// Normalizing
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

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
    namedWindow( "OpenCV Harris", CV_WINDOW_AUTOSIZE );
    imshow( "OpenCV Harris", dst_norm_scaled );

}

void grayImage(Mat &img, Mat& gray_image){

    // Ensure size & type
    gray_image.create(img.rows, img.cols, CV_8U);


    // Loop and average

    MatIterator_<Vec3b> image_it, end;
    MatIterator_<uchar> gray_image_it = gray_image.begin<uchar>();
    for( image_it = img.begin<Vec3b>(), end = img.end<Vec3b>(); image_it != end; ++image_it, ++gray_image_it)
    {
        *gray_image_it = ((*image_it)[0] + (*image_it)[1] + (*image_it)[2]) / 3;
    }

}