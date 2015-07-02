#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <math.h>
#include <dirent.h>

using namespace cv;
using namespace std;

#define BASE_PATH "./data/"
#define POSITIVE_PATH BASE_PATH "positive/"
#define NEGATIVE_PATH BASE_PATH "negative/"
#define TEST_PATH BASE_PATH "test/"

void hist(Mat& img, Mat& hist);
void grayImage(Mat& img, Mat& gray_image);
void lbp(Mat& img, Mat& lbp_img);
void learn_face_detect();
void detect_face();
int countImages(char* path);

int main( int argc, char** argv )
{

    learn_face_detect();
    detect_face();
    return 0;
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

void hist(Mat& img, Mat& hist){

    hist.create(1, 16, CV_32FC1);

    Mat tmp = Mat::zeros(1, 16, CV_32FC1);

    int nRows = img.rows;
    int nCols = img.cols;

    // Loop over pixels
    int i,j;
    for( i = 1; i < nRows - 1; i++){

        for ( j = 1; j < nCols - 1; j++){

            int pix = img.ptr<int>(i)[j];
            int val = (int)floor((pix / 255.0) * 15);

            tmp.ptr<float>(0)[val] += 1;

        }
    }

    tmp.copyTo(hist);

}

void lbp(Mat& img, Mat& lbp_img){

    Mat tmp = Mat(img.size(), CV_32FC1);

    int nRows = img.rows;
    int nCols = img.cols;

    // Loop over pixels
    int i,j;
    for( i = 1; i < nRows - 1; i++){

        for ( j = 1; j < nCols - 1; j++){

            int center = img.ptr<int>(i)[j];
            int binaryValues[8];

            binaryValues[7] =  img.ptr<int>(i)[j - 1] > center ? 1 : 0;
            binaryValues[6] =  img.ptr<int>(i - 1)[j - 1] > center ? 1 : 0;
            binaryValues[5] =  img.ptr<int>(i - 1)[j] > center ? 1 : 0;
            binaryValues[4] =  img.ptr<int>(i - 1)[j + 1] > center ? 1 : 0;
            binaryValues[3] =  img.ptr<int>(i)[j + 1] > center ? 1 : 0;
            binaryValues[2] =  img.ptr<int>(i + 1)[j + 1] > center ? 1 : 0;
            binaryValues[1] =  img.ptr<int>(i + 1)[j] > center ? 1 : 0;
            binaryValues[0] =  img.ptr<int>(i + 1)[j - 1] > center ? 1 : 0;

            int newValue = 0;
            for (int k = 0; k < 8; k++){
                newValue = newValue * 2 + binaryValues[k];
            }

            tmp.ptr<int>(i)[j] = newValue;
        }
    }

    tmp.copyTo(lbp_img);
}

void learn_face_detect(){

    printf("preparing samples...\n");

    int i = 0;

    printf("negative\n");

    int samplesCount = countImages(NEGATIVE_PATH);
    samplesCount += countImages(POSITIVE_PATH);

    Mat samples(samplesCount, 16, CV_32FC1);
    Mat responses(samplesCount, 1, CV_32S);

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (NEGATIVE_PATH)) != NULL) {

      while ((ent = readdir (dir)) != NULL) {
            
        if(string(ent->d_name).find("jpg") != std::string::npos ||
            string(ent->d_name).find("png") != std::string::npos){

            Mat img = imread(string(NEGATIVE_PATH) + string(ent->d_name));
            Mat lbp_image, lbp_hist;
            grayImage(img, lbp_image);
            equalizeHist(lbp_image, lbp_image);
            lbp(lbp_image, lbp_image);
            hist(lbp_image, lbp_hist);

            lbp_hist.copyTo(samples.rowRange(i, i + 1));
            responses.ptr<int>(i)[0] = -1;

            // printf("%i\n", i);
            i++;
        }
      }
      
      closedir (dir);
    }

    printf("positive\n");

    if ((dir = opendir (POSITIVE_PATH)) != NULL) {

        while ((ent = readdir (dir)) != NULL) {
            
            if(string(ent->d_name).find("jpg") != std::string::npos ||
                string(ent->d_name).find("png") != std::string::npos){

                Mat img = imread(string(POSITIVE_PATH) + string(ent->d_name));
                Mat lbp_image, lbp_hist;
                grayImage(img, lbp_image);
                equalizeHist(lbp_image, lbp_image);
                lbp(lbp_image, lbp_image);
                hist(lbp_image, lbp_hist);

                lbp_hist.copyTo(samples.rowRange(i, i + 1));
                responses.ptr<int>(i)[0] = 1;

                // printf("%i\n", i);
                i++;

            }
        }
    }

    printf("learning\n");

    CvNormalBayesClassifier bayesClassifier(samples, responses);
    bayesClassifier.save("new_classifier");

}

void detect_face(){

    printf("detecting...\n");

    CvNormalBayesClassifier bayesClassifier;
    bayesClassifier.load("new_classifier");

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (TEST_PATH)) != NULL) {

      while ((ent = readdir (dir)) != NULL) {
            
        if(string(ent->d_name).find("jpg") != std::string::npos ||
            string(ent->d_name).find("png") != std::string::npos){

            Mat img = imread(string(TEST_PATH) + string(ent->d_name));
            Mat lbp_image, lbp_hist;
            grayImage(img, lbp_image);
            equalizeHist(lbp_image, lbp_image);
            lbp(lbp_image, lbp_image);
            hist(lbp_image, lbp_hist);

            float clazz = bayesClassifier.predict(lbp_hist);

            bool is_face = string(ent->d_name).find("not") == std::string::npos;

            char* result;
            if(is_face){

                if(clazz > 0)
                    result = "success";
                else
                    result = "-----";

            }else{

                if(clazz < 0)
                    result = "success";
                else
                    result = "-----";

            }

            printf("result: %s // face: %s // File Name: %s\n", result, is_face ? "true" : "false", ent->d_name);
        }
      }
      
      closedir (dir);
    }

}

int countImages(char* path){

    int count = 0;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (path)) != NULL) {

      while ((ent = readdir (dir)) != NULL) {
            
        if(string(ent->d_name).find("jpg") != std::string::npos ||
            string(ent->d_name).find("png") != std::string::npos){

            count++;
        }
      }
      
      closedir (dir);
    }

    return count;
}


