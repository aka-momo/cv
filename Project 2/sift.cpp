#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <math.h>
#include <dirent.h>

using namespace cv;
using namespace std;

#define IMAGES_PATH "./images/"

void refineMatchesWithHomography(vector<KeyPoint>& queryKeypoints,
    vector<KeyPoint>& trainKeypoints,
    vector<DMatch>& matches, Mat& homography){

        float reprojectionThreshold = 1.0;

        // Prepare data for cv::findHomography
            vector<Point2f> srcPoints(matches.size());
            vector<Point2f> dstPoints(matches.size());

        for (size_t i = 0; i < matches.size(); i++){
                srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
                dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
            }
        // Find homography matrix and get inliers mask
            vector<unsigned char> inliersMask(srcPoints.size());
            homography = findHomography(srcPoints,  dstPoints, CV_FM_RANSAC, reprojectionThreshold, inliersMask);
            vector<DMatch> inliers;
        for (size_t i=0; i<inliersMask.size(); i++)
            if (inliersMask[i])
                        inliers.push_back(matches[i]);
        matches.swap(inliers);
}

int main(int argc, char const *argv[]){

    initModule_nonfree();

    vector<Mat> images_descriptors;
    vector<vector<KeyPoint> > images_key_points;
    vector<string> images_names;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (IMAGES_PATH)) != NULL) {

        while ((ent = readdir (dir)) != NULL) {

            Mat img = imread(string(IMAGES_PATH) + string(ent->d_name));

            if(img.data){

                
                vector<KeyPoint> key_points;
                SiftFeatureDetector detector;
                detector.detect(img, key_points);

                images_key_points.push_back(key_points);

                Mat descriptors;
                SiftDescriptorExtractor extractor;
                extractor.compute(img, key_points, descriptors);

                images_descriptors.push_back(descriptors);
                images_names.push_back(string(ent->d_name));
            }

        }
      
      closedir (dir);
    }

    for(int i = 0; i < images_descriptors.size(); i++){

        Mat query_descriptor = images_descriptors[i];
        vector<KeyPoint> query_key_points = images_key_points[i];
        FlannBasedMatcher matcher;

        vector<Mat> descriptors_vector;
        vector<vector<KeyPoint> > train_key_points_vectors;
        vector<string> descriptors_images_names_vector;

        for(int j = 0; j < images_descriptors.size(); j++){

            if(i != j){

                Mat train_descriptor = images_descriptors[j];
                vector<KeyPoint> train_key_points_vector = images_key_points[j];
                string train_descriptor_image_name = images_names[j];

                descriptors_vector.push_back(train_descriptor);
                descriptors_images_names_vector.push_back(train_descriptor_image_name);
                train_key_points_vectors.push_back(train_key_points_vector);

            }

        }


        matcher.add(descriptors_vector);

        vector<DMatch> matches_vector;
        matcher.match(query_descriptor, matches_vector);

        map<int, int> matches_counts;

        for(int k = 0; k < descriptors_vector.size(); k++)
            matches_counts[k] = 0;

        for (int k = 0; k < matches_vector.size(); k++){
            
            DMatch match = matches_vector[k];

            matches_counts[match.imgIdx] += 1;

        }

        int max_count = 0;
        int max_count_idx = 0;
        map<int, int>::iterator it_begin, it_end;
        for (it_begin = matches_counts.begin(), it_end = matches_counts.end(); 
            it_begin != it_end; it_begin++){

            if(it_begin->second > max_count){

                max_count = it_begin->second;
                max_count_idx = it_begin->first;
            }

        }

        // cout << "Best match for: " << images_names[i] << " is: " << descriptors_images_names_vector[max_count_idx] << "\n-------------\n";


       
        // good matches

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int k = 0; k < matches_vector.size(); k++ ){

            DMatch match = matches_vector[k];

            if(match.imgIdx == max_count_idx){

                double dist = match.distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            } 
        }

        std::vector< DMatch > good_matches;

        for( int k = 0; k < matches_vector.size(); k++ ){ 
            
            DMatch match = matches_vector[k];

            if(match.imgIdx == max_count_idx){

                if( match.distance < 3 * min_dist ){
                    
                    good_matches.push_back( match); 
                }

            }
        }


        Mat homography;
        refineMatchesWithHomography(query_key_points,train_key_points_vectors[max_count_idx],good_matches,homography);

         cout << "\n------------------------________-------------------------\n";
        for( int k = 0; k < good_matches.size(); k++ ){
            
            cout << "Best match for: " << images_names[i] << " is: " << descriptors_images_names_vector[good_matches[k].imgIdx] << "\n-------------\n";
        }

    }


    return 0;
}

