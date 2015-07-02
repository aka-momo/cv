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

#define IMAGES_PATH "/Users/mohammedamer/Python Workspace/cv/Project 2/images/"

Size size_aspect(Size src, Size target);

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

                // resize(img, img, size_aspect(img.size(), Size(256, 256)));
                
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

        cout << "Best match for: " << images_names[i] << " is: " << descriptors_images_names_vector[max_count_idx] << "\n-------------\n";

        Mat object_img = imread(string(IMAGES_PATH) + images_names[i]);
        Mat scene_img = imread(string(IMAGES_PATH) + descriptors_images_names_vector[max_count_idx]);
        
        // resize(object_img, object_img, size_aspect(object_img.size(), Size(256, 256)));
        // resize(scene_img, scene_img, size_aspect(scene_img.size(), Size(256, 256)));

       
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

        Mat img_matches;
        drawMatches( object_img, query_key_points, scene_img, train_key_points_vectors[max_count_idx],
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


        //-- Localize the object
        vector<Point2f> obj;
        vector<Point2f> scene;

        for( int k = 0; k < good_matches.size(); k++ ){

            DMatch good_match = good_matches[k];
            
            obj.push_back( query_key_points[ good_match.queryIdx ].pt );
            scene.push_back( train_key_points_vectors[ good_match.imgIdx ][good_match.trainIdx].pt );
        }

        Mat H = findHomography( obj, scene, CV_RANSAC);

        //-- Get the corners from the image_1 ( the object to be "detected" )
        vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( object_img.cols, 0 );
        obj_corners[2] = cvPoint( object_img.cols, object_img.rows ); obj_corners[3] = cvPoint( 0, object_img.rows );
        std::vector<Point2f> scene_corners(4);

        perspectiveTransform( obj_corners, scene_corners, H);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f( object_img.cols, 0), scene_corners[1] + Point2f( object_img.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f( object_img.cols, 0), scene_corners[2] + Point2f( object_img.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f( object_img.cols, 0), scene_corners[3] + Point2f( object_img.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f( object_img.cols, 0), scene_corners[0] + Point2f( object_img.cols, 0), Scalar( 0, 255, 0), 4 );

        imshow( images_names[i].c_str(), img_matches );

        waitKey(0);
    }


    return 0;
}

Size size_aspect(Size src, Size target){

    Size new_size;

    float ar = float(src.width) / float(src.height);

    if(src.width >= src.height){

        new_size.width = target.width;
        new_size.height = (1 / ar) * new_size.width;

    }else{

        new_size.height = target.height;
        new_size.width = ar * new_size.height;

    }
    
    return new_size;
}

