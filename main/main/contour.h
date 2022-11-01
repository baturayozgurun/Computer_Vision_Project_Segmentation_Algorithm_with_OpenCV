//  Developer: Baturay Özgürün
//  Copyright © 2018 Baturay Özgürün
//  If you have any requests or questions please do not hesitate to contact me via baturay.ozgurun@gmail.com

//  Project Desciption: This code presents countours of segments with bounding boxes by capitalizing on the Canny Edge Detection method.

//Include Headers
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

//Include Namespaces
using namespace cv;
using namespace std;

RNG rng(12345);

void contour(){
    Mat Image, SegmentedImage,BinaryImage, SegmentedImageAppearance; //Define Mat variables
    int Ns; //Define integer variable for the total number of segments
    
    Image=imread("data/input.jpg",CV_LOAD_IMAGE_GRAYSCALE); //Read the image in grayscale
    threshold(Image, BinaryImage, 60, 255, THRESH_BINARY); //Convert the grayscale image to binary image
    //connectedComponents calculates segments of the given binary image (BinaryImage) with 8  connectedness
    Ns = connectedComponents(BinaryImage, SegmentedImage, 8, CV_32S); //It also counts total number of segments (Ns)
    
    Mat list=Mat::zeros(Ns,1,CV_32SC1); //Create list variable to keep the number of segments
    for (int k=0;k<Ns;k++){
        int tut=0;
        for (int j=0; j<SegmentedImage.cols; j++){
            for (int i=0; i<SegmentedImage.rows;i++){
                if(SegmentedImage.at<int>(i,j)==k){
                    tut=tut+1;
                }
            }
        }
        list.at<int>(k,0)=tut; //Write the number of segments. Row number indicates the segment numbers.
    }
    
    //Apply threshold to the segments to remove from the image using segment numbers (kept in list variable)
    int t=round(0.00001 *SegmentedImage.cols*SegmentedImage.rows); //Threshold Value
    for (int k=0;k<Ns;k++){
        if(list.at<int>(k,0)<t){ //When the value in the list is less than the threshold
            for (int j=0; j<SegmentedImage.cols; j++){ //Scan across column
                for (int i=0; i<SegmentedImage.rows;i++){ //Scan across row
                    if(SegmentedImage.at<int>(i,j)==k){ //When the pixel is equal to the conditioned k value
                        SegmentedImage.at<int>(i,j)=0; //Make that pixel zero
                    }
                }
            }
        }
    }
    
    cout << t << endl;
    cout << Ns << endl;
    
    normalize(SegmentedImage,SegmentedImageAppearance,0,255,NORM_MINMAX,CV_8U); //Seperate calculated segments in 8 bit scale
    applyColorMap(SegmentedImageAppearance,SegmentedImageAppearance,COLORMAP_JET); //Make colorful representation of the segmented image.
    
    //Define variables for the contour calculation.
    Mat EdgeImage;
    vector<vector<Point> > Contours;
    vector<Vec4i> Hierarchy;
    
    Canny(SegmentedImageAppearance,EdgeImage,50,200,3); //Detect the image edges using Canny method
    findContours(EdgeImage,Contours,Hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0)); //Evaulate contours of the image.
    
    //Define variables for the moment and mass calculations of the segments.
    vector<Moments> mu(Contours.size());
    vector<Point2f> mc(Contours.size());
    vector<vector<Point>> ContoursPoly(Contours.size());
    vector<Rect> BoundSquare(Contours.size());
    Mat Shape=Mat::zeros(EdgeImage.size(),CV_8UC3);
    
    //Get the mass centers
    for(int i=0;i<Contours.size();i++){
        Scalar Color=Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)); //Define color
        mu[i]=moments(Contours[i],false); //Get the moments of the image
        mc[i]=Point2f(static_cast<float>(mu[i].m10/mu[i].m00),static_cast<float>(mu[i].m01/mu[i].m00)); //Get the mass center
        approxPolyDP(Contours[i],ContoursPoly[i],3,true); //Aproximate a polygonal curve
        BoundSquare[i] = boundingRect(ContoursPoly[i]); //Define segment boundaries
        rectangle(Shape,BoundSquare[i].tl(),BoundSquare[i].br(),Color,2,8,0); //Draw rectangle shape for the segments
    }
    
    imshow("Segmentation Boundaries", EdgeImage);
    imwrite("results/Segmentation Boundaries.jpg",EdgeImage); //Write the segmentation boundaries.
    imshow("Segments with Bounding Boxes",Shape); //Show the drawn rectangle thresholded segments.
    imwrite("results/Segments with Bounding Boxes.jpg",Shape); //Write the drawn-rectangle-thresholded segments.
    waitKey(1000); //wait to show images
}

