#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <iostream>
#include <algorithm>


std::string getImagePath();

int main() {

    std::string path;
    bool flag = false;

    do {
        if (flag) {
            std::cout << "Incorrect path please check the path and try again!\n";
        }
        path = getImagePath();
        flag = true;
    } while (!cv::haveImageReader(path));
    

    cv::Mat img = cv::imread(path), imgdest;

    float kernelLength, kernelWidth , normalizationFactor;

    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
    std::cout << "Choose Kernel length: ";
    std::cin >> kernelLength;
    std::cout << "Choose Kernel width: ";
    std::cin >> kernelWidth;
    std::cout << "Choose the normalization factor for the default value enter 0: ";
    std::cin >> normalizationFactor;
    
    if (normalizationFactor <= 0) {
        normalizationFactor = kernelLength * kernelWidth;
    }

    std::cout << "\nYour configurations are:\nImage path: " << path
        <<"\nLength = " << kernelLength
        << "\nWidth = " <<  kernelWidth
        << "\nNormalization factor = " << normalizationFactor;
    
    cv::Mat kernel = cv::Mat::ones(kernelLength, kernelWidth, CV_32F) / normalizationFactor;
    
    cv::filter2D(img, imgdest, CV_8U, kernel);

    
    cv::imshow("Image Before", img);
    cv::imshow("Image After", imgdest);
    cv::moveWindow("Image Before", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

std::string getImagePath() {
    std::string path;

    std::cout << "Enter image path: ";
    std::getline(std::cin, path);

    std::replace(path.begin(), path.end(), '\\', '/');

    return path;
}