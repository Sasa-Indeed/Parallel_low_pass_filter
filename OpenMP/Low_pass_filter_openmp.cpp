#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <omp.h>

std::string getImagePath();

int main() {
    std::string path;
    bool flag = false;
    int threadID;

    do {
        if (flag) {
            std::cout << "Incorrect path please check the path and try again!\n";
        }
        path = getImagePath();
        flag = true;
    } while (!cv::haveImageReader(path));

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Failed to load image.\n";
        return -1;
    }

    float kernelLength, kernelWidth, normalizationFactor;

    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
    std::cout << "Choose Kernel length: ";
    std::cin >> kernelLength;
    std::cout << "Choose Kernel width: ";
    std::cin >> kernelWidth;
    std::cout << "Choose the normalization factor (or 0 for default): ";
    std::cin >> normalizationFactor;

    if (normalizationFactor <= 0) {
        normalizationFactor = kernelLength * kernelWidth;
    }

    std::cout << "\nYour configurations are:\nImage path: " << path
        << "\nLength = " << kernelLength
        << "\nWidth = " << kernelWidth
        << "\nNormalization factor = " << normalizationFactor << "\n";

    cv::Mat kernel = cv::Mat::ones(kernelLength, kernelWidth, CV_32F) / normalizationFactor;

    cv::Mat imgdest = cv::Mat::zeros(img.size(), img.type());

    int totalRows = img.rows;

#pragma omp parallel private(threadID)
    {
        int numThreads = omp_get_num_threads();
        threadID = omp_get_thread_num();

        int rowsPerThread = totalRows / numThreads;
        int startRow = threadID * rowsPerThread;
        int endRow = (threadID == numThreads - 1) ? totalRows : startRow + rowsPerThread;


        int pad = static_cast<int>(kernelLength / 2);
        int safeStart = std::max(startRow - pad, 0);
        int safeEnd = std::min(endRow + pad, totalRows);

        cv::Mat subImage = img.rowRange(safeStart, safeEnd);
        cv::Mat subDest;


        cv::filter2D(subImage, subDest, CV_8U, kernel);

        int copyStart = startRow - safeStart;
        int copyLength = endRow - startRow;
        subDest.rowRange(copyStart, copyStart + copyLength).copyTo(imgdest.rowRange(startRow, endRow));
    }


    cv::imshow("Image Before", img);
    cv::imshow("Image After (Parallel Filter)", imgdest);
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
