//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/opencv.hpp>
//#include <string>
//#include <iostream>
//#include <algorithm>
//#include <omp.h>
//
//std::string getImagePath();
//cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel);
//
//int main() {
//    std::string path;
//    bool flag = false;
//    int threadID;
//    std::string grayFlag = "y";
//
//    do {
//        if (flag) {
//            std::cout << "Incorrect path please check the path and try again!\n";
//        }
//        path = getImagePath();
//        flag = true;
//    } while (!cv::haveImageReader(path));
//
//    cv::Mat img = cv::imread(path);
//    if (img.empty()) {
//        std::cerr << "Failed to load image.\n";
//        return -1;
//    }
//
//    std::cout << "Do want the image in gray scale?[y/n]: y\n";
//    std::cin >> grayFlag;
//
//    if (grayFlag[0] == 'y') {
//        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
//    }
//
//    float kernelLength, kernelWidth, normalizationFactor;
//
//    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
//    std::cout << "Choose Kernel length: ";
//    std::cin >> kernelLength;
//    std::cout << "Choose Kernel width: ";
//    std::cin >> kernelWidth;
//    std::cout << "Choose the normalization factor (or 0 for default): ";
//    std::cin >> normalizationFactor;
//
//    if (normalizationFactor <= 0) {
//        normalizationFactor = kernelLength * kernelWidth;
//    }
//
//    std::cout << "\nYour configurations are:\nImage path: " << path
//        << "\nLength = " << kernelLength
//        << "\nWidth = " << kernelWidth
//        << "\nNormalization factor = " << normalizationFactor << "\n";
//
//    double start_time = omp_get_wtime();
//
//    cv::Mat kernel = cv::Mat::ones(kernelLength, kernelWidth, CV_32F) / normalizationFactor;
//
//    cv::Mat imgdest = cv::Mat::zeros(img.size(), img.type());
//
//    int totalRows = img.rows;
//
//#pragma omp parallel private(threadID)
//    {
//        int numThreads = omp_get_num_threads();
//        threadID = omp_get_thread_num();
//
//        int rowsPerThread = totalRows / numThreads;
//        int startRow = threadID * rowsPerThread;
//        int endRow = (threadID == numThreads - 1) ? totalRows : startRow + rowsPerThread;
//
//
//        int pad = static_cast<int>(kernelLength / 2);
//        int safeStart = std::max(startRow - pad, 0);
//        int safeEnd = std::min(endRow + pad, totalRows);
//
//        cv::Mat subImage = img.rowRange(safeStart, safeEnd);
//        cv::Mat subDest;
//
//
//        subDest = applyLowPassFilter(subImage, kernel);
//
//        int copyStart = startRow - safeStart;
//        int copyLength = endRow - startRow;
//        subDest.rowRange(copyStart, copyStart + copyLength).copyTo(imgdest.rowRange(startRow, endRow));
//    }
//
//    double end_time = omp_get_wtime();
//    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
//
//    cv::imshow("Image Before", img);
//    cv::imshow("Image After (Parallel Filter)", imgdest);
//    cv::moveWindow("Image Before", 0, 45);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//
//    return 0;
//}
//
//std::string getImagePath() {
//    std::string path;
//    std::cout << "Enter image path: ";
//    std::getline(std::cin, path);
//    std::replace(path.begin(), path.end(), '\\', '/');
//    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());
//    return path;
//}
//
///**
// * Apply a low pass filter to an image using a custom convolution implementation
// * @param src The source image
// * @param kernel The convolution kernel
// * @return The filtered image
// **/
//cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel) {
//    double start_time = omp_get_wtime();
//
//    /* Create output image with same size and type as input */
//    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
//    /* Identifies how many color channels the image has (1 for grayscale, 3 for RGB) */
//    int channels = src.channels();
//    /* Calculate padding */
//    int padH = kernel.rows / 2;
//    int padW = kernel.cols / 2;
//    /* Create padded version of source image */
//    cv::Mat padded;
//    cv::copyMakeBorder(src, padded, padH, padH, padW, padW, cv::BORDER_REPLICATE);
//
//    /* Cache kernel values to avoid repeated access */
//    std::vector<float> kernelValues(kernel.rows * kernel.cols);
//    for (int ky = 0; ky < kernel.rows; ky++) {
//        for (int kx = 0; kx < kernel.cols; kx++) {
//            kernelValues[ky * kernel.cols + kx] = kernel.at<float>(ky, kx);
//        }
//    }
//
//    /* Apply convolution for each pixel in the image using OpenMP */
//    if (channels == 1) {
//        /* Grayscale processing */
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < src.rows; y++) {
//            for (int x = 0; x < src.cols; x++) {
//                float sum = 0.0;
//                /* Apply kernel */
//                for (int ky = 0; ky < kernel.rows; ky++) {
//                    for (int kx = 0; kx < kernel.cols; kx++) {
//                        /* Calculate position in padded image */
//                        int pixelY = y + ky;
//                        int pixelX = x + kx;
//                        sum += static_cast<float>(padded.at<uchar>(pixelY, pixelX)) *
//                            kernelValues[ky * kernel.cols + kx];
//                    }
//                }
//                /* Update output pixel */
//                dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
//            }
//        }
//    }
//    else {
//        /* Color image processing */
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < src.rows; y++) {
//            for (int x = 0; x < src.cols; x++) {
//                float sum[3] = { 0.0, 0.0, 0.0 };
//                /* Apply kernel */
//                for (int ky = 0; ky < kernel.rows; ky++) {
//                    for (int kx = 0; kx < kernel.cols; kx++) {
//                        /* Calculate position in padded image */
//                        int pixelY = y + ky;
//                        int pixelX = x + kx;
//                        float kernelVal = kernelValues[ky * kernel.cols + kx];
//                        cv::Vec3b pixel = padded.at<cv::Vec3b>(pixelY, pixelX);
//
//                        /* Process all channels together for better cache efficiency */
//                        sum[0] += static_cast<float>(pixel[0]) * kernelVal;
//                        sum[1] += static_cast<float>(pixel[1]) * kernelVal;
//                        sum[2] += static_cast<float>(pixel[2]) * kernelVal;
//                    }
//                }
//                /* Update output pixel */
//                cv::Vec3b& outPixel = dst.at<cv::Vec3b>(y, x);
//                outPixel[0] = cv::saturate_cast<uchar>(sum[0]);
//                outPixel[1] = cv::saturate_cast<uchar>(sum[1]);
//                outPixel[2] = cv::saturate_cast<uchar>(sum[2]);
//            }
//        }
//    }
//
//    double end_time = omp_get_wtime();
//    std::cout << "Filter function execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
//
//    return dst;
//}

//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <string>
//#include <iostream>
//#include <algorithm>
//#include <omp.h>
//#include <vector>
//
//std::string getImagePath();
//cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel);
//cv::Mat applyLowPassFilterOptimized(const cv::Mat& src, const cv::Mat& kernel);
//
//int main() {
//    std::string path;
//    bool flag = false;
//    std::string grayFlag = "y";
//    int threadID;
//
//    do {
//        if (flag) {
//            std::cout << "Incorrect path please check the path and try again!\n";
//        }
//        path = getImagePath();
//        flag = true;
//    } while (!cv::haveImageReader(path));
//
//    cv::Mat img = cv::imread(path);
//    if (img.empty()) {
//        std::cerr << "Failed to load image.\n";
//        return -1;
//    }
//
//    std::cout << "Do want the image in gray scale?[y/n]: y\n";
//    std::cin >> grayFlag;
//
//    if (grayFlag[0] == 'y') {
//        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
//    }
//
//    float kernelLength, kernelWidth, normalizationFactor;
//
//    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
//    std::cout << "Choose Kernel length: ";
//    std::cin >> kernelLength;
//    std::cout << "Choose Kernel width: ";
//    std::cin >> kernelWidth;
//    std::cout << "Choose the normalization factor (or 0 for default): ";
//    std::cin >> normalizationFactor;
//
//    if (normalizationFactor <= 0) {
//        normalizationFactor = kernelLength * kernelWidth;
//    }
//
//    std::cout << "\nYour configurations are:\nImage path: " << path
//              << "\nLength = " << kernelLength
//              << "\nWidth = " << kernelWidth
//              << "\nNormalization factor = " << normalizationFactor << "\n";
//
//    cv::Mat kernel = cv::Mat::ones(kernelLength, kernelWidth, CV_32F) / normalizationFactor;
//
//    // Original implementation (for comparison)
//    double start_time_orig = omp_get_wtime();
//    cv::Mat imgdest_orig = applyLowPassFilter(img, kernel);
//    double end_time_orig = omp_get_wtime();
//    std::cout << "Original filter execution time: " << (end_time_orig - start_time_orig) * 1000.0 << " milliseconds" << std::endl;
//
//    // Optimized implementation
//    double start_time = omp_get_wtime();
//    cv::Mat imgdest = applyLowPassFilterOptimized(img, kernel);
//    double end_time = omp_get_wtime();
//    std::cout << "Optimized filter execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
//    std::cout << "Speedup: " << (end_time_orig - start_time_orig) / (end_time - start_time) << "x" << std::endl;
//
//    // Show all images for comparison
//    cv::imshow("Original Image", img);
//    cv::imshow("Original Implementation", imgdest_orig);
//    cv::imshow("Optimized Implementation", imgdest);
//    cv::moveWindow("Original Image", 0, 45);
//    cv::moveWindow("Original Implementation", 0, img.rows + 90);
//    cv::moveWindow("Optimized Implementation", img.cols + 30, 45);
//
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//
//    return 0;
//}
//
//std::string getImagePath() {
//    std::string path;
//    std::cout << "Enter image path: ";
//    std::getline(std::cin, path);
//    std::replace(path.begin(), path.end(), '\\', '/');
//    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());
//    return path;
//}
//
///**
// * Original low pass filter implementation (for comparison)
// */
//cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel) {
//    double start_time = omp_get_wtime();
//
//    /* Create output image with same size and type as input */
//    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
//    /* Identifies how many color channels the image has (1 for grayscale, 3 for RGB) */
//    int channels = src.channels();
//    /* Calculate padding */
//    int padH = kernel.rows / 2;
//    int padW = kernel.cols / 2;
//    /* Create padded version of source image */
//    cv::Mat padded;
//    cv::copyMakeBorder(src, padded, padH, padH, padW, padW, cv::BORDER_REPLICATE);
//
//    /* Cache kernel values to avoid repeated access */
//    std::vector<float> kernelValues(kernel.rows * kernel.cols);
//    for (int ky = 0; ky < kernel.rows; ky++) {
//        for (int kx = 0; kx < kernel.cols; kx++) {
//            kernelValues[ky * kernel.cols + kx] = kernel.at<float>(ky, kx);
//        }
//    }
//
//    /* Apply convolution for each pixel in the image using OpenMP */
//    if (channels == 1) {
//        /* Grayscale processing */
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < src.rows; y++) {
//            for (int x = 0; x < src.cols; x++) {
//                float sum = 0.0;
//                /* Apply kernel */
//                for (int ky = 0; ky < kernel.rows; ky++) {
//                    for (int kx = 0; kx < kernel.cols; kx++) {
//                        /* Calculate position in padded image */
//                        int pixelY = y + ky;
//                        int pixelX = x + kx;
//                        sum += static_cast<float>(padded.at<uchar>(pixelY, pixelX)) *
//                               kernelValues[ky * kernel.cols + kx];
//                    }
//                }
//                /* Update output pixel */
//                dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
//            }
//        }
//    }
//    else {
//        /* Color image processing */
//#pragma omp parallel for collapse(2)
//        for (int y = 0; y < src.rows; y++) {
//            for (int x = 0; x < src.cols; x++) {
//                float sum[3] = { 0.0, 0.0, 0.0 };
//                /* Apply kernel */
//                for (int ky = 0; ky < kernel.rows; ky++) {
//                    for (int kx = 0; kx < kernel.cols; kx++) {
//                        /* Calculate position in padded image */
//                        int pixelY = y + ky;
//                        int pixelX = x + kx;
//                        float kernelVal = kernelValues[ky * kernel.cols + kx];
//                        cv::Vec3b pixel = padded.at<cv::Vec3b>(pixelY, pixelX);
//
//                        /* Process all channels together for better cache efficiency */
//                        sum[0] += static_cast<float>(pixel[0]) * kernelVal;
//                        sum[1] += static_cast<float>(pixel[1]) * kernelVal;
//                        sum[2] += static_cast<float>(pixel[2]) * kernelVal;
//                    }
//                }
//                /* Update output pixel */
//                cv::Vec3b& outPixel = dst.at<cv::Vec3b>(y, x);
//                outPixel[0] = cv::saturate_cast<uchar>(sum[0]);
//                outPixel[1] = cv::saturate_cast<uchar>(sum[1]);
//                outPixel[2] = cv::saturate_cast<uchar>(sum[2]);
//            }
//        }
//    }
//
//    double end_time = omp_get_wtime();
//    std::cout << "Filter function execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
//
//    return dst;
//}
//
///**
// * Optimized low pass filter implementation
// * Key optimizations:
// * 1. Continuous memory blocks for better cache usage
// * 2. Block-based processing for cache locality
// * 3. Direct pointer access instead of at<>() calls
// * 4. Improved thread management with dynamic scheduling
// * 5. Loop tiling and data prefetching
// * 6. Optimized memory access patterns
// */
//cv::Mat applyLowPassFilterOptimized(const cv::Mat& src, const cv::Mat& kernel) {
//    // Create output image with same size and type as input
//    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
//    int channels = src.channels();
//    int padH = kernel.rows / 2;
//    int padW = kernel.cols / 2;
//
//    // Create padded version of source image
//    cv::Mat padded;
//    cv::copyMakeBorder(src, padded, padH, padH, padW, padW, cv::BORDER_REPLICATE);
//
//    // Convert kernel to 1D array for faster access with precomputed offsets
//    std::vector<float> kernelValues(kernel.rows * kernel.cols);
//    for (int ky = 0; ky < kernel.rows; ky++) {
//        for (int kx = 0; kx < kernel.cols; kx++) {
//            kernelValues[ky * kernel.cols + kx] = kernel.at<float>(ky, kx);
//        }
//    }
//
//    // Ensure continuous memory for better performance
//    if (!padded.isContinuous()) {
//        padded = padded.clone();
//    }
//    if (!dst.isContinuous()) {
//        dst = dst.clone();
//    }
//
//    // Set optimal thread count based on CPU cores and image size
//    int maxThreads = omp_get_max_threads();
//    int optimalThreads;
//    int imagePixels = src.rows * src.cols;
//
//    // Scale threads based on image size to prevent overhead on small images
//    if (imagePixels < 250000) { // Small image (500x500)
//        optimalThreads = std::min(maxThreads, 4);
//    } else if (imagePixels < 1000000) { // Medium image (1000x1000)
//        optimalThreads = std::min(maxThreads, 8);
//    } else { // Large image
//        optimalThreads = std::min(maxThreads, 16);
//    }
//    omp_set_num_threads(optimalThreads);
//
//    int width = src.cols;
//    int height = src.rows;
//    int kernelWidth = kernel.cols;
//    int kernelHeight = kernel.rows;
//
//    // Calculate optimal block size (fixed value, not variable)
//    const int BLOCK_SIZE = 32; // Process in smaller blocks to fit in L1 cache
//
//    if (channels == 1) {
//        // Grayscale processing - process in blocks for better cache utilization
//#pragma omp parallel for schedule(dynamic, 8)
//        for (int blockY = 0; blockY < height; blockY += BLOCK_SIZE) {
//            for (int blockX = 0; blockX < width; blockX += BLOCK_SIZE) {
//                // Calculate the valid range for this block
//                int endY = std::min(blockY + BLOCK_SIZE, height);
//                int endX = std::min(blockX + BLOCK_SIZE, width);
//
//                // Pre-compute row pointers for the block (avoid recalculation)
//                // Use std::vector instead of variable-length array
//                std::vector<uchar*> paddedRows(BLOCK_SIZE + 2*padH);
//                for (int y = 0; y < (endY - blockY) + 2*padH; y++) {
//                    paddedRows[y] = padded.ptr<uchar>(blockY + y);
//                }
//
//                // Process pixels in the block
//                for (int y = blockY; y < endY; y++) {
//                    uchar* dstRow = dst.ptr<uchar>(y);
//                    int localY = y - blockY;
//
//                    for (int x = blockX; x < endX; x++) {
//                        float sum = 0.0f;
//
//                        // Specialized fast paths for common kernel sizes
//                        if (kernelWidth == 3 && kernelHeight == 3) {
//                            // Optimized path for 3x3 kernels (very common)
//                            for (int ky = 0; ky < 3; ky++) {
//                                uchar* paddedRow = paddedRows[localY + ky];
//                                int kernelBase = ky * 3;
//
//                                sum += paddedRow[x] * kernelValues[kernelBase];
//                                sum += paddedRow[x + 1] * kernelValues[kernelBase + 1];
//                                sum += paddedRow[x + 2] * kernelValues[kernelBase + 2];
//                            }
//                        }
//                        else if (kernelWidth == 5 && kernelHeight == 5) {
//                            // Optimized path for 5x5 kernels
//                            for (int ky = 0; ky < 5; ky++) {
//                                uchar* paddedRow = paddedRows[localY + ky];
//                                int kernelBase = ky * 5;
//
//                                // Manually unrolled inner loop
//                                sum += paddedRow[x] * kernelValues[kernelBase];
//                                sum += paddedRow[x + 1] * kernelValues[kernelBase + 1];
//                                sum += paddedRow[x + 2] * kernelValues[kernelBase + 2];
//                                sum += paddedRow[x + 3] * kernelValues[kernelBase + 3];
//                                sum += paddedRow[x + 4] * kernelValues[kernelBase + 4];
//                            }
//                        }
//                        else {
//                            // General case for arbitrary kernel sizes
//                            for (int ky = 0; ky < kernelHeight; ky++) {
//                                uchar* paddedRow = paddedRows[localY + ky];
//                                int kernelBase = ky * kernelWidth;
//
//                                // Use SIMD-friendly memory access pattern
//                                for (int kx = 0; kx < kernelWidth; kx += 4) {
//                                    // Process up to 4 elements at once if possible
//                                    int remainingKx = std::min(4, kernelWidth - kx);
//                                    for (int k = 0; k < remainingKx; k++) {
//                                        sum += paddedRow[x + kx + k] * kernelValues[kernelBase + kx + k];
//                                    }
//                                }
//                            }
//                        }
//
//                        dstRow[x] = cv::saturate_cast<uchar>(sum);
//                    }
//                }
//            }
//        }
//    } else {
//        // Color image processing - optimized for 3-channel images
//#pragma omp parallel for schedule(dynamic, 4)
//        for (int blockY = 0; blockY < height; blockY += BLOCK_SIZE) {
//            for (int blockX = 0; blockX < width; blockX += BLOCK_SIZE) {
//                int endY = std::min(blockY + BLOCK_SIZE, height);
//                int endX = std::min(blockX + BLOCK_SIZE, width);
//
//                // Pre-compute row pointers for the block using vector
//                std::vector<cv::Vec3b*> paddedRows(BLOCK_SIZE + 2*padH);
//                for (int y = 0; y < (endY - blockY) + 2*padH; y++) {
//                    paddedRows[y] = padded.ptr<cv::Vec3b>(blockY + y);
//                }
//
//                for (int y = blockY; y < endY; y++) {
//                    cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
//                    int localY = y - blockY;
//
//                    for (int x = blockX; x < endX; x++) {
//                        float sum[3] = {0.0f, 0.0f, 0.0f};
//
//                        // Process each row of the kernel
//                        for (int ky = 0; ky < kernelHeight; ky++) {
//                            cv::Vec3b* paddedRow = paddedRows[localY + ky];
//                            int kernelBase = ky * kernelWidth;
//
//                            // Process each element of the kernel
//                            for (int kx = 0; kx < kernelWidth; kx++) {
//                                float kernelVal = kernelValues[kernelBase + kx];
//                                cv::Vec3b& pixel = paddedRow[x + kx];
//
//                                // Process all channels together
//                                sum[0] += pixel[0] * kernelVal;
//                                sum[1] += pixel[1] * kernelVal;
//                                sum[2] += pixel[2] * kernelVal;
//                            }
//                        }
//
//                        // Store results
//                        dstRow[x][0] = cv::saturate_cast<uchar>(sum[0]);
//                        dstRow[x][1] = cv::saturate_cast<uchar>(sum[1]);
//                        dstRow[x][2] = cv::saturate_cast<uchar>(sum[2]);
//                    }
//                }
//            }
//        }
//    }
//
//    return dst;
//}

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <omp.h>

std::string getImagePath();
cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel);

int main() {
    std::string path;
    bool flag = false;
    std::string grayFlag = "y";

    // Get valid image path
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

    // Option to convert to grayscale
    std::cout << "Do you want the image in grayscale? [y/n]: ";
    std::cin >> grayFlag;

    if (grayFlag[0] == 'y') {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }

    int kernelLength, kernelWidth;

    // Get kernel dimensions
    std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
    std::cout << "Choose Kernel length: ";
    std::cin >> kernelLength;
    std::cout << "Choose Kernel width: ";
    std::cin >> kernelWidth;

    std::cout << "\nYour configurations are:\n"
              << "Image path: " << path << "\n"
              << "Length = " << kernelLength << "\n"
              << "Width = " << kernelWidth << "\n";

    double start_time = omp_get_wtime();

    // Create separable 1D kernels, normalized to sum to 1
    cv::Mat horizontalKernel = cv::Mat::ones(1, kernelWidth, CV_32F) / kernelWidth;
    cv::Mat verticalKernel = cv::Mat::ones(kernelLength, 1, CV_32F) / kernelLength;

    cv::Mat tempImg = cv::Mat::zeros(img.size(), img.type());
    cv::Mat imgdest = cv::Mat::zeros(img.size(), img.type());

    // Apply horizontal pass
    int totalRows = img.rows;
#pragma omp parallel
    {
        int threadID = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int rowsPerThread = totalRows / numThreads;
        int startRow = threadID * rowsPerThread;
        int endRow = (threadID == numThreads - 1) ? totalRows : startRow + rowsPerThread;

        cv::Mat subImg = img.rowRange(startRow, endRow);
        cv::Mat subDest = applyLowPassFilter(subImg, horizontalKernel);
        subDest.copyTo(tempImg.rowRange(startRow, endRow));
    }

    // Apply vertical pass
#pragma omp parallel
    {
        int threadID = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int rowsPerThread = totalRows / numThreads;
        int startRow = threadID * rowsPerThread;
        int endRow = (threadID == numThreads - 1) ? totalRows : startRow + rowsPerThread;

        int pad = kernelLength / 2;
        int safeStart = std::max(startRow - pad, 0);
        int safeEnd = std::min(endRow + pad, totalRows);

        cv::Mat subImg = tempImg.rowRange(safeStart, safeEnd);
        cv::Mat subDest = applyLowPassFilter(subImg, verticalKernel);

        int copyStart = startRow - safeStart;
        int copyLength = endRow - startRow;
        subDest.rowRange(copyStart, copyStart + copyLength).copyTo(imgdest.rowRange(startRow, endRow));
    }

    double end_time = omp_get_wtime();
    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;

    // Display results
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
    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());
    return path;
}

//cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel) {
//    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
//    int channels = src.channels();
//
//    bool isHorizontal = (kernel.rows == 1);
//    bool isVertical = (kernel.cols == 1);
//
//    int padH = isVertical ? kernel.rows / 2 : 0;
//    int padW = isHorizontal ? kernel.cols / 2 : 0;
//
//    cv::Mat padded;
//    cv::copyMakeBorder(src, padded, padH, padH, padW, padW, cv::BORDER_REPLICATE);
//
//    std::vector<float> kernelValues;
//    if (isHorizontal) {
//        kernelValues.resize(kernel.cols);
//        for (int kx = 0; kx < kernel.cols; ++kx)
//            kernelValues[kx] = kernel.at<float>(0, kx);
//    } else if (isVertical) {
//        kernelValues.resize(kernel.rows);
//        for (int ky = 0; ky < kernel.rows; ++ky)
//            kernelValues[ky] = kernel.at<float>(ky, 0);
//    }
//
//    if (channels == 1) {
//        if (isHorizontal) {
//            for (int y = 0; y < src.rows; ++y) {
//                const uchar* paddedRow = padded.ptr<uchar>(y);
//                uchar* dstRow = dst.ptr<uchar>(y);
//                for (int x = 0; x < src.cols; ++x) {
//                    float sum = 0.0f;
//                    for (int kx = 0; kx < kernel.cols; ++kx)
//                        sum += paddedRow[x + kx] * kernelValues[kx];
//                    dstRow[x] = cv::saturate_cast<uchar>(sum);
//                }
//            }
//        } else if (isVertical) {
//            for (int y = 0; y < src.rows; ++y) {
//                uchar* dstRow = dst.ptr<uchar>(y);
//                for (int x = 0; x < src.cols; ++x) {
//                    float sum = 0.0f;
//                    for (int ky = 0; ky < kernel.rows; ++ky)
//                        sum += padded.ptr<uchar>(y + ky)[x] * kernelValues[ky];
//                    dstRow[x] = cv::saturate_cast<uchar>(sum);
//                }
//            }
//        }
//    } else {
//        if (isHorizontal) {
//            for (int y = 0; y < src.rows; ++y) {
//                const cv::Vec3b* paddedRow = padded.ptr<cv::Vec3b>(y);
//                cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
//                for (int x = 0; x < src.cols; ++x) {
//                    cv::Vec3f sum(0, 0, 0);
//                    for (int kx = 0; kx < kernel.cols; ++kx) {
//                        const cv::Vec3b& pixel = paddedRow[x + kx];
//                        float kv = kernelValues[kx];
//                        sum[0] += pixel[0] * kv;
//                        sum[1] += pixel[1] * kv;
//                        sum[2] += pixel[2] * kv;
//                    }
//                    dstRow[x] = cv::Vec3b(
//                            cv::saturate_cast<uchar>(sum[0]),
//                            cv::saturate_cast<uchar>(sum[1]),
//                            cv::saturate_cast<uchar>(sum[2])
//                    );
//                }
//            }
//        } else if (isVertical) {
//            for (int y = 0; y < src.rows; ++y) {
//                cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
//                for (int x = 0; x < src.cols; ++x) {
//                    cv::Vec3f sum(0, 0, 0);
//                    for (int ky = 0; ky < kernel.rows; ++ky) {
//                        const cv::Vec3b& pixel = padded.ptr<cv::Vec3b>(y + ky)[x];
//                        float kv = kernelValues[ky];
//                        sum[0] += pixel[0] * kv;
//                        sum[1] += pixel[1] * kv;
//                        sum[2] += pixel[2] * kv;
//                    }
//                    dstRow[x] = cv::Vec3b(
//                            cv::saturate_cast<uchar>(sum[0]),
//                            cv::saturate_cast<uchar>(sum[1]),
//                            cv::saturate_cast<uchar>(sum[2])
//                    );
//                }
//            }
//        }
//    }
//
//    return dst;
//}

cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    int channels = src.channels();

    bool isHorizontal = (kernel.rows == 1);
    bool isVertical = (kernel.cols == 1);

    int padH = isVertical ? kernel.rows / 2 : 0;
    int padW = isHorizontal ? kernel.cols / 2 : 0;

    cv::Mat padded;
    cv::copyMakeBorder(src, padded, padH, padH, padW, padW, cv::BORDER_REPLICATE);

    std::vector<float> kernelValues;
    if (isHorizontal) {
        kernelValues.resize(kernel.cols);
        for (int kx = 0; kx < kernel.cols; ++kx)
            kernelValues[kx] = kernel.at<float>(0, kx);
    } else if (isVertical) {
        kernelValues.resize(kernel.rows);
        for (int ky = 0; ky < kernel.rows; ++ky)
            kernelValues[ky] = kernel.at<float>(ky, 0);
    }

    if (channels == 1) {
        if (isHorizontal) {
            for (int y = 0; y < src.rows; ++y) {
                const uchar* paddedRow = padded.ptr<uchar>(y);
                uchar* dstRow = dst.ptr<uchar>(y);
#pragma omp simd
                for (int x = 0; x < src.cols; ++x) {
                    float sum = 0.0f;
                    for (int kx = 0; kx < kernel.cols; ++kx)
                        sum += paddedRow[x + kx] * kernelValues[kx];
                    dstRow[x] = cv::saturate_cast<uchar>(sum);
                }
            }
        } else if (isVertical) {
            for (int y = 0; y < src.rows; ++y) {
                uchar* dstRow = dst.ptr<uchar>(y);
#pragma omp simd
                for (int x = 0; x < src.cols; ++x) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < kernel.rows; ++ky)
                        sum += padded.ptr<uchar>(y + ky)[x] * kernelValues[ky];
                    dstRow[x] = cv::saturate_cast<uchar>(sum);
                }
            }
        }
    } else {
        if (isHorizontal) {
            for (int y = 0; y < src.rows; ++y) {
                const cv::Vec3b* paddedRow = padded.ptr<cv::Vec3b>(y);
                cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
#pragma omp simd
                for (int x = 0; x < src.cols; ++x) {
                    float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
                    for (int kx = 0; kx < kernel.cols; ++kx) {
                        const cv::Vec3b& pixel = paddedRow[x + kx];
                        float kv = kernelValues[kx];
                        sumB += pixel[0] * kv;
                        sumG += pixel[1] * kv;
                        sumR += pixel[2] * kv;
                    }
                    dstRow[x] = cv::Vec3b(
                            cv::saturate_cast<uchar>(sumB),
                            cv::saturate_cast<uchar>(sumG),
                            cv::saturate_cast<uchar>(sumR)
                    );
                }
            }
        } else if (isVertical) {
            for (int y = 0; y < src.rows; ++y) {
                cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
#pragma omp simd
                for (int x = 0; x < src.cols; ++x) {
                    float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
                    for (int ky = 0; ky < kernel.rows; ++ky) {
                        const cv::Vec3b& pixel = padded.ptr<cv::Vec3b>(y + ky)[x];
                        float kv = kernelValues[ky];
                        sumB += pixel[0] * kv;
                        sumG += pixel[1] * kv;
                        sumR += pixel[2] * kv;
                    }
                    dstRow[x] = cv::Vec3b(
                            cv::saturate_cast<uchar>(sumB),
                            cv::saturate_cast<uchar>(sumG),
                            cv::saturate_cast<uchar>(sumR)
                    );
                }
            }
        }
    }

    return dst;
}