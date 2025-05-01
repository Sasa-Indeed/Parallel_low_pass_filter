#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/utility.hpp>
#include <omp.h>
#include <iostream>

cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel);


int main() {
    // Load image
    cv::Mat img = cv::imread("D:\\ASU\\5-Senior 2\\Spring 25\\CSE455 HPC\\Project\\land.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Get number of threads
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    std::cout << "Applying low pass filter using " << max_threads << " threads" << std::endl;

    // Create kernel
    double start_time = omp_get_wtime();
    int kernel_size = 7;
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) / (kernel_size * kernel_size);
    int kernel_radius = (kernel_size - 1) / 2;

    // Calculate strip height
    int strip_height = gray.rows / max_threads;

    // Create storage for strips and results
    std::vector<cv::Mat> strips(max_threads);
    std::vector<cv::Mat> filtered_strips(max_threads);
    std::vector<cv::Range> row_ranges(max_threads);
    std::vector<int> top_offsets(max_threads);  // How many padding rows at the top

    // Step 1: Create the strips with appropriate padding
    for (int i = 0; i < max_threads; ++i) {
        // Calculate original strip boundaries
        int start_row = i * strip_height;
        int end_row = (i == max_threads - 1) ? gray.rows : (i + 1) * strip_height;

        // Store original ranges
        row_ranges[i] = cv::Range(start_row, end_row);

        // Calculate padded strip boundaries
        int padded_start = std::max(0, start_row - kernel_radius);
        int padded_end = std::min(gray.rows, end_row + kernel_radius);

        // Store offset information for later use
        top_offsets[i] = start_row - padded_start;

        // Create padded strip
        strips[i] = gray(cv::Range(padded_start, padded_end), cv::Range::all()).clone();
    }
    // Step 2: Filter each strip in parallel
#pragma omp parallel for
    for (int i = 0; i < max_threads; ++i) {
        // Apply filter to the strip
        filtered_strips[i] = cv::Mat(strips[i].size(), CV_8U);
        filtered_strips[i] = applyLowPassFilter(strips[i], kernel);
    }

    // Step 3: Extract valid portions and merge
    std::vector<cv::Mat> result_strips(max_threads);
    for (int i = 0; i < max_threads; ++i) {
        // Original strip height
        int orig_height = row_ranges[i].end - row_ranges[i].start;

        // Extract the valid region (excluding padding)
        result_strips[i] = filtered_strips[i](
            cv::Range(top_offsets[i], top_offsets[i] + orig_height),
            cv::Range::all()
            ).clone();

        // Debug: verify sizes
        std::cout << "Result strip " << i << " size: " << result_strips[i].rows << "x" << result_strips[i].cols << std::endl;
    }

    // Step 4: Concatenate the strips
    cv::Mat parallel_result;
    cv::vconcat(result_strips, parallel_result);
    // Step 5: get end time
    double end_time = omp_get_wtime();
    std::cout << "Execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;
    // Step 6: Display the results
    cv::imshow("Original Image", img);
    cv::imshow("Filtered Image", parallel_result);

    cv::waitKey(0);
    return 0;
}

/**
 * Apply a low pass filter to an image using a custom convolution implementation
 * @param src The source image
 * @param kernel The convolution kernel
 * @return The filtered image
 **/
cv::Mat applyLowPassFilter(const cv::Mat& src, const cv::Mat& kernel) {
    double start_time = omp_get_wtime();

    /* Create output image with same size and type as input */
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    /* Identifies how many color channels the image has (1 for grayscale, 3 for RGB) */
    int channels = src.channels();
    /* Calculate padding */
    int padH = kernel.rows / 2;
    int padW = kernel.cols / 2;
    /* Create padded version of source image */
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, padH, padH, padW, padW, cv::BORDER_REPLICATE);

    /* Cache kernel values to avoid repeated access */
    std::vector<float> kernelValues(kernel.rows * kernel.cols);
    for (int ky = 0; ky < kernel.rows; ky++) {
        for (int kx = 0; kx < kernel.cols; kx++) {
            kernelValues[ky * kernel.cols + kx] = kernel.at<float>(ky, kx);
        }
    }

    /* Apply convolution for each pixel in the image using OpenMP */
    if (channels == 1) {
        /* Grayscale processing */
#pragma omp parallel for collapse(2)
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                float sum = 0.0;
                /* Apply kernel */
                for (int ky = 0; ky < kernel.rows; ky++) {
                    for (int kx = 0; kx < kernel.cols; kx++) {
                        /* Calculate position in padded image */
                        int pixelY = y + ky;
                        int pixelX = x + kx;
                        sum += static_cast<float>(padded.at<uchar>(pixelY, pixelX)) *
                            kernelValues[ky * kernel.cols + kx];
                    }
                }
                /* Update output pixel */
                dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
            }
        }
    }
    else {
        /* Color image processing */
#pragma omp parallel for collapse(2)
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                float sum[3] = { 0.0, 0.0, 0.0 };
                /* Apply kernel */
                for (int ky = 0; ky < kernel.rows; ky++) {
                    for (int kx = 0; kx < kernel.cols; kx++) {
                        /* Calculate position in padded image */
                        int pixelY = y + ky;
                        int pixelX = x + kx;
                        float kernelVal = kernelValues[ky * kernel.cols + kx];
                        cv::Vec3b pixel = padded.at<cv::Vec3b>(pixelY, pixelX);

                        /* Process all channels together for better cache efficiency */
                        sum[0] += static_cast<float>(pixel[0]) * kernelVal;
                        sum[1] += static_cast<float>(pixel[1]) * kernelVal;
                        sum[2] += static_cast<float>(pixel[2]) * kernelVal;
                    }
                }
                /* Update output pixel */
                cv::Vec3b& outPixel = dst.at<cv::Vec3b>(y, x);
                outPixel[0] = cv::saturate_cast<uchar>(sum[0]);
                outPixel[1] = cv::saturate_cast<uchar>(sum[1]);
                outPixel[2] = cv::saturate_cast<uchar>(sum[2]);
            }
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "Filter function execution time: " << (end_time - start_time) * 1000.0 << " milliseconds" << std::endl;

    return dst;
}