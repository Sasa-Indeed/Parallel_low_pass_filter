#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/utility.hpp>
#include <omp.h>
#include <iostream>

int main() {
    // Load image
    cv::Mat img = cv::imread("C:\\Users\\Mohamed Ali\\Documents\\Final_Semester\\HPC\\parallel_low_pass_filter\\image.png", cv::IMREAD_COLOR);
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
    int kernel_size = 49;
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) / (kernel_size * kernel_size);
    int kernel_radius = (kernel_size - 1) / 2;

    // Calculate strip height
    int strip_height = gray.rows / max_threads;

    // Create storage for strips and results
    std::vector<cv::Mat> strips(max_threads);
    std::vector<cv::Mat> filtered_strips(max_threads);
    std::vector<cv::Range> row_ranges(max_threads);
    std::vector<int> top_offsets(max_threads);  // How many padding rows at the top

    // Step 0: get start time
    double start_time = omp_get_wtime();
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
        cv::filter2D(strips[i], filtered_strips[i], CV_8U, kernel);
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
    std::cout << "Time taken for parallel processing: " << (end_time - start_time) << " seconds" << std::endl;
    // Step 6: Display the results
    cv::imshow("Original Image", img);
    cv::imshow("Filtered Image", parallel_result);
    
    cv::waitKey(0);
    return 0;
}