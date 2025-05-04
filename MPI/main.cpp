#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

std::string getImagePath();


void applyBlurToChannel(const Mat &input_channel, Mat &output_channel, int kernel_size)
{
    int rows = input_channel.rows;
    int cols = input_channel.cols;
    int k = kernel_size / 2;

    Mat temp(rows, cols, CV_32F);

    for (int y = 0; y < rows; y++)
    {
        const uchar *in_row = input_channel.ptr<uchar>(y);
        float *temp_row = temp.ptr<float>(y);

        for (int x = 0; x < cols; x++)
        {
            float sum = 0.0f;
            int count = 0;

            for (int i = max(0, x - k); i <= min(cols - 1, x + k); i++)
            {
                sum += in_row[i];
                count++;
            }

            temp_row[x] = sum / count;
        }
    }

    output_channel = Mat(rows, cols, CV_8U);

    for (int x = 0; x < cols; x++)
    {
        for (int y = 0; y < rows; y++)
        {
            float sum = 0.0f;
            int count = 0;

            for (int j = max(0, y - k); j <= min(rows - 1, y + k); j++)
            {
                sum += temp.at<float>(j, x);
                count++;
            }

            output_channel.at<uchar>(y, x) = saturate_cast<uchar>(sum / count);
        }
    }
}

void applyBoxBlur(const Mat &input, Mat &output, int kernel_size)
{
    vector<Mat> channels(3);
    vector<Mat> blurred_channels(3);
    split(input, channels);

    for (int c = 0; c < 3; c++)
    {
        applyBlurToChannel(channels[c], blurred_channels[c], kernel_size);
    }

    merge(blurred_channels, output);
}

Mat createComparisonImage(const Mat &input, const Mat &output)
{
    Mat resized_output;
    if (input.size() != output.size())
    {
        resize(output, resized_output, input.size());
    }
    else
    {
        resized_output = output.clone();
    }

    int height = input.rows;
    int width = input.cols;
    Mat comparison(height + 60, width * 2 + 10, CV_8UC3, Scalar(255, 255, 255));

    putText(comparison, "Original", Point(width / 2 - 50, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
    putText(comparison, "Blurred", Point(width + width / 2 - 50, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);

    input.copyTo(comparison(Rect(0, 50, width, height)));
    resized_output.copyTo(comparison(Rect(width + 10, 50, width, height)));

    line(comparison, Point(width + 5, 0), Point(width + 5, height + 50), Scalar(0, 0, 0), 2);

    return comparison;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    bool flag = false;
    string input_path;
    string output_path = "outImage.jpg";
    int kernel_size = 0;

    Mat image;
    int rows = 0, cols = 0, channels = 0;

    if (rank == 0)
    {
        do {
            if (flag) {
                std::cout << "Incorrect path please check the path and try again!\n";
            }
            input_path = getImagePath();
            flag = true;
        } while (!cv::haveImageReader(input_path));

        std::cout << "Dynamic kernel choices must be > 1 and ideally odd (3, 5, 7...)\n";
        std::cout << "Choose Kernel length: ";
        std::cin >> kernel_size;

        image = imread(input_path, IMREAD_COLOR);
        if (image.empty())
        {
            cout << "Could not open or find the image." << endl;
            MPI_Finalize();
            return -1;
        }

        rows = image.rows;
        cols = image.cols;
        channels = image.channels();
    }

    auto start_time = high_resolution_clock::now();

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);  // NEW: broadcast kernel size

    int k = kernel_size / 2;
    int rows_per_process = rows / num_processes;
    int remainder = rows % num_processes;

    int start_row = rank * rows_per_process + min(rank, remainder);
    int local_rows = rows_per_process + (rank < remainder ? 1 : 0);

    int top_row = max(0, start_row - k);
    int bottom_row = min(rows, start_row + local_rows + k);
    int actual_rows = bottom_row - top_row;

    vector<int> sendcounts(num_processes);
    vector<int> displs(num_processes);

    for (int i = 0; i < num_processes; ++i)
    {
        int i_start = i * rows_per_process + min(i, remainder);
        int i_rows = rows_per_process + (i < remainder ? 1 : 0);
        int i_top = max(0, i_start - k);
        int i_bottom = min(rows, i_start + i_rows + k);
        int i_actual = i_bottom - i_top;

        sendcounts[i] = i_actual * cols * channels;
        displs[i] = i_top * cols * channels;
    }

    Mat local_chunk(actual_rows, cols, CV_8UC3); // FIXED: moved after actual_rows is defined

    if (rank == 0)
    {
        MPI_Scatterv(image.data, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
            local_chunk.data, sendcounts[rank], MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
            local_chunk.data, sendcounts[rank], MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);
    }

    string input_chunk_filename = "process_" + to_string(rank) + "_input.jpg";
    imwrite(input_chunk_filename, local_chunk);

    int valid_start = (start_row <= top_row) ? 0 : (start_row - top_row);
    int valid_end = valid_start + local_rows;

    Mat blurred_chunk;
    applyBoxBlur(local_chunk, blurred_chunk, kernel_size);

    Mat valid_result = blurred_chunk(Range(valid_start, valid_end), Range::all());

    vector<int> recvcounts(num_processes);
    vector<int> recvdispls(num_processes);

    for (int i = 0; i < num_processes; ++i)
    {
        int i_start = i * rows_per_process + min(i, remainder);
        int i_rows = rows_per_process + (i < remainder ? 1 : 0);

        recvcounts[i] = i_rows * cols * channels;
        recvdispls[i] = i_start * cols * channels;
    }

    Mat final_image;
    if (rank == 0)
    {
        final_image = Mat(rows, cols, CV_8UC3);
    }

    MPI_Gatherv(valid_result.data, recvcounts[rank], MPI_UNSIGNED_CHAR,
        rank == 0 ? final_image.data : nullptr, recvcounts.data(),
        recvdispls.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        double elapsed_time = duration.count() / 1000.0;

        imwrite(output_path, final_image);
        cout << "Blurred image saved to " << output_path << endl;

        Mat full_comparison = createComparisonImage(image, final_image);
        imwrite("full_comparison.jpg", full_comparison);
        cout << "Side-by-side comparison saved to full_comparison.jpg" << endl;

        cout << "----------------------------------------" << endl;
        cout << "Performance Statistics:" << endl;
        cout << "Total execution time: " << elapsed_time << " seconds" << endl;
        cout << "Image size: " << rows << "x" << cols << " pixels" << endl;
        cout << "Number of processes: " << num_processes << endl;
        cout << "Kernel size: " << kernel_size << endl;
        cout << "----------------------------------------" << endl;
    }

    MPI_Finalize();
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