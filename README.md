# Parallel_low_pass_filter

## Overview

This project implements a **Low Pass Filter** for image processing. Low pass filtering is a technique used to make images appear smoother by reducing noise. It works by allowing **low frequency components** of the image to pass through while **blocking high frequencies**, which often represent noise or sharp edges.

The filtering process is done using **convolution**, where each pixel of the image is recalculated using a surrounding neighborhood defined by a **fixed-size kernel**. This operation blends nearby pixels together, effectively smoothing the image.

## Purpose

The primary goal of this project is to demonstrate the implementation of a low pass filter using three different approaches:

1. **Sequential Implementation** (Compulsory)
2. **OpenMP Parallel Implementation** (Compulsory)
3. **MPI Distributed Implementation** (Compulsory)

By comparing these three methods, we aim to highlight the performance differences between sequential and parallel processing strategies, as well as gain insight into parallel programming models.

## Features

- Convolution-based image smoothing
- Adjustable kernel size
- Three implementations:
  - **Sequential**: Basic, single-threaded processing
  - **OpenMP**: Multi-threaded shared-memory parallelism
  - **MPI**: Distributed-memory parallelism for scalability across multiple nodes

## Dependencies

- C/C++ Compiler (e.g., GCC)
- OpenMP support
- MPI Library (e.g., MPICH or OpenMPI)
- Image processing library (OpenCV)
