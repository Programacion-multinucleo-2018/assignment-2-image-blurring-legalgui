// SYSTEM LIBRARIES
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
// OPEN CV LIRBARIES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// CUDA LIBRARY
#include <cuda_runtime.h>
// CUDA CUSTOM LIBRARY
#include "common.h"

// FILTER SIZE
#define g_FILTER_SIZE 5
// CONSTANT MEMORY FOR THE CONVOLUTION FILTER
__constant__ float cst_filter[g_FILTER_SIZE][g_FILTER_SIZE] = {
  {0.04, 0.04, 0.04, 0.04, 0.04},
  {0.04, 0.04, 0.04, 0.04, 0.04},
  {0.04, 0.04, 0.04, 0.04, 0.04},
  {0.04, 0.04, 0.04, 0.04, 0.04},
  {0.04, 0.04, 0.04, 0.04, 0.04}
};

// input - input image one dimensional array
// ouput - output image one dimensional array
// width, height - width and height of the images
// colorWidthStep - number of color bytes (cols * colors)
// grayWidthStep - number of gray bytes
__global__ void image_blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int filterBoundary, int filterSize){

	// 2D INDEX OF CURRENT THREAD
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  // CHANNEL ACCUMULATORS
  float blueSum = 0.0;
  float greenSum = 0.0;
  float redSum = 0.0;

  // HANDLER
  float value;

	// ONLY VALID THREADS IN MEMORY
	if ((xIndex < width) && (yIndex < height)){

		// LOCATION OF CURRENT COLORED PIXEL
		const int currentTid = yIndex * colorWidthStep + (3 * xIndex);

    int i, j, i_f, j_f;
    for(i = -filterBoundary, i_f = 0; (i <= filterBoundary) && (i_f < filterSize); i++, i_f++){
      for(j = -filterBoundary, j_f = 0; (j <= filterBoundary) && (j_f < filterSize); j++, j_f++){

        // COMPUTE ONLY FOR BOUNDS
        if(yIndex + i >= 0 && xIndex + j >= 0 && yIndex + i < height && xIndex + i < width){
          const int neighborTid = (yIndex+i)*colorWidthStep + (3 * (xIndex + j));
          value = cst_filter[i_f][j_f];
          // MULTIPLY EACH CHANNEL BY THE MATRIX VALUE
          blueSum += input[neighborTid] * value;
          greenSum += input[neighborTid + 1] * value;
          redSum += input[neighborTid + 2] * value;
          }
      }
    }

    // FORBIDDEN OUT OF RANGE VALUES
    if(blueSum > 255) blueSum = 255;
    if(greenSum > 255) greenSum = 255;
    if(redSum > 255) redSum = 255;

    // ASSIGN THE SUM TO THE ELEMENT OF EACH CHANNEL OF THE OUTPUT
    output[currentTid] = static_cast<unsigned char>(blueSum);
    output[currentTid + 1] = static_cast<unsigned char>(greenSum);
    output[currentTid + 2] = static_cast<unsigned char>(redSum);
  }
}

// FUNCTION TO SUMMON THE KERNEL
void image_blur(const cv::Mat& input, cv::Mat& output){

  // FILTER SIZE AND BOUNDARY DEFINITION
  const int filterSize = g_FILTER_SIZE;
  const int filterBoundary = floor(g_FILTER_SIZE/2);

  // INPUT.STEP GETS THE NUMBER OF BYTES FOR EACH ROW
	std::cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << std::endl;
  // CALCULATE TOTAL NUMBER OF BYTES OF INPUT AND OUTPUT IMAGE
	// STEP = COLS * NUMBER OF COLORS
	size_t colorBytes = input.step * input.rows;
	size_t blurredBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// ALLOCATE DEVICE MEMORY
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, blurredBytes), "CUDA Malloc Failed");

	// COPY DATA FROM OPENCV INPUT IMAGE TO DEVICE MEMORY
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// SPECIFY A REASONABLE BLOCK SIZE
	const dim3 block(16, 16);

	// CALCULATE GRID SIZE AND PRINT
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("image_blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);


	// LAUNCH THE KERNEL
  auto start_cpu = std::chrono::high_resolution_clock::now();
	image_blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), filterBoundary, filterSize);
  auto end_cpu =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  printf("Applying filter to image elapsed %f ms\n", duration_ms.count());

	// SYNCH AND CHECK FOR ERRORS
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// COPY BACK DATA FROM DESTINATION DEVICE MEMORY TO OPENCV OUTPUT IMAGE
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, blurredBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// FREE DEVICE MEMORY
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[]){
  // GET THE IMAGE PATH
	std::string imagePath;
  (argc < 2) ? imagePath = "image.jpg" : imagePath = argv[1];

	// READ INPUT IMAGE FROM DISK
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty()){
		std::cout << "Image Not Found!" << std::endl;
		std::cin.get();
		return -1;
	}

	// CREATE OUTPUT IMAGE
	cv::Mat output = input.clone();

	// CALL THE WRAPPER FUNCTION
	image_blur(input, output);
  cv::imwrite("output.jpg", output);

	// UNCOMMENT FOR REVIEW
	//Allow the windows to resize
	//namedWindow("Input", cv::WINDOW_NORMAL);
	//namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	//imshow("Input", input);
	//imshow("Output", output);

	//Wait for key press
	//cv::waitKey();

	return 0;
}
