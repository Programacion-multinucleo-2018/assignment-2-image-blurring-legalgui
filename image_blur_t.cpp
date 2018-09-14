// SYSTEM LIBRARIES
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
// OPEN CV LIRBARIES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// FILTER SIZE
#define g_FILTER_SIZE 5
// GLOBAL MEMORY
float filter[g_FILTER_SIZE][g_FILTER_SIZE] = {
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
void image_blur_host(const cv::Mat& input, cv::Mat& output, const int filterBoundary, const int filterSize){

  #pragma omp parallel for private(i, j, k, l) shared(input, output)
  // FOR ALL IMAGE
  for(int i = 0; i < input.rows; i++) {
    for(int j = 0; j < input.cols; j++) {
      // ACCUMULATORS
      float blueSum = 0.0;
      float greenSum = 0.0;
      float redSum = 0.0;
      float value = 0;
      // FOR FILTER BOUNDARY
      for(int k = -filterBoundary, k_f = 0; (k <= filterBoundary) && (k_f <= filterSize); k++, k_f++) {
        for(int l = -filterBoundary, l_f = 0; (l <= filterBoundary) && (l_f <= filterSize); l++, l_f++) {

          // GET ID FOR MATRIX
          int idX = i + k;
          int idY = j + l;
          // GET THE VALUE FILTER
          value = filter[k_f][l_f];
          // DON'T GET OUT OF BOUNDS
          if(idY > 0 && idX > 0 && idY < input.cols && idX < input.rows) {
            blueSum += input.at<cv::Vec3b>(idX, idY)[0] * value;
            greenSum += input.at<cv::Vec3b>(idX, idY)[1] * value;
            redSum += input.at<cv::Vec3b>(idX, idY)[2] * value;
          }
        }
      }
      output.at<cv::Vec3b>(i, j)[0] = blueSum;
      output.at<cv::Vec3b>(i, j)[1] = greenSum;
      output.at<cv::Vec3b>(i, j)[2] = redSum;
    }
  }
}

// CALL WRAPPER FUNCTION
int image_blur(cv::Mat &input, cv::Mat output){
  // FILTER SIZE AND BOUNDARY DEFINITION
  const int filterSize = g_FILTER_SIZE;
  const int filterBoundary = floor(g_FILTER_SIZE/2);
  // INPUT.STEP GETS THE NUMBER OF BYTES FOR EACH ROW
  std::cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << std::endl;
  // CALCULATE TOTAL NUMBER OF BYTES OF OUTPUT IMAGE
  // SUMMON FUNCTION
  auto start_cpu = std::chrono::high_resolution_clock::now();
  image_blur_host(input, output, filterBoundary, filterSize);
  auto end_cpu =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  printf("Applying filter to image on host elapsed %f ms\n", duration_ms.count());

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
  image_blur(input, output);
  cv::imwrite("output_host_threads.jpg", output);

	return 0;
}
