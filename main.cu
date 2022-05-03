#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;

__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;
    if( i < cols && j < rows ) {
        g[ j * cols + i ] = (
                                    307 * rgb[ 3 * ( j * cols + i ) ]
                                    + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
                                    + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
                            ) / 1024;
    }
}

int main()
{

    int count = 0;

    cudaGetDeviceCount( &count );

    std::cout << count << " device(s) found.\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "maxBlocksPerMultiProcessor " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerBlock " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "multiProcessorCount " << prop.multiProcessorCount << std::endl;
    std::cout << "warpSize " << prop.warpSize << std::endl;

    Mat m_in = imread("../in.jpg", IMREAD_UNCHANGED );
    std::cout << "Image Size : " << m_in.rows * m_in.cols << std::endl;



    auto rgb = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    std::vector< unsigned char > g( rows * cols );
    cv::Mat m_out( rows, cols, CV_8UC1, g.data() );
    unsigned char * rgb_d;
    unsigned char * g_d;
    cudaMalloc( &rgb_d, 3 * rows * cols );
    cudaMalloc( &g_d, rows * cols );
    cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
    dim3 t( 32, 32 );
    dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
    grayscale<<< b, t >>>( rgb_d, g_d, cols, rows );
    cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost );
    cv::imwrite( "../out.jpg", m_out );
    cudaFree( rgb_d);
    cudaFree( g_d);

    return 0;
}
