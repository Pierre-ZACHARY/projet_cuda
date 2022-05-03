#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;

__global__ void vecadd( int * v1, int * v2 )
{
    auto tid = threadIdx.x;

    v2[tid] += v1[tid];

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


    std::vector< int > v1( 10 );
    std::vector< int > v2( 10 );

    int * v1_d = nullptr;
    int * v2_d = nullptr;

    for( std::size_t i = 0 ; i < v1.size() ; ++i )
    {
        v1[ i ] =  5;
        v2[ i ] =  4;
    }

    cudaMalloc( &v1_d, v1.size() * sizeof( int ) );
    cudaMalloc( &v2_d, v2.size() * sizeof( int ) );

    cudaMemcpy(v1_d, v1.data(), v1.size() * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy(v2_d, v2.data(), v2.size() * sizeof( int ), cudaMemcpyHostToDevice );

    vecadd<<< 1, 10 >>>( v1_d, v2_d );

    cudaMemcpy(v2.data(),v2_d, v2.size() * sizeof( int ), cudaMemcpyDeviceToHost );



    for (size_t idex = 0; idex < v1.size(); idex++)
        std::cout <<   v2[idex] << " ";
    std::cout << std::endl;



    cudaFree( v1_d );
    cudaFree( v2_d );

    return 0;
}
