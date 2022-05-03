#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

enum class ImageFormat : int{ Grayscale = 1, RGB = 3, RGBA = 4 };

vector<float> edge_detection_kernel = {-1.0f, -1.0f, -1.0f,
                                       -1.0f,  8.0f, -1.0f,
                                       -1.0f, -1.0f, -1.0f};


vector<float> gaussian_blur = {0, 0,  0,  5,  0, 0,0,
                               0, 5, 18, 32, 18, 5,0,
                               0,18, 64,100, 64,18,0,
                               5,32,100,100,100,32,5,
                               0,18, 64,100, 64,18,0,
                               0, 5, 18, 32, 18, 5,0,
                               0, 0,  0,  5,  0, 0,0};

void execKernelCpuSeq(Mat* image, vector< unsigned char >* output, vector<float>* kernel, ImageFormat format){

    int nbChannel = (int) format;
    assert(kernel->size()%2 == 1);

    int kernelRowSize = sqrt(kernel->size());

    float kernelSum = 0;
    for(float k : *kernel){
        kernelSum += abs(k);
    }

    cout << "Convolution version séquentielle, taille de l'image : "<<image->cols<<"x"<<image->rows<<", taille du kernel : "<<kernelRowSize<<"x"<<kernelRowSize<<"...\n";
    time_point<system_clock> start, end;
    start = system_clock::now();

    for(int i=0; i<image->rows; i++){
        for(int j=0; j<image->cols; j++){ // pour chaque pixel de l'image
            for(int c=0; c<nbChannel; c++){ // pour chaque couleur de ce pixel

                float product_sum = 0.0f;

                for(int iKernel = 0; iKernel < kernelRowSize; iKernel++){
                    for(int jKernel = 0; jKernel < kernelRowSize; jKernel ++){ // pour chaque case du kernel

                        float kernelValue = (*kernel)[iKernel*kernelRowSize+jKernel] / kernelSum; // on divise la valeur de cette case par la somme des valeurs du kernel

                        int pixel_ind_i = i + iKernel - kernelRowSize/2 ;
                        int pixel_ind_j = j + jKernel - kernelRowSize/2 ;

                        int pixel_val;

                        if(pixel_ind_i < 0 || pixel_ind_i>image->rows || pixel_ind_j<0 || pixel_ind_j>image->cols){ // si le pixel est en dehors de l'image
                            pixel_val = 127;
                        }
                        else{ // sinon
                            pixel_val = image->data[(pixel_ind_i*image->cols+pixel_ind_j)*nbChannel + c];
                        }

                        product_sum += ((float) pixel_val) * kernelValue; // on ajoute à product_sum la valeur du pixel * la valeur du kernel
                    }
                }
                (*output)[(i*image->cols+j)*nbChannel + c] = (unsigned char) ((int) product_sum);
            }
        }
    }

    end = system_clock::now();
    duration<double> elapsed_seconds = end - start;
    cout << "Temps : "<<elapsed_seconds.count()<<"s\n";
}

void runCpu(Mat* image, ImageFormat format, const String& outputPath ){

    int nbChannel = (int) format;
    vector< unsigned char > g( image->cols * image->rows * nbChannel );
    Mat m_out( image->rows, image->cols, image->type(), g.data() );

    cout<<"BLUR ...\n";

    vector<float> blur3x3 = vector<float>(3*3, 1.0f);
    execKernelCpuSeq(image, &g, &blur3x3, format);
    imwrite( outputPath+"cpu_blur3x3.jpg", m_out );

    vector<float> blur9x9 = vector<float>(9*9, 1.0f);
    execKernelCpuSeq(image, &g, &blur9x9, format);
    imwrite( outputPath+"cpu_blur9x9.jpg", m_out );

    vector<float> blur15x15 = vector<float>(15*15, 1.0f);
    execKernelCpuSeq(image, &g, &blur15x15, format);
    imwrite( outputPath+"cpu_blur15x15.jpg", m_out );


    cout<<"Edge Detection ...\n";

    execKernelCpuSeq(image, &g, &edge_detection_kernel, format);
    imwrite( outputPath+"cpu_edge_detection.jpg", m_out );

    cout<<"Gaussian Blur ...\n";

    execKernelCpuSeq(image, &g, &gaussian_blur, format);
    imwrite( outputPath+"cpu_gaussian_blur.jpg", m_out );

}

void runGpu(Mat* image, vector<float>* kernel, ImageFormat format){

}



int main()
{
    Mat m_in = imread("../in.jpg", IMREAD_UNCHANGED );
    runCpu(&m_in, ImageFormat::RGB, "../output/cpu/");

    return 0;
}
