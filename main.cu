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


__global__ void gpuConvolution(unsigned char * input, unsigned char * output, float * kernel, size_t kernel_size, size_t nb_color_channels, size_t start_index, size_t rows, size_t cols ){
    extern __shared__ float kernel_shared[];
    auto id = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x<kernel_size){ // ça implique qu'il doit y avoir au moins autant de thread par block que de case dans le kernel
        kernel_shared[threadIdx.x] = kernel[threadIdx.x]; // on charge en mémoire shared l'ensemble de la matrice kernel, ça évite que chaque thread aient besoin d'autant d'appels en mémoire global qu'il y a d'éléments dans kernel
    }
    __syncthreads(); // il faut que tous les threads du bloc attendent que kernel_shared soit chargé


    float local_sum = 0;

    int kernel_row_size =(int) sqrtf((float) kernel_size);
    float kernel_sum = 0;
    for(int ik = 0; ik<kernel_size; ik++){
        kernel_sum += kernel_shared[ik];
    }

    int ligne_actuelle = (((start_index + id)/nb_color_channels))/cols;
    int colonne_actuelle =  (((start_index+ id)/nb_color_channels) )%cols;
    int chan_actuel =  id%nb_color_channels;

    for(int ik = 0; ik<kernel_size; ik++){
        int ligne = ik/kernel_row_size - kernel_row_size/2;
        int col = ik%kernel_row_size - kernel_row_size/2;
        unsigned char other_pixel;
        // On doit vérifier que l'autre pixel se trouve bien dans l'image, sinon on lui donne une valeur par défaut
        if(ligne_actuelle+ligne<0 || ligne_actuelle+ligne>=rows || colonne_actuelle+col<0 || colonne_actuelle+col>=cols){
            other_pixel = (unsigned char) 127;
        }
        else{
            other_pixel = input[(ligne_actuelle+ligne)*cols*nb_color_channels + (colonne_actuelle+col)*nb_color_channels + chan_actuel];
        }


        local_sum += ((kernel_shared[ik]/kernel_sum) * ((float) other_pixel));
    }


    output[(ligne_actuelle)*cols*nb_color_channels + (colonne_actuelle)*nb_color_channels + chan_actuel] = (unsigned char) ((int) local_sum);

}

void handle_error(cudaError_t cudaStatus, const String& info){
    if(cudaStatus != cudaSuccess){
        cout << "Error : "<<cudaStatus<< " on "<<info<<"\n";
    }
}

void execKernelGpu(Mat* image, vector< unsigned char >* output, vector<float>* kernel, ImageFormat format, unsigned int nb_stream){

    // Setup event
    cudaEvent_t start, stop;
    handle_error(cudaEventCreate( &start ), "cudaEventCreate( &start )");
    handle_error(cudaEventCreate( &stop ), "cudaEventCreate( &stop )");
    handle_error(cudaEventRecord( start ), "cudaEventRecord( start )");

    // Streams declaration.
    cudaStream_t streams[ nb_stream ];

    // Malloc Host & Device
    unsigned char * host_input;
    unsigned char * host_output;
    unsigned char * device_input;
    unsigned char * device_output;
    float * host_kernel;
    float * device_kernel;
    int nbChannel = (int) format;
    size_t const input_size = nbChannel * image->cols * image->rows;
    size_t const input_sizeb = input_size * sizeof( unsigned char ); // pour une image rgb de 1920*1200 pixel, on a 1920*1200*3* sizeof(unsigned char)=1 octets
    size_t const kernel_sizeb = kernel->size() * sizeof(float);
    handle_error(cudaMallocHost( &host_input, input_sizeb ), "cudaMallocHost( &host_input, input_sizeb )");
    handle_error(cudaMallocHost( &host_output, input_sizeb ),"cudaMallocHost( &host_output, input_sizeb )");
    handle_error(cudaMallocHost( &host_kernel, kernel_sizeb ),"cudaMallocHost( &host_kernel, kernel_sizeb )");
    handle_error(cudaMalloc(&device_input, input_sizeb),"cudaMalloc(&device_input, input_sizeb)");
    handle_error(cudaMalloc(&device_output, input_sizeb),"cudaMalloc(&device_output, input_sizeb)");
    handle_error(cudaMalloc(&device_kernel, kernel_sizeb),"cudaMalloc(&device_kernel, kernel_sizeb)");

    // Set Host Input & Host Kernel
    for(int h_i = 0; h_i<input_size; h_i++){ // je sais pas si c'est vraiment nécessaire, on aurait peut être pu faire host_input = image->data
        host_input[h_i] = image->data[h_i];
    }
    for(int k_i = 0; k_i<kernel->size(); k_i++){
        host_kernel[k_i] = (*kernel)[k_i];
    }


    int nb_ligne_supplementaire = sqrt(kernel->size())/2; // pour un kernel de size 3x3, on a besoin d'une ligne supplémentaire de chaque côté ( en haut et en bas ) donc division entiere kernel.rows / 2
    int taille_dune_ligne_bits = (image->cols) * nbChannel * sizeof(unsigned char); // marge de 1 pour les coins

//    cout<<nb_ligne_supplementaire<<endl;

    for(int s = 0; s<nb_stream; s++){ // pour chaque stream
        // Creation.
        handle_error(cudaStreamCreate( &streams[ s ] ), "cudaStreamCreate( &streams[ "+to_string(s)+" ] )");


        // on a besoin d'avoir accès à un certain nombre de ligne avant et après les threads qui sont géré par le stream
        int marge_inferieur = 0;
        int marge_superieur = 0;

        if(s == 0 && s!= nb_stream-1){
             marge_superieur = nb_ligne_supplementaire* taille_dune_ligne_bits + nbChannel * sizeof(unsigned char);
        }
        else if(s == nb_stream-1 && s != 0){ // rien après donc -1 +1
            marge_inferieur = nb_ligne_supplementaire * taille_dune_ligne_bits + nbChannel * sizeof(unsigned char);
            marge_superieur = nb_ligne_supplementaire* taille_dune_ligne_bits;
        }
        else if(s != 0 && s!= nb_stream-1){ // une ligne avant + une ligne après ( donc - 1 + 2 )
            marge_inferieur = nb_ligne_supplementaire*taille_dune_ligne_bits + nbChannel * sizeof(unsigned char);
            marge_superieur = 2*nb_ligne_supplementaire*taille_dune_ligne_bits + nbChannel * sizeof(unsigned char);
        }

        //cout<<"stream :"<<s<<", "<<marge_inferieur<<"/"<<marge_superieur<<endl;
        //cout<<"last item :"<<(int) host_input[input_size/nb_stream * sizeof(unsigned char) + marge_superieur]<<endl;

        // Copy Host Input -> Device Input & Host Kernel -> Device Kernel
        handle_error(cudaMemcpyAsync( device_input+((s*(input_size/nb_stream)) - marge_inferieur),
                         host_input+(s*(input_size/nb_stream) - marge_inferieur),
                         input_size/nb_stream * sizeof(unsigned char) + marge_superieur ,
                         cudaMemcpyHostToDevice,
                         streams[ s ] ),"cudaMemcpyAsync Host_Input -> Device_Input for stream "+to_string(s));
        handle_error(cudaMemcpyAsync( device_kernel,
                         host_kernel,
                         kernel->size() * sizeof(float),
                         cudaMemcpyHostToDevice,
                         streams[ s ] ), "cudaMemcpyAsync Host_Kernel -> Device_Kernel for stream "+to_string(s)); // tous les streams ont besoin du kernel complet


        //dim3 block( 32, 4 ); // on va faire des block de la taille d'un warp * 4
        //dim3 grid( ( cols - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 ); // il nous faut une grille de block qui couvre tous les pixels gérer par le stream actuel

        // chaque calcul est indépendant donc on a pas besoin de ghosts, il faut juste envoyer une ou plusieurs lignes supplémentaires pour que les dernier thread aient accès aux pixels non gérer
        gpuConvolution<<< ((input_size/1024))/nb_stream+1, 1024, kernel->size() * sizeof(float), streams[ s ] >>>( device_input, device_output, device_kernel, kernel->size(), nbChannel, s*(input_size/nb_stream), image->rows, image->cols);
        handle_error(cudaGetLastError(), "gpuConvolution for stream : "+ to_string(s));


        // Copy Device Output -> Host Output
        handle_error(cudaMemcpyAsync( host_output+(s*(input_size/nb_stream)),
                         device_output+(s*(input_size/nb_stream)),
                         input_size/nb_stream * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost,
                         streams[ s ] ),"cudaMemcpyAsync Device_Output -> Host_Output for stream "+ to_string(s));
    }

    // Synchronize everything.
    handle_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");

    // Set output data from Host Output
    for(int h_d = 0; h_d<input_size; h_d++){
        (*output)[h_d] = host_output[h_d];
    }

    // Destroy streams.
    for(int s = 0; s<nb_stream; s++){
        handle_error(cudaStreamDestroy( streams[ s ] ), "cudaStreamDestroy( streams[ "+ to_string(s)+" ] )");
    }

    // Record event
    handle_error(cudaEventRecord( stop ), "cudaEventRecord( stop )");
    handle_error(cudaEventSynchronize( stop ), "cudaEventSynchronize( stop )");
    float duration;
    handle_error(cudaEventElapsedTime( &duration, start, stop ), "cudaEventElapsedTime( &duration, start, stop )");
    std::cout << "time=" << duration << std::endl;
    handle_error(cudaEventDestroy(start), "cudaEventDestroy(start)");
    handle_error(cudaEventDestroy(stop), "cudaEventDestroy(stop)");


}

void runGpu(Mat* image, ImageFormat format, const String& outputPath){
    int nbChannel = (int) format;
    vector< unsigned char > g( image->cols * image->rows * nbChannel );
    Mat m_out( image->rows, image->cols, image->type(), g.data() );

    cout<<"BLUR ...\n";

    vector<float> blur3x3 = vector<float>(3*3, 1.0f);
    execKernelGpu(image, &g, &blur3x3, format, 4);
    imwrite( outputPath+"gpu_blur3x3.jpg", m_out );


    vector<float> blur15x15 = vector<float>(15*15, 1.0f);
    execKernelGpu(image, &g, &blur15x15, format, 4);
    imwrite( outputPath+"gpu_blur15x15.jpg", m_out );

}



int main()
{
    cout << "Première image ...\n";
    Mat m_in = imread("../in.jpg", IMREAD_UNCHANGED );
    //runCpu(&m_in, ImageFormat::RGB, "../output/cpu/");
    runGpu(&m_in, ImageFormat::RGB, "../output/gpu/");


    return 0;
}
