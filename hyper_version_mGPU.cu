
#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <matrix.h>
#include<omp.h>
#include <iostream>
#include "error_util.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <algorithm>
using namespace std;

const int FILTER_NUM = 20;
const int COV_LEN = 19;
const int STRIDE = 2;
const int POOLONG_LEN = 2;
const int NEU_NUM1 = 100;
const int NEU_NUM2 = 13;
const int NEIGHBOR = 8;
double learning_rate = 0.08;
const double MIN_ERR = 0.001;
const int VALID_BATCH = 5;
const int DATA_BATCH =400;
int gpu[4];
//Initialize CUDA
bool InitCUDA(){
	int count;
	cudaGetDeviceCount(&count);
	if(count==0){
		fprintf(stderr,"There is no device.\n");
		return false;
	}
	int i;
	for (i =0; i<count;i++){
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){
			if(prop.major>=1){       
                break;
			}
		}
	}
	if(i==count){
		fprintf(stderr,"There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}

struct Tensor{
    int length;
    double * data_h;
    double * data_d;

    Tensor();
};

Tensor::Tensor()
{
    length = 0;
    data_h = NULL;
    data_d = NULL;
}

struct DataLayer{
    Tensor input;
    Tensor labels;

    void initData(int input_size, int labels_size);
    ~DataLayer();
};

void DataLayer::initData(int input_size, int labels_size)
{
    input.length = input_size;
    labels.length = labels_size;

    checkCudaErrors(cudaMalloc((void **)&input.data_d, sizeof(double) * input.length));
    checkCudaErrors(cudaMalloc((void **)&labels.data_d, sizeof(double) * labels.length));
}

DataLayer::~DataLayer()
{
    if(input.data_h != NULL)
        delete [] input.data_h;
    if(labels.data_h != NULL)
        delete [] labels.data_h;
    if(input.data_d != NULL)
        cudaFree(input.data_d);
    if(labels.data_d != NULL)
        cudaFree(labels.data_d);
}

struct Layer{
    Tensor input;
    Tensor output;
    Tensor weights;
    Tensor bias;
    Tensor deltaW;
    Tensor deltaB;

    void initLayer(int input_size, int weights_size, int bias_size, int output_size, int batch_size, bool isMaxpooling, bool copyback);
    ~Layer();

private:
    void allocMemcpyCuda(int size, double ** data_h, double ** data_d, bool isMalloc, bool isCopyback);
};

void Layer::allocMemcpyCuda(int size, double **data_h, double **data_d, bool isMallochost, bool isInitalize)
{
    size_t sizeBytes = size * sizeof(double);
    checkCudaErrors(cudaMalloc(data_d, sizeBytes));

    if (isMallochost)
    {
        *data_h = new double [size];

        if (isInitalize)
        {
            for (int i = 0; i < size; i ++)
            {
                data_h[0][i] = (2 * (rand()/double(RAND_MAX)) - 1) / 20;
            }
            checkCudaErrors(cudaMemcpy(*data_d, *data_h, sizeBytes, cudaMemcpyHostToDevice));
        }
    }
}

void Layer::initLayer (int input_size, int weights_size, int bias_size, int output_size, int batch_size, bool isMaxpooling, bool copyback)
{
    input.length = input_size * batch_size;
    weights.length = weights_size;
    deltaW.length = weights_size * batch_size;
    output.length = output_size * batch_size;
    bias.length = bias_size;
    deltaB.length = bias_size * batch_size;

    if ( isMaxpooling )
    {
        bias.length = bias_size * batch_size;
    }

    allocMemcpyCuda(input.length, &input.data_h, &input.data_d, false, false);
    allocMemcpyCuda(weights.length, &weights.data_h, &weights.data_d, true, true);
    allocMemcpyCuda(bias.length, &bias.data_h, &bias.data_d, true, true);
    allocMemcpyCuda(output.length, &output.data_h, &output.data_d, copyback, false);
    allocMemcpyCuda(deltaB.length, &deltaB.data_h, &deltaB.data_d, false, false);
    allocMemcpyCuda(deltaW.length, &deltaW.data_h, &deltaW.data_d, false, false);
   
    if (isMaxpooling )
       checkCudaErrors(cudaMemset(deltaW.data_d, 0, sizeof(double) * deltaW.length)); 
}

Layer::~Layer ()
{
    if ( input.data_h != NULL )
        delete [] input.data_h;
    if ( weights.data_h != NULL )
        delete [] weights.data_h;
    if ( output.data_h != NULL )
        delete [] output.data_h;
    if ( bias.data_h != NULL )
        delete [] bias.data_h;
    if ( deltaW.data_h != NULL )
        delete [] deltaW.data_h;
    if ( deltaB.data_h != NULL )
        delete [] deltaB.data_h;
    if ( input.data_d != NULL )
        cudaFree(input.data_d);
    if ( output.data_d != NULL )
        cudaFree(output.data_d);
    if ( weights.data_d != NULL )
        cudaFree(weights.data_d);
    if ( bias.data_d != NULL )
        cudaFree(bias.data_d);
    if ( deltaW.data_d != NULL )
        cudaFree(deltaW.data_d);
    if ( deltaB.data_d != NULL )
        cudaFree(deltaB.data_d);
}

// copy data to shared memory
__device__ void copy_data_to_shared( double * data, double * data_tmp, int tid, int offset, int head, int length )
{
	for ( size_t i = tid * offset; i < (tid + 1) * offset && (i < length); i ++ )
    {
		data_tmp[i] = data[i + head];
	}
	__syncthreads();

}
__global__ void do_nothing(){
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// forward propagation kernels
// forward convolution
__global__ static void convolution( int data_id,
                                    int batch_id,
                                    int perLayerSize,
                                    int cube_len,
                                    int conv_len,
                                    int filter_num,
                                    int stride,
                                    double * input, 
                                    double * filters, 
                                    double * bias,
                                    double * output )
{

	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int output_size = (cube_len - conv_len) / stride + 1;

	if ( tid < output_size && bid < filter_num )
	{
		extern __shared__ double tmp[];
        int cube_size = cube_len * perLayerSize;
        int head = data_id * cube_size;
		int length = cube_size;
        int offset = (length - 1) / output_size + 1;
		copy_data_to_shared(input, tmp, tid, offset, head, length);
        
        int filterSize = conv_len * perLayerSize;
		head = bid * filterSize;
		length = filterSize;
		offset = (length - 1) / output_size + 1;
		copy_data_to_shared(filters, tmp + cube_size, tid, offset, head, length);
		__syncthreads();

		double mid = 0;
		for ( int i = 0; i < filterSize; i ++ ) {
            mid = mid + tmp[i + cube_size] * tmp[tid * perLayerSize * stride + i];
		}
		mid = mid + bias[bid];

        output[tid + bid * output_size + batch_id * output_size * filter_num] = 2 / (1 + (1 / exp(2 * mid))) - 1;
	}
}

// forward maxpooling
__global__ static void maxpooling( int batch_id,
                                   int input_size,
                                   int pooling_size,
                                   int group_num,
                                   double * input,
                                   double * output, 
                                   double * output_index )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int output_size = (input_size - 1) / pooling_size + 1;

	if ( tid < output_size && bid < group_num )
    {
		double max;
		int max_index, head, tail;
		head = tid * pooling_size + bid * input_size + batch_id * input_size * group_num;
		tail = (tid + 1) * pooling_size + bid * input_size + batch_id * input_size * group_num;
		max = input[head];
		max_index = head;

		for ( int i=head; i < tail && (i < (bid + 1) * input_size + batch_id * input_size * group_num); i ++ )
        {
			if ( max < input[i] ) {
				max = input[i];
				max_index=i;
			}
		}

		output[tid + bid * output_size + batch_id * output_size * group_num] = max;
		output_index[tid + bid * output_size + batch_id * output_size * group_num] = max_index;
	}
}

// forward fully connection
__global__ static void fully_connect( int batch_id,
                                      int input_size,
                                      int neuron_num,
                                      double * input,
                                      double * weights,
                                      double * bias,
                                      double * output )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if ( tid < input_size && bid < neuron_num )
    {
		extern __shared__ double tmp[];
		tmp[tid] = weights[bid + tid * neuron_num] * input[tid + batch_id * input_size];
		__syncthreads();

		int length = input_size;
		int offset = (length - 1) / 2 + 1;

		while ( length >= 2 )
		{
			if( tid + offset < length )
			{
				tmp[tid] = tmp[tid] + tmp[tid + offset];
			}
			offset = (offset - 1) / 2 + 1;
			length = (length - 1) / 2 + 1;
			__syncthreads();
		}

        if ( tid < 1 )
		    output[bid + batch_id * neuron_num] = 2 / (1 + 1 / exp((tmp[0] + bias[bid]) * 2)) - 1;
	}
}

// forward output
__global__ static void output_and_dvalue( int data_id,
                                          int batch_id,
                                          int input_size,
                                          int neuron_num, 
                                          bool isBackwardPropagation,
                                          double * input, 
                                          double * weights, 
                                          double * bias, 
                                          double * output,
                                          double * labels,
                                          double * dValue )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid < neuron_num )
    {
		// copy to shared memory
		extern __shared__ double temp[];
		int offset = (input_size - 1) / neuron_num + 1;
		copy_data_to_shared(input, temp, tid, offset, batch_id * input_size, input_size);
		__syncthreads();

		double mid = 0;
		for ( int i = 0; i < input_size; i ++ ) {
			mid = mid + weights[tid + i * neuron_num] * temp[i];
		}

		temp[tid + input_size] = exp(mid + bias[tid]);
		__syncthreads(); 

		int length = neuron_num;
		offset = (length - 1) / 2 + 1;
		while(length >= 2)
		{
			if(tid + offset < length){
				temp[tid + input_size] = temp[tid + input_size] + temp[tid + input_size + offset];
			}
			offset = (offset - 1) / 2 + 1;
			length = (length - 1) / 2 + 1;
			__syncthreads();
		}

		output[tid + batch_id * neuron_num] = exp(mid + bias[tid]) / temp[input_size];
        
        if ( isBackwardPropagation )
            dValue[tid + batch_id * neuron_num] = (output[tid + batch_id * neuron_num] - labels[tid + data_id * neuron_num]) / neuron_num;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// backward propagation kernels
// output layer
/*__global__ static void bp_output( int batch_id,
                                  int input_size,
                                  int output_size, 
                                  double * weights, 
                                  double * deltaB, 
                                  double * deltaW,
                                  double * data, 
                                  double * fol_deltaZ )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if ( tid < output_size && bid < input_size ) {
		extern __shared__ double delta_A[];
		delta_A[tid] = weights[tid + bid * output_size] * deltaB[tid + batch_id * output_size];
		__syncthreads();

		deltaW[tid + bid * output_size + batch_id * input_size * output_size] = data[bid + batch_id * input_size] * deltaB[tid + batch_id * output_size]; 

		int length = output_size;
		int offset = (length - 1) / 2 + 1;
		while ( length >= 2 ) 
        {
			if(tid + offset < length){
				delta_A[tid] = delta_A[tid] + delta_A[tid + offset];
			}
			length = (length - 1)/2 + 1;
			offset = (offset - 1)/2 + 1;
			__syncthreads();
		}

        if ( tid < 1 )
		    fol_deltaZ[bid + batch_id * input_size] = delta_A[0] * 
                                                      (1 + data[bid + batch_id * input_size]) * 
                                                      (1 - data[bid + batch_id * input_size]);
	}
}*/

// fully_connect layer
__global__ static void bp_fully_connect( int batch_id, 
                                         int input_size,
                                         int output_size,
                                         double * weights,
                                         double * deltaB,
                                         double * deltaW,
                                         double * data,
                                         //double * data_index,
                                         double * fol_deltaZ )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if( tid < output_size && bid < input_size )
    {
		extern __shared__ double mid[];
		mid[tid] = weights[tid + bid * output_size] * deltaB[tid + batch_id * output_size];
		__syncthreads();

		int length = output_size;
		int offset = (length - 1)/2 + 1;
		while ( length >= 2 ) {
			if(tid + offset < length){
				mid[tid] = mid[tid] + mid[tid+offset];
			}
			length = (length - 1) / 2 + 1;
			offset = (offset - 1) / 2 + 1;
			__syncthreads();
		}

		deltaW[tid + bid * output_size + batch_id * input_size * output_size] = data[bid + batch_id * input_size] * deltaB[tid + batch_id * output_size];
        
        if ( tid < 1 )
            fol_deltaZ[bid + batch_id * input_size] = mid[0] * (1 + data[bid + batch_id * input_size]) * (1 - data[bid + batch_id * input_size]);
	}
}

__global__ static void bp_maxpooling( int batch_id,
                                      int input_size,
                                      int output_size,
                                      double * bias, 
                                      double * deltaB, 
                                      double * fol_deltaZ )
{
    int tid = threadIdx.x;
    
    if ( tid < output_size )
    {
        int idx = (int)(bias[tid + batch_id * output_size]);

        fol_deltaZ[idx] = deltaB[tid + batch_id * output_size];
	
    }

}

// convolutional layer
__global__ static void bp_convolution( int data_id, 
                                       int batch_id,  
                                       int stride,
                                       int perLayerSize, 
                                       int cube_len,
                                       int filter_size,
                                       int filter_num,
                                       int output_size,
                                       double * pre_deltaB, 
                                       double * deltaW, 
                                       double * deltaB, 
                                       double * data,
                                       double * output )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

    int re_size = output_size / filter_num;
	if ( tid < filter_size && bid < filter_num)
	{
        int cube_size = cube_len * perLayerSize;
		int head = data_id * cube_size;
		int length = cube_size;
		int offset = (length - 1) / filter_size + 1;
		extern __shared__ double data_tmp[];
		copy_data_to_shared(data, data_tmp, tid, offset, head, length);
        __syncthreads();

		double mid0 = 0, mid1 = 0;
		for ( int i = 0; i < re_size; i++ ) {
            mid0 = mid0 + pre_deltaB[i + bid * re_size + batch_id * output_size]*(1 - output[i + bid * re_size + batch_id * output_size]) * (1 + output[i + bid * re_size + batch_id * output_size]) * data_tmp[tid + i * perLayerSize * stride];
            mid1 = mid1 + pre_deltaB[i + bid * re_size + batch_id * output_size] * (1 - output[i + bid * re_size + batch_id * output_size]) * (1 + output[i + bid * re_size + batch_id * output_size]);
		}

		deltaW[tid + bid * filter_size + batch_id * filter_size * filter_num] = mid0 / re_size;
		
		if ( tid < 1 )
			deltaB[bid + batch_id * filter_num] = mid1 / re_size;
	}
}
__global__ static void avg_fully_connect(int batch_size,
	int input_size,
	int output_size,
	double *deltaW,
	double *deltaB){
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	double temp0=0, temp1=0;
	if (tid < output_size&&bid < input_size){
		for (int i = 0; i < batch_size; i++){
			temp0 += deltaW[i*input_size*output_size + bid*output_size + tid];
		}
		deltaW[bid*output_size + tid] = temp0;
	}
	if (tid < output_size&&bid <1)
	{
		for (int i = 0; i < batch_size; i++){
			temp1 += deltaB[i*output_size + tid];
		}
		deltaB[tid] = temp1;
	}

}
__global__ static void avg_convolution(int batch_size,
	int filter_size,
	int filter_num,
	double *deltaW,
	double *deltaB){
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	double temp0 = 0, temp1 = 0;
	if (tid < filter_size&&bid < filter_num){
		for (int i = 0; i < batch_size; i++)
		{
			temp0 += deltaW[i*filter_num*filter_size + bid*filter_size + tid];
		}
		deltaW[bid*filter_size + tid] = temp0;
		if (tid < 1){
			for (int i = 0; i < batch_size; i++)
			{
				temp1+= deltaB[i*filter_num + bid];
			}
			deltaB[bid] = temp1;
		}
	}

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// update params kernels
__global__ static void update_fully_connect( int device_choosed_num,
											 int batch_size, 
                                             int input_size,
                                             int output_size,
											 int total_size,
                                             double lr,
											 double * delta,
                                             double * weights, 
											 int w_offset,
                                             double * bias,
                                             int b_offset )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    if ( tid < output_size && bid < input_size )
    {
        double tmp0 = 0, tmp1 = 0;
        for ( int i = 0; i < device_choosed_num; i ++ )
        {
            tmp0 = tmp0 + delta[tid + bid * output_size + i * total_size+w_offset]; 
        }

        weights[tid + bid * output_size] = weights[tid + bid * output_size] - lr * tmp0 / batch_size;

        if ( bid < 1 ) {
            for ( int j = 0; j < device_choosed_num; j++ )
				tmp1 = tmp1 + delta[tid + j *total_size + b_offset];

            bias[tid] = bias[tid] - lr * tmp1 / batch_size;
        }
    }
}

// convolution layer
__global__ static void update_convolution( int device_choosed_num,
										   int batch_size, 
                                           int filter_size,
                                           int filter_num,
										   int total_size,
                                           double lr, 
										   double * delta,
                                           double * filters, 
										   int w_offset,
                                           double * bias,
										   int b_offset)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if ( tid < filter_size && bid < filter_num )
	{
		double tmp0 = 0, tmp1 = 0;
		for ( int i = 0; i < device_choosed_num; i ++ ) 
        {
			tmp0 = tmp0 + delta[tid + bid * filter_size + i *total_size+w_offset];
			tmp1 = tmp1 + delta[bid + i *total_size + b_offset];
		}

		filters[tid + bid * filter_size] = filters[tid + bid * filter_size] - lr * tmp0 / batch_size;
		
		if ( tid < 1 ) {
			bias[bid] = bias[bid] - lr * tmp1 / batch_size;
		}	
	}
}

__global__ static void loss_function(int batch_id, 
                                     int sub_batch_size,
                                     int train_size, 
                                     int output_size,
                                     double * output, 
                                     double * labels, 
                                     double * loss_values)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tid < sub_batch_size ) 
    {
        extern __shared__ double tmp[];
        tmp[tid] = 0;
		for (size_t i = 0; i < output_size; i++)
		{
			tmp[tid] += labels[i + (batch_id * sub_batch_size + tid) * output_size] * log(output[i + tid * output_size]) +
				(1 - labels[i + (batch_id * sub_batch_size + tid) * output_size]) * log(1 - output[i + tid * output_size]);
		}
        __syncthreads();
        int length = sub_batch_size;
        int offset = (length - 1) / 2 + 1;
        while ( length >= 2 )
        {
            if ( tid + offset < length )
            {
                tmp[tid] = tmp[tid] + tmp[tid + offset];
            }
            length = (length - 1) / 2 + 1;
            offset = (offset - 1) / 2 + 1;
            __syncthreads();
        }
        if ( tid < 1 )
            loss_values[0] = - tmp[0] / output_size;
    }
}
//preprocessing
__global__ static void preprocessing(int iter, double * data, int * train_index, double * processed_data, int x, int y, int z, int train_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	if (id < train_size){
		int idx = id * (NEIGHBOR+1) * z;
		int i, j;
		for (i=0; i<z; i++){
			for (j=0; j<(NEIGHBOR+1); j++){
				processed_data[idx] = data[train_index[j + id*(NEIGHBOR+1)] + i * x*y];
				idx = idx + 1;	
			}
		}
	}
}

// compute correct rate
double count_err(double * test_labels, double * output, int test_idx)
{
	double right=0;
	double max =0;
	int idx = 0;
	
	for ( int i = 0; i < NEU_NUM2; i ++ )
    {
		if ( output[i] > max )
        {
			max = output[i];
			idx = i;
		}
	}
	if ( (idx + 1) == int(test_labels[test_idx]) )
		right = 1;
	
	return right;
}

// Insert current loss value to the queue
void insert_line(double * a, double b){
	for(int i=1; i<VALID_BATCH; i++){
		a[i-1] = a[i];
	}
	a[VALID_BATCH-1] = b;
}

// shuffle
void shuffle(int * data, double * labels, int dim_row, int width){
	int index,  i;
	int temp;
	double tmp;
	srand(time(NULL));
	for ( i = 0; i < width; i ++ ) {
		index=rand()%(width-i) + i;
		if ( index != i ) {
			for ( int j = 0; j < dim_row; j ++ ) {
				temp = data[j + i * dim_row];
				data[j + i * dim_row] = data[j + index * dim_row];
				data[j + index * dim_row] = temp;
			}

			for ( int j = 0; j < NEU_NUM2; j ++ ) {
				tmp = labels[j + i * NEU_NUM2];
				labels[j + i * NEU_NUM2] = labels[j + index * NEU_NUM2];
				labels[j + index * NEU_NUM2] = tmp;
			}
		}
	}
}

inline void PeerToPeerMemcpy(int dev_src, double * data_src, int dev_dest, double * data_dest, int size)
{
     cudaMemcpyPeerAsync(data_dest, dev_dest, data_src, dev_src, size*sizeof(double));
    /*double * temp=new double[size] ;
	//checkCublasErrors(cudaMallocHost(&temp, size* sizeof(double)));
    cudaSetDevice(dev_src);
    checkCudaErrors(cudaMemcpy(temp, data_src, size * sizeof(double), cudaMemcpyDeviceToHost));
    cudaSetDevice(dev_dest);
    checkCudaErrors(cudaMemcpy(data_dest, temp, size * sizeof(double), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaFreeHost(temp));
	delete[]temp;*/
}

double training(double * data, double * labels, int x, int y, int z, int device_choosed_num, int master_choosed)
{
	clock_t start, end;
	start = clock();	
	double * gpu_data;//original hyperspectral image, saved in global memory
	int * gpu_train_index;//index of train samples and their neighbors
	int * gpu_test_index;//index of test samples
	
	//preprocessing
	int data_size = 0;
	int * data_index = new int [x * y];
	for ( int i = 0; i < x * y; i ++ ) {
		if ( labels[i] != 0 ) {
			data_index[data_size]=i;
			data_size ++;
		}
	}
	int test_size = (data_size - 1) / 5 + 1;
	int train_size = data_size - test_size;
	int * train_index = new int [train_size * (NEIGHBOR + 1)];
	int * test_index = new int [test_size * (NEIGHBOR + 1)];

	double * processed_labels = new double [train_size * NEU_NUM2]();
	double * test_labels = new double [test_size]();

	int tr=0, te=0;
	for ( int i = 0; i < data_size; i ++ ) {
		if ( i % 5 != 0 ) {
			train_index[(NEIGHBOR / 2) + tr * (NEIGHBOR + 1)] = data_index[i]; // index of current labeled pixel
			if ( NEIGHBOR == 4 )
			{
				train_index[(NEIGHBOR / 2) + tr * (NEIGHBOR + 1) - 1] = data_index[i] - 1;
				train_index[(NEIGHBOR / 2) + tr * (NEIGHBOR + 1) + 1] = data_index[i] + 1;
				train_index[0 + tr * (NEIGHBOR + 1)] = data_index[i] - x;
				train_index[NEIGHBOR + tr * (NEIGHBOR + 1)] = data_index[i] + x;
				

				if ( (data_index[i] % x) == 0 ) { // first row
					train_index[(NEIGHBOR / 2) + tr * (NEIGHBOR + 1) - 1] = train_index[(NEIGHBOR / 2) + tr * (NEIGHBOR + 1) + 1];
				}
				if ( (data_index[i] % x ) == (x-1) ) { // last row
					train_index[(NEIGHBOR / 2) + tr * (NEIGHBOR + 1) + 1] = train_index[(NEIGHBOR / 2) + tr * (NEIGHBOR + 1) - 1];
				}
				if ( (data_index[i] / x) == 0 ) { // first column
					train_index[0 + tr * (NEIGHBOR + 1)] = train_index[NEIGHBOR + tr * (NEIGHBOR + 1)];
				}
				if ( (data_index[i] / x) == (y-1) ) {//last column
					train_index[NEIGHBOR + tr * (NEIGHBOR + 1)] = train_index[0 + tr * (NEIGHBOR + 1)];
				}
			}
			if ( NEIGHBOR == 8 )
			{
				train_index[(NEIGHBOR / 2) + tr * (NEIGHBOR + 1) - 1] = data_index[i] - 1;
				train_index[(NEIGHBOR / 2) + tr * (NEIGHBOR + 1) + 1] = data_index[i] + 1;
				for ( int j0 = 0; j0 < 3; j0 ++ ) {
					train_index[j0 + tr * (NEIGHBOR + 1)] = data_index[i] - 1 - x + j0;
					train_index[j0 + 6 + tr * (NEIGHBOR + 1)] = data_index[i] - 1 + x + j0;
				}

				if ( (data_index[i] % x) == 0 ) { // first row
					for (int j = 0; j < 3; j ++ )
						train_index[j * 3 + tr * (NEIGHBOR + 1)] = train_index[j * 3 + 2 + tr * (NEIGHBOR + 1)];
				}
				if ( (data_index[i] % x) == (x - 1) ) { // last row
					for ( int j = 0; j < 3; j ++ )
							train_index[j * 3 + 2 + tr * (NEIGHBOR + 1)] = train_index[j * 3 + tr * (NEIGHBOR + 1)];
				}
				if ( (data_index[i] / x) == 0 ) { // first column
					for ( int j = 0; j < 3; j ++ )
						train_index[j + tr * (NEIGHBOR + 1)] = train_index[j + 6 + tr*(NEIGHBOR + 1)];
				}
				if ( (data_index[i] / x) == (y - 1) ) { // last column
					for ( int j = 0; j < 3; j ++ )
						train_index[j + 6 + tr * (NEIGHBOR + 1)] = train_index[j + tr * (NEIGHBOR + 1)];
				}
			}

			int mid = int(labels[data_index[i]]) - 1 + tr * NEU_NUM2;
			processed_labels[mid] = 1;
			tr = tr + 1;
		}
		if( i % 5 == 0 ) {
			test_index[(NEIGHBOR / 2) + te * (NEIGHBOR + 1)] = data_index[i]; // index of current labeled pixel
			if ( NEIGHBOR == 4 )
			{
				test_index[(NEIGHBOR / 2) + te * (NEIGHBOR + 1) - 1] = data_index[i] - 1;
				test_index[(NEIGHBOR / 2) + te * (NEIGHBOR + 1) + 1] = data_index[i] + 1;
				test_index[0 + te * (NEIGHBOR + 1)] = data_index[i] - x;
				test_index[NEIGHBOR + te * (NEIGHBOR + 1)] = data_index[i] + x;

				if ( (data_index[i] % x) == 0 ) { // first row
					test_index[(NEIGHBOR / 2) + te * (NEIGHBOR + 1) - 1] = test_index[(NEIGHBOR / 2) + te * (NEIGHBOR + 1) + 1];
				}
				if ( (data_index[i] % x) == (x - 1) ) { // last row
					test_index[(NEIGHBOR / 2) + te * (NEIGHBOR + 1) + 1] = test_index[(NEIGHBOR / 2) + te * (NEIGHBOR + 1) - 1];
				}
				if ( (data_index[i] / x) == 0 ) { // first column
					test_index[0 + te * (NEIGHBOR + 1)] = test_index[NEIGHBOR+ te * (NEIGHBOR + 1)];
				}
				if ( (data_index[i] / x) == (y - 1) ) { // last column
					test_index[NEIGHBOR + te * (NEIGHBOR + 1)] = test_index[0 + te * (NEIGHBOR + 1)];
				}
			}
			if ( NEIGHBOR == 8 )
			{
				test_index[(NEIGHBOR / 2) + te * (NEIGHBOR + 1) - 1] = data_index[i] - 1;
				test_index[(NEIGHBOR / 2) + te * (NEIGHBOR + 1) + 1] = data_index[i] + 1;
				for ( int j0 = 0; j0 < 3; j0 ++ ) {
					test_index[j0 + te * (NEIGHBOR + 1)] = data_index[i] - 1 - x + j0;
					test_index[j0 + 6 + te * (NEIGHBOR + 1)] = data_index[i] - 1 + x + j0;
				}

				if ( (data_index[i] % x) == 0 ) { // first row
					for ( int j = 0; j < 3; j ++ )
						test_index[j * 3 + te * (NEIGHBOR + 1)] = test_index[j * 3 + 2 + te * (NEIGHBOR + 1)];
				}
				if ( (data_index[i] % x) == (x - 1) ) { // last row
					for ( int j = 0; j < 3; j ++ )
						test_index[j * 3 + 2 + te * (NEIGHBOR + 1)] = test_index[j * 3 + te * (NEIGHBOR + 1)];
				}
				if ( (data_index[i] / x) == 0 ) { // first column
					for ( int j = 0; j < 3; j ++ )
						test_index[j + te * (NEIGHBOR + 1)] = test_index[j + 6 + te * (NEIGHBOR + 1)];
				}
				if ( (data_index[i] / x) == (y - 1) ) { // last column
					for ( int j = 0;j < 3; j ++ )
						test_index[j + 6 + te * (NEIGHBOR + 1)] = test_index[j + te * (NEIGHBOR + 1)];
				}
			}

			test_labels[te] = labels[data_index[i]];
			te = te + 1;
		}
	}

	shuffle(train_index, processed_labels, (NEIGHBOR + 1), train_size);//shuffle the samples in training set
    cudaSetDevice(gpu[master_choosed]);
    DataLayer dataLayer[device_choosed_num];
    dataLayer[master_choosed].initData(train_size * (NEIGHBOR + 1) * z, train_size * NEU_NUM2);
	//double *train_data;
	//double *train_label;
	//checkCudaErrors(cudaMalloc(*train_data,)
    DataLayer testData;
    testData.initData(test_size * (NEIGHBOR + 1) * z, 0);

	//malloc GPU memory, copy data to GPU
	checkCudaErrors(cudaMalloc((void **) &gpu_data, sizeof(double) * x * y * z));
	checkCudaErrors(cudaMemcpy(gpu_data, data, sizeof(double)* x * y * z, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &gpu_train_index, sizeof(int) * train_size * (NEIGHBOR + 1)));
	checkCudaErrors(cudaMemcpy(gpu_train_index, train_index, sizeof(int) * train_size * (NEIGHBOR + 1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **) &gpu_test_index, sizeof(int) * test_size * (NEIGHBOR + 1)));
	checkCudaErrors(cudaMemcpy(gpu_test_index, test_index, sizeof(int) * test_size * (NEIGHBOR + 1), cudaMemcpyHostToDevice));

    delete [] data_index;
    delete [] train_index;
    delete [] test_index;

    int gridsize =128;
    int blocksize = 512;
	int iter=0;

	preprocessing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_train_index, dataLayer[master_choosed].input.data_d, x, y, z, train_size);
	preprocessing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_test_index, testData.input.data_d, x, y, z, test_size);
	//cudaDeviceSynchronize();
	end = clock();
	float tt = float(end - start);
	fprintf(stdout,"[Samples prepared with %d Nearest-Neighbor-Pixels Strategy] Proportion of Training Samples: %d%%  Execution time: %.3f sec\n", 
            NEIGHBOR, 80, tt/CLOCKS_PER_SEC);
	checkCudaErrors(cudaFree(gpu_data));
	checkCudaErrors(cudaFree(gpu_train_index));
	checkCudaErrors(cudaFree(gpu_test_index));
	
	// calculate length of convolutional results
	int re_size = 0;
	for ( int i = 0; i + COV_LEN-1 < z; i += STRIDE )
    {
		re_size ++;
	}
	int mre_size = (re_size - 1) / POOLONG_LEN + 1;
    int pooling_input_length = re_size * FILTER_NUM;
    int pooling_output_length = mre_size * FILTER_NUM;
	//int ful_weights_size = pooling_output_length * NEU_NUM1;    // SAVE DATA
	//int out_weights_size = NEU_NUM1 * NEU_NUM2;                 // SAVE DATA
    int filter_size = (NEIGHBOR + 1) * COV_LEN;
    int cube_size = (NEIGHBOR + 1) * z;
	
    double * gpu_loss_values;

	// copy labels to GPU
	//checkCudaErrors(cudaMemcpy(dataLayer[master_choosed].labels.data_d, processed_labels, sizeof(double) * train_size * NEU_NUM2, 
    //            cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dataLayer[master_choosed].labels.data_d , processed_labels, sizeof(double) * train_size * NEU_NUM2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaHostAlloc((void **)&gpu_loss_values, sizeof(double)*device_choosed_num, cudaHostAllocPortable));
    delete [] processed_labels;
	
    double loss;
    double * logloss = new double [300]();
    double * loss_values = new double [DATA_BATCH];
	double * correct_rate = new double [VALID_BATCH];
    for ( int i = 0; i < VALID_BATCH; i ++ ) {
    	correct_rate[i] = 1;
    }

 	double cur_min = 1;
	int count = 1;
	int batch_size = DATA_BATCH;
    int sub_batch_size = batch_size / device_choosed_num;
	int batch_num = train_size / batch_size;

	start = clock();

    Layer conv[device_choosed_num];
    conv[master_choosed].initLayer( cube_size, filter_size * FILTER_NUM, FILTER_NUM, pooling_input_length, DATA_BATCH, false, false);

    Layer pooling[device_choosed_num];
	pooling[master_choosed].initLayer(pooling_input_length, pooling_input_length, pooling_output_length, pooling_output_length, DATA_BATCH, true, false);
    
    Layer fulconnect[device_choosed_num];
	fulconnect[master_choosed].initLayer(pooling_output_length, pooling_output_length * NEU_NUM1, NEU_NUM1, NEU_NUM1, DATA_BATCH, false, false);

    Layer out[device_choosed_num];
	out[master_choosed].initLayer(NEU_NUM1, NEU_NUM1 * NEU_NUM2, NEU_NUM2, NEU_NUM2, DATA_BATCH, false, true);
   
    // creat CUDA streams
	cudaStream_t stream[DATA_BATCH];
	for (int i = 1;i < device_choosed_num; i++){
		cudaSetDevice(gpu[i]);
		cudaDeviceEnablePeerAccess(gpu[master_choosed],0);
		cudaSetDevice(gpu[master_choosed]);
		cudaDeviceEnablePeerAccess(gpu[i], 0);
	}
	cudaStream_t avg[device_choosed_num];
	for(int i = 0; i < device_choosed_num; i++){
	cudaSetDevice(gpu[i]);
	cudaStreamCreate(&avg[i]);
	}
	cudaStream_t update[3];
	cudaSetDevice(gpu[master_choosed]);
	for(int i = 0; i < 3; i++){
	cudaStreamCreate(&update[i]);
	}
	int conv_wsize = filter_size * FILTER_NUM, conv_bsize = FILTER_NUM;
	int fc1_wsize = pooling_output_length * NEU_NUM1, fc1_bsize = NEU_NUM1;
	int fc2_wsize = NEU_NUM1 * NEU_NUM2, fc2_bsize = NEU_NUM2;
	int total_size = conv_wsize + conv_bsize + fc1_wsize + fc1_bsize + fc2_wsize + fc2_bsize;
	double *tmp_param;
	cudaSetDevice(gpu[master_choosed]);
	cudaMalloc(&tmp_param, device_choosed_num*total_size*sizeof(double));
    int sub = 1;
    for ( int i = 0; i < device_choosed_num; i ++ )
    {
        cudaSetDevice(gpu[i]);
        for ( int j = 0; j < sub_batch_size; j ++ )
        {
            cudaStreamCreate(&stream[i * sub_batch_size + j]);
        }

        if ( i != master_choosed ) 
        {
            dataLayer[i].initData(train_size / device_choosed_num * cube_size, train_size / device_choosed_num * NEU_NUM2);
            PeerToPeerMemcpy(gpu[master_choosed], dataLayer[master_choosed].input.data_d + (train_size / device_choosed_num * cube_size) * sub, 
                             gpu[i], dataLayer[i].input.data_d, 
                             train_size / device_choosed_num * cube_size);
            PeerToPeerMemcpy(gpu[master_choosed], dataLayer[master_choosed].labels.data_d + (train_size / device_choosed_num * NEU_NUM2) * sub, 
                             gpu[i], dataLayer[i].labels.data_d, 
                             train_size / device_choosed_num * NEU_NUM2);
							 
            conv[i].initLayer(cube_size, filter_size * FILTER_NUM, FILTER_NUM, pooling_input_length, sub_batch_size, false, false);

            pooling[i].initLayer(pooling_input_length, pooling_input_length, pooling_output_length, pooling_output_length, sub_batch_size, true, false);

            fulconnect[i].initLayer(pooling_output_length, pooling_output_length * NEU_NUM1, NEU_NUM1, NEU_NUM1, sub_batch_size, false, false);

            out[i].initLayer(NEU_NUM1, NEU_NUM1 * NEU_NUM2, NEU_NUM2, NEU_NUM2, sub_batch_size, false, true);

        sub ++;    
        }
    }	
	for(int d=0;d<device_choosed_num;d++){
				if ( d != master_choosed){
					PeerToPeerMemcpy(gpu[master_choosed], conv[master_choosed].weights.data_d, gpu[d], conv[d].weights.data_d, filter_size * FILTER_NUM);
					PeerToPeerMemcpy(gpu[master_choosed], conv[master_choosed].bias.data_d, gpu[d], conv[d].bias.data_d, FILTER_NUM);
					PeerToPeerMemcpy(gpu[master_choosed], fulconnect[master_choosed].weights.data_d,gpu[d], fulconnect[d].weights.data_d, pooling_output_length * NEU_NUM1);
					PeerToPeerMemcpy(gpu[master_choosed], fulconnect[master_choosed].bias.data_d, gpu[d], fulconnect[d].bias.data_d, NEU_NUM1);
					PeerToPeerMemcpy(gpu[master_choosed], out[master_choosed].weights.data_d, gpu[d], out[d].weights.data_d, NEU_NUM1 * NEU_NUM2);
					PeerToPeerMemcpy(gpu[master_choosed], out[master_choosed].bias.data_d, gpu[d], out[d].bias.data_d, NEU_NUM2);
                }
		}

	cout << "GPU num is "<<device_choosed_num<<"\n";
	
    int max_iter =10;
	//batch_num=2;
	omp_set_num_threads(device_choosed_num);  // create as many CPU threads as there are CUDA devices
    fprintf(stdout, "[Cube CNN training with MBGD algo.  BatchSize = %d] lr = %lf\n", DATA_BATCH, learning_rate);    
	for ( int iter = 0; iter < max_iter; iter ++ ){
		loss = 0;
        double iter_start = omp_get_wtime();
		for ( int i0 = 0; i0 < batch_num; i0 ++ )
		{
                // forward propagation				
		#pragma omp parallel
		//
			{
				int d = omp_get_thread_num();
				cudaSetDevice(gpu[d]);
				for (int i1 = 0; i1 < sub_batch_size; i1++){				
					convolution << < FILTER_NUM,
						re_size,
						(cube_size + filter_size) * sizeof(double),
						stream[i1 + d * sub_batch_size] >> >(i0 * sub_batch_size + i1,
						i1,
						(NEIGHBOR + 1),
						z,
						COV_LEN,
						FILTER_NUM,
						STRIDE,
						dataLayer[d].input.data_d,
						conv[d].weights.data_d,
						conv[d].bias.data_d,
						conv[d].output.data_d);
					maxpooling << < FILTER_NUM,
						mre_size,
						0,
						stream[i1 + d * sub_batch_size] >> >(i1,
						re_size,
						POOLONG_LEN,
						FILTER_NUM,
						conv[d].output.data_d,
						pooling[d].output.data_d,
						pooling[d].bias.data_d);
					fully_connect << < NEU_NUM1,
						pooling_output_length,
						pooling_output_length * sizeof(double),
						stream[i1 + d * sub_batch_size] >> >(i1,
						pooling_output_length,
						NEU_NUM1,
						pooling[d].output.data_d,
						fulconnect[d].weights.data_d,
						fulconnect[d].bias.data_d,
						fulconnect[d].output.data_d);
					output_and_dvalue << < 1,
						NEU_NUM2,
						(NEU_NUM1 + NEU_NUM2) * sizeof(double),
						stream[i1 + d * sub_batch_size] >> >(i0 * sub_batch_size + i1,
						i1,
						NEU_NUM1,
						NEU_NUM2,
						true,
						fulconnect[d].output.data_d,
						out[d].weights.data_d,
						out[d].bias.data_d,
						out[d].output.data_d,
						dataLayer[d].labels.data_d,
						out[d].deltaB.data_d);
					}
						for (int i1 = 0; i1 < sub_batch_size; i1++)
							bp_fully_connect << < NEU_NUM1,
								NEU_NUM2,
								NEU_NUM2 * sizeof(double),
								stream[i1 + d * sub_batch_size] >> >(i1,
								NEU_NUM1,
								NEU_NUM2,
								out[d].weights.data_d,
								out[d].deltaB.data_d,
								out[d].deltaW.data_d,
								fulconnect[d].output.data_d,
								fulconnect[d].deltaB.data_d);
							do_nothing << <1, 1 >> >();
							avg_fully_connect << < NEU_NUM1, NEU_NUM2,0,avg[d]>> >(sub_batch_size,
								NEU_NUM1,
								NEU_NUM2,
								out[d].deltaW.data_d,
								out[d].deltaB.data_d);
							cudaMemcpyPeerAsync(tmp_param + total_size*d + conv_wsize + conv_bsize + fc1_wsize + fc1_bsize, gpu[master_choosed], out[d].deltaW.data_d, gpu[d], fc2_wsize*sizeof(double),avg[d]);
							cudaMemcpyPeerAsync(tmp_param + total_size*d + conv_wsize + conv_bsize + fc1_wsize + fc1_bsize + fc2_wsize, gpu[master_choosed], out[d].deltaB.data_d, gpu[d], fc2_bsize*sizeof(double),avg[d]);
						for (int i1 = 0; i1 < sub_batch_size; i1++){
							bp_fully_connect << < pooling_output_length,
								NEU_NUM1,
								NEU_NUM1 * sizeof(double),
								stream[i1 + d * sub_batch_size] >> >(i1,
								pooling_output_length,
								NEU_NUM1,
								fulconnect[d].weights.data_d,
								fulconnect[d].deltaB.data_d,
								fulconnect[d].deltaW.data_d,
								pooling[d].output.data_d,
								//pooling[d].bias.data_d,
								pooling[d].deltaB.data_d);}
							do_nothing << <1, 1 >> >();
							avg_fully_connect << < pooling_output_length, NEU_NUM1,0,avg[d]>> >(sub_batch_size,
								pooling_output_length,
								NEU_NUM1,
								fulconnect[d].deltaW.data_d,
								fulconnect[d].deltaB.data_d);
							cudaMemcpyPeerAsync(tmp_param + total_size*d + conv_wsize + conv_bsize, gpu[master_choosed], fulconnect[d].deltaW.data_d, gpu[d], fc1_wsize*sizeof(double),avg[d]);
							cudaMemcpyPeerAsync(tmp_param + total_size*d + conv_wsize + conv_bsize + fc1_wsize, gpu[master_choosed], fulconnect[d].deltaB.data_d, gpu[d], fc1_bsize*sizeof(double),avg[d]);
						for (int i1 = 0; i1 < sub_batch_size; i1++)
							bp_maxpooling << < 1,
								pooling_output_length,
								0,
								stream[i1 + d * sub_batch_size] >> >(i1,
								pooling_input_length,
								pooling_output_length,
								pooling[d].bias.data_d,
								pooling[d].deltaB.data_d,
								pooling[d].deltaW.data_d);
						for (int i1 = 0; i1 < sub_batch_size; i1++)
							bp_convolution << < FILTER_NUM,
								filter_size,
								cube_size * sizeof(double),
								stream[i1 + d * sub_batch_size] >> >(i0 * sub_batch_size + i1,
								i1,
								STRIDE,
								(NEIGHBOR + 1),
								z,
								filter_size,
								FILTER_NUM,
								pooling_input_length,
								pooling[d].deltaW.data_d,
								conv[d].deltaW.data_d,
								conv[d].deltaB.data_d,
								dataLayer[d].input.data_d,
								conv[d].output.data_d);
							do_nothing << <1, 1 >> >();
							avg_convolution << < FILTER_NUM, filter_size,0,avg[d]>> >(sub_batch_size,
								filter_size,
								FILTER_NUM,
								conv[d].deltaW.data_d,
								conv[d].deltaB.data_d);
							cudaMemcpyPeerAsync(tmp_param + total_size*d, gpu[master_choosed], conv[d].deltaW.data_d, gpu[d], conv_wsize*sizeof(double),avg[d]);
							cudaMemcpyPeerAsync(tmp_param + total_size*d + conv_wsize, gpu[master_choosed], conv[d].deltaB.data_d, gpu[d], conv_bsize*sizeof(double),avg[d]);

				loss_function << < 1,
					sub_batch_size,
					sizeof(double) * sub_batch_size >> >(i0,
					sub_batch_size,
					train_size,
					NEU_NUM2,
					out[d].output.data_d,
					dataLayer[d].labels.data_d,
					gpu_loss_values+d);
					
			}

          		for (int j = 0; j < device_choosed_num;j++)
           			 loss += gpu_loss_values[j];
			cudaSetDevice(gpu[master_choosed]);
			//update parameters
			update_fully_connect<<< NEU_NUM1, NEU_NUM2,0,update[0] >>>( device_choosed_num,
															batch_size,
                                                            NEU_NUM1, 
                                                            NEU_NUM2,
															total_size,
                                                            learning_rate, 
															tmp_param,
                                                            out[master_choosed].weights.data_d,
                                                            conv_wsize+conv_bsize+fc1_wsize+fc1_bsize,
                                                            out[master_choosed].bias.data_d, 
															conv_wsize + conv_bsize + fc1_wsize + fc1_bsize+fc2_wsize);
			for(int d=1;d<device_choosed_num;d++){
					//PeerToPeerMemcpy(gpu[master_choosed], out[master_choosed].weights.data_d, gpu[d], out[d].weights.data_d, NEU_NUM1 * NEU_NUM2,update[0]);
					cudaMemcpyPeerAsync(out[d].weights.data_d,gpu[d],out[master_choosed].weights.data_d,gpu[master_choosed],NEU_NUM1 * NEU_NUM2*sizeof(double),update[0]);
					//PeerToPeerMemcpy(gpu[master_choosed], out[master_choosed].bias.data_d, gpu[d], out[d].bias.data_d, NEU_NUM2,update[0]);
					cudaMemcpyPeerAsync(out[d].bias.data_d,gpu[d],out[master_choosed].bias.data_d,gpu[master_choosed],NEU_NUM2*sizeof(double),update[0]);
		}


			update_fully_connect<<< pooling_output_length, NEU_NUM1 ,0,update[1] >>>( device_choosed_num,
																		 batch_size,
                                                                         pooling_output_length,
                                                                         NEU_NUM1, 
																		 total_size,
                                                                         learning_rate, 
																		 tmp_param,
                                                                         fulconnect[master_choosed].weights.data_d, 
																		 conv_wsize + conv_bsize,
                                                                         fulconnect[master_choosed].bias.data_d,
																		 conv_wsize + conv_bsize + fc1_wsize);
			for(int d=1;d<device_choosed_num;d++){
					//PeerToPeerMemcpy(gpu[master_choosed], fulconnect[master_choosed].weights.data_d,gpu[d], fulconnect[d].weights.data_d, pooling_output_length * NEU_NUM1,update[1]);
					cudaMemcpyPeerAsync( fulconnect[d].weights.data_d,gpu[d],fulconnect[master_choosed].weights.data_d,gpu[master_choosed],pooling_output_length * NEU_NUM1*sizeof(double),update[1]);
					//PeerToPeerMemcpy(gpu[master_choosed], fulconnect[master_choosed].bias.data_d, gpu[d], fulconnect[d].bias.data_d, NEU_NUM1,update[1]);
					cudaMemcpyPeerAsync( fulconnect[d].bias.data_d,gpu[d],fulconnect[master_choosed].bias.data_d,gpu[master_choosed],NEU_NUM1*sizeof(double),update[1]);
		}

			update_convolution<<< FILTER_NUM, filter_size,0,update[2]  >>>( device_choosed_num,
															   batch_size,
                                                               filter_size,
                                                               FILTER_NUM,
															   total_size,
                                                               learning_rate,
															   tmp_param, 
                                                               conv[master_choosed].weights.data_d, 
															   0,
                                                               conv[master_choosed].bias.data_d,
															   conv_wsize);
			for(int d=1;d<device_choosed_num;d++){
					//PeerToPeerMemcpy(gpu[master_choosed], conv[master_choosed].weights.data_d, gpu[d], conv[d].weights.data_d, filter_size * FILTER_NUM,update[2]);
					cudaMemcpyPeerAsync(conv[d].weights.data_d, gpu[d], conv[master_choosed].weights.data_d, gpu[master_choosed],filter_size * FILTER_NUM*sizeof(double),update[2]);
					//PeerToPeerMemcpy(gpu[master_choosed], conv[master_choosed].bias.data_d, gpu[d], conv[d].bias.data_d, FILTER_NUM,update[2]);
					cudaMemcpyPeerAsync(conv[d].bias.data_d, gpu[d], conv[master_choosed].bias.data_d, gpu[master_choosed],FILTER_NUM*sizeof(double),update[2]);
		}
	cudaDeviceSynchronize();
           cudaMemset(pooling[master_choosed].deltaW.data_d, 0, sizeof(double) * pooling_input_length * DATA_BATCH);
		
        }//iter
   
        double iter_stop = omp_get_wtime();
        float iter_time = float(iter_stop - iter_start) ;
		double single_rate = loss/train_size;
        logloss[iter] = single_rate;
        char str[50];
        sprintf(str, "%d", iter + 1);
        strcat(str, ",");
		fprintf(stdout,"[Cube CNN training with MBGD algo.  BatchSize = %d  Execution time: %.3f sec] Iteration %-4s loss = %lf;\n", 
                DATA_BATCH, iter_time, str, single_rate);
        	
		insert_line(correct_rate, single_rate); // insert current loss into the line
		double new_min = *min_element(correct_rate, correct_rate + VALID_BATCH);
        if ( cur_min > new_min ) {
           	cur_min = new_min;
		   	count = 1;
        }
        else {
           	count++;
        }
        if ( count >= VALID_BATCH ) {
           	learning_rate = learning_rate * 0.9;
           	fprintf(stdout,"[Cube CNN training with MBGD algo.  BatchSize = %d] lr = %lf\n",
                            DATA_BATCH, learning_rate);
           	count = 1;
            cur_min = new_min;
        }
        if ( single_rate < MIN_ERR )
        	break;
	} // iter

	fprintf(stdout,"[Cube CNN training with MBGD algo.  BatchSize = %d]", DATA_BATCH);
	end = clock();
	tt = double(end - start);
	fprintf(stdout," Completed! Global Exesution time is %.3f sec\n", tt/CLOCKS_PER_SEC);

	start = clock();
	cudaDeviceSynchronize();
	/*checkCudaErrors(cudaMemcpy(conv.weights.data_h, conv.weights.data_d, sizeof(double) * filter_size * FILTER_NUM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(conv.bias.data_h, conv.bias.data_d, sizeof(double) * FILTER_NUM, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fulconnect.bias.data_h, fulconnect.bias.data_d, sizeof(double) * NEU_NUM1, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(out.bias.data_h, out.bias.data_d, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fulconnect.weights.data_h, fulconnect.weights.data_d, sizeof(double) * ful_weights_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(out.weights.data_h, out.weights.data_d, sizeof(double) * out_weights_size, cudaMemcpyDeviceToHost));
	
	// Write the parameters into a mat file
	MATFile * pmatFile;
	pmatFile = matOpen("model/model.mat","w");
	mxArray * m1 = mxCreateDoubleMatrix(filter_size, FILTER_NUM, mxREAL);
	memcpy((void *)mxGetPr(m1), (void *)conv.weights.data_h, sizeof(double) * filter_size * FILTER_NUM);
	matPutVariable(pmatFile, "filters", m1);

	mxArray * m2 = mxCreateDoubleMatrix(FILTER_NUM, 1, mxREAL);
	memcpy((void *)mxGetPr(m2), (void *)conv.bias.data_h, sizeof(double) * FILTER_NUM);
	matPutVariable(pmatFile, "bias0", m2);

	mxArray * m3 = mxCreateDoubleMatrix(NEU_NUM1, pooling_output_length, mxREAL);
	memcpy((void *)mxGetPr(m3), (void *)fulconnect.weights.data_h, sizeof(double) * ful_weights_size);
	matPutVariable(pmatFile, "omega1", m3);

	mxArray * m4 = mxCreateDoubleMatrix(NEU_NUM1, 1, mxREAL);
    memcpy((void *)mxGetPr(m4), (void *)fulconnect.bias.data_h, sizeof(double) * NEU_NUM1);
	matPutVariable(pmatFile, "bias1", m4);

	mxArray * m5 = mxCreateDoubleMatrix(NEU_NUM2, NEU_NUM1, mxREAL);
	memcpy((void *)mxGetPr(m5), (void *)out.weights.data_h, sizeof(double) * out_weights_size);
	matPutVariable(pmatFile, "omega2", m5);

	mxArray * m6 = mxCreateDoubleMatrix(NEU_NUM2, 1, mxREAL);
	memcpy((void *)mxGetPr(m6), (void *)out.bias.data_h, sizeof(double) * NEU_NUM2);
	matPutVariable(pmatFile, "bias2", m6);

    mxArray * m7 = mxCreateDoubleMatrix(300, 1, mxREAL);
    memcpy((void *)mxGetPr(m7), (void *)logloss, sizeof(double) * 300);
    matPutVariable(pmatFile, "loss", m7);

	matClose(pmatFile);*/

    delete [] logloss;
    delete [] loss_values;
    delete [] correct_rate;

	for(int i=0; i<DATA_BATCH; i++){
		cudaStreamDestroy(stream[i]);
	}
	
	// model testing
    cudaSetDevice(gpu[master_choosed]);
	double right = 0;
	double accuracy_count = 0;

	for ( int i1 = 0; i1 < test_size; i1 ++ ) {
		convolution<<< FILTER_NUM, re_size, (cube_size + filter_size) * sizeof(double) >>>( i1,
                                                                                            0,
                                                                                            (NEIGHBOR + 1),
                                                                                            z,
                                                                                            COV_LEN,
                                                                                            FILTER_NUM,
                                                                                            STRIDE,
                                                                                            testData.input.data_d,
                                                                                            conv[master_choosed].weights.data_d,
                                                                                            conv[master_choosed].bias.data_d,
                                                                                            conv[master_choosed].output.data_d );
		//cudaDeviceSynchronize();

		maxpooling<<< FILTER_NUM, mre_size, 0 >>>( 0,
                                                   re_size,
                                                   POOLONG_LEN,
                                                   FILTER_NUM,
                                                   conv[master_choosed].output.data_d, 
                                                   pooling[master_choosed].output.data_d, 
                                                   pooling[master_choosed].bias.data_d );
		//cudaDeviceSynchronize();

		fully_connect<<< NEU_NUM1, pooling_output_length, pooling_output_length * sizeof(double) >>>( 0, 
                                                                                                      pooling_output_length,
                                                                                                      NEU_NUM1,
                                                                                                      pooling[master_choosed].output.data_d, 
                                                                                                      fulconnect[master_choosed].weights.data_d,
                                                                                                      fulconnect[master_choosed].bias.data_d,
                                                                                                      fulconnect[master_choosed].output.data_d );

		output_and_dvalue<<< 1, NEU_NUM2, (NEU_NUM1 + NEU_NUM2) * sizeof(double) >>>( i1,
                                                                                      0,
                                                                                      NEU_NUM1,
                                                                                      NEU_NUM2,
                                                                                      false,
                                                                                      fulconnect[master_choosed].output.data_d,
                                                                                      out[master_choosed].weights.data_d,
                                                                                      out[master_choosed].bias.data_d,
                                                                                      out[master_choosed].output.data_d,
                                                                                      NULL,
                                                                                      NULL );
		//cudaDeviceSynchronize();

		checkCudaErrors(cudaMemcpy(out[master_choosed].output.data_h, 
                                   out[master_choosed].output.data_d, 
                                   sizeof(double) * NEU_NUM2, 
                                   cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		right = count_err(test_labels, out[master_choosed].output.data_h, i1);
		accuracy_count = accuracy_count + right;
	}

    delete [] test_labels;

	end = clock();
	tt = float(end - start);
	fprintf(stdout, "[Cube CNN testing] Done. Execution time: %.3f sec. ", tt/CLOCKS_PER_SEC);
  
    return accuracy_count/test_size;
	//return 0;
}


int main(int argc, char * argv[])
{
    fprintf(stdout, "[Cube CNN training with MBGD algo] ");
  	//if(!InitCUDA()){
	//	return 0;
	//}
	printf("CUDA initialized.\n");

    fprintf(stdout, "[Cube CNN training with MBGD algo] Available Device List: ");
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int device;
    for (device = 0; device < deviceCount; ++ device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        if (device == 0)
            printf("Device %d -- %s(Default)  ", device, deviceProp.name);
        else
            printf("Device %d -- %s  ", device, deviceProp.name);
    }
    cout<<endl;
	gpu[0]= 0; gpu[1] = 1; gpu[2] = 2; gpu[3] = 3;
    if ( argc != 3 ) {
	    fprintf(stderr, "3 input arguments required!\n");
		return 0;
	}

    //int device_choosed_num = (int)atoi(argv[3]);
int device_choosed_num = 4;
    if ( device_choosed_num > deviceCount )
        fprintf(stderr, "Lack of GPU devices!\n");

    int master_choosed = 0;
    //master_choosed = (int)atoi(argv[4]);

    fprintf(stdout, "[Cube CNN training with MBGD algo] Choosing \033[31mDevice %d\033[0m as \033[31mMASTER\033[0m.\n", master_choosed);
    cudaSetDevice(gpu[master_choosed]);

	double *trainset,*trainlabels;

	MATFile * datamat = matOpen(argv[1], "r");
mxArray * train = matGetVariable(datamat,"KSC");
	MATFile * labelmat = matOpen(argv[2], "r");
mxArray * labels = matGetVariable(labelmat,"KSC_gt");

	trainset = (double*)mxGetData(train);
	trainlabels = (double*)mxGetData(labels);

	const mwSize * dim;
	dim = mxGetDimensions(train);
	matClose(datamat);

	double correct = training(trainset, trainlabels, dim[0], dim[1], dim[2], device_choosed_num, master_choosed);
	fprintf(stdout,"Accuracy: %.3f%% \n", correct * 100);
    
    cudaDeviceReset();
	return 0;
}
