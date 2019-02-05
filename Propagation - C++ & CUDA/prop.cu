// Kappa - this is project file for NTO project - Light Propagation with GPU

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

using namespace std;
// --- Main Part ---

/*
 * complie: nvcc -o prop.x prop.cu -O3 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_60,code=sm_60 -I/usr/local/cuda/inc -L/usr/local/cuda/lib -lcufft
 */
int main(int *argc, char *argv[])
{
	ifstream inputFile;
	int COL = atoi(argv[2]);
	int ROW = atoi(argv[3]);
	float u_in[ROW][COL];
	cout << "DUPA WELCOME" << " | " << argv[0] << " | " << argv[1] << " | " << argv[2] << " | " << argv[3] << endl;
	cout << "ROW: " << ROW << " | " << "COL: " << COL <<endl;
	inputFile.open(argv[1]);
	if (inputFile)
	{
		int i,j = 0;
		for (i = 0; i < ROW; i++)
		{
			for (j = 0; j < COL; j++)
			{	
				inputFile >> u_in[i][j];
				//cout << u_in[i][j];
			}
		//cout << endl;
		}
		cout << endl;
	} else {
		cout << "Error opening the file.\n";
	}
	inputFile.close();
				// --- FFT tablicy wejsciowej --- //
	int NX = 2*COL;
	int NY = 2*ROW;
	cufftHandle plan;
	
	cufftComplex* data;
	data = (cufftComplex *) malloc ( sizeof(cufftComplex)* NX * NY);

	cufftComplex* dData;
	cudaMalloc((void **) &dData, sizeof(cufftComplex)* NX * NY);

	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return -1;
	}

	for(int ii=0; ii < NY ; ii++)
	{
		for(int jj=0; jj < NX ; jj++)
		{
			data[ii*NX+jj].x = 0;
			data[ii*NX+jj].y = 0;
		}
	}

	for(int ii=0; ii < (int)NY/2 ; ii++)
	{
		for(int jj=0; jj < (int)NX/2 ; jj++)
		{
			data[(ii*NX+jj)+(NX*NY/4+NX/4)].x = (float)u_in[ii][jj];
		}
	}
/*
	printf( "Org vals: \n");
	for(int ii=0; ii<NX*NY ; ii++)
	{
		if (ii%NX == 0){
			printf("\n");
		}
		printf ( "%.0f ", data[ii].x );	
	}
*/
	size_t pitch;
 	cudaMallocPitch(&dData, &pitch, sizeof(cufftComplex)*NX, NY);
	cudaMemcpy2D(dData,pitch,data,sizeof(cufftComplex)*NX,sizeof(cufftComplex)*NX,NX,cudaMemcpyHostToDevice);
 	
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return -1;	
	}

// Create a 2D FFT plan. 
	if (cufftPlan2d(&plan, NX, NY, CUFFT_C2C) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		return -1;	
	}

	if (cufftExecC2C(plan, dData, dData, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		return -1;		
	}

	if (cudaDeviceSynchronize() != cudaSuccess){
  		fprintf(stderr, "Cuda error: Failed to synchronize\n");
   		return -1;
	}	

	cudaMemcpy(data, dData, sizeof(cufftComplex)*NX*NY, cudaMemcpyDeviceToHost);
	
	printf( "\nCUFFT vals: \n");
	
// TEST - wypisania
	/*
	for(int ii=0; ii<NX*NY; ii++)
	{	
		data[ii].x = ii;
	}
	*/
// KONIEC TESTU
/*	int kappa = 0;
	for(int ii=NX*NY/4+NX/4; ii<(NX*NY)-(NX*NY/4+NX/4); ii++)
	{	
		data[ii].x = sqrt(pow(data[ii].x, 2) + pow(data[ii].y, 2));
		kappa++;
		if (kappa == NX/2){ ii += NX/2; kappa = 0;}
	}
	
	float mini_data = data[NX*NY/4+NX/4].x;
	

	kappa = 0;
	for(int ii=NX*NY/4+NX/4; ii<(NX*NY)-(NX*NY/4+NX/4); ii++)
	{
		if (data[ii].x < mini_data){ mini_data = data[ii].x; }
		kappa++;
		if (kappa == NX/2){ ii += NX/2; kappa = 0;}
	}
	
	float max_data = data[NX*NY/4+NX/4].x;
	kappa = 0;
	for(int ii=NX*NY/4+NX/4; ii<(NX*NY)-(NX*NY/4+NX/4); ii++)
	{
		data[ii].x = data[ii].x + abs(mini_data);
		if (data[ii].x > max_data) { max_data = data[ii].x; }
		kappa++;
		if (kappa == NX/2){ ii += NX/2; kappa = 0;}
	}

	kappa = 0;
	for(int ii=NX*NY/4+NX/4; ii<(NX*NY)-(NX*NY/4+NX/4); ii++)
	{	
		data[ii].x = data[ii].x / max_data * 255;
		//if (ii%NX/2 == 0){printf("\n");}
		printf ( "%.0f ", data[ii].x);
		
		kappa++;
		if (kappa == NX/2){ ii += NX/2; kappa = 0; printf("\n");}
	}
*/

// Czytanie calosci

	for(int ii=0; ii<(NX*NY); ii++)
	{	
		data[ii].x = sqrt(pow(data[ii].x, 2) + pow(data[ii].y, 2));
	}
	
	float mini_data = data[0].x;
	
	for(int ii=0; ii<(NX*NY); ii++)
	{		
		if (data[ii].x < mini_data){ mini_data = data[ii].x; }
	}
	
	float max_data = data[0].x;

	for(int ii=0; ii<(NX*NY); ii++)
	{		
		data[ii].x = data[ii].x + abs(mini_data);
		if (data[ii].x > max_data) { max_data = data[ii].x; }
	}

	for(int ii=0; ii<(NX*NY); ii++)
	{	
		if (ii%NX == 0){printf("\n");}
		data[ii].x = data[ii].x / max_data * 255;
		printf ( "%.0f ", data[ii].x);
	}


	cufftDestroy(plan);
	cudaFree(data);
	cudaFree(dData);
	return 0;
} 
