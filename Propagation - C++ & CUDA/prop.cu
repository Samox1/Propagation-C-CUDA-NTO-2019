// Kappa - this is project file for NTO project - Light Propagation with GPU
// Autorzy: Szymon Baczyński && Łukasz Szeląg
// Projekt na przedmiot NTO 2018/2019

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

using namespace std;


__global__ void multiplyElementwise(cufftDoubleComplex* f0, cufftDoubleComplex* f1, int size)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size)
    {
        double a, b, c, d;
        a = f0[i].x; 
        b = f0[i].y;
        c = f1[i].x; 
        d = f1[i].y;
        f0[i].x = a*c - b*d;
        f0[i].y = a*d + b*c;
    }
}


void h_z(double lam, double z, double k, double sampling, int NX, int NY, cufftDoubleComplex* h_z_cutab)
{
	std::complex<double>* h_z_tab;
	h_z_tab = (std::complex<double> *) malloc ( sizeof(std::complex<double>)* NX * NY);

	double fi = k * z;
	double teta = k / (2.0 * z);
	double lam_z = lam * z;
	double quad = 0.0;
	double teta1 = 0.0;	

	for(int iy=0; iy < NY; iy++)
	{
		//printf("\n");
		for(int ix=0; ix < NX ; ix++)
		{
			quad = pow(((double)ix-((double)NX/2.0))*sampling, 2) + pow(((double)iy-((double)NY/2.0))*sampling, 2);
			teta1 = teta * quad;
			h_z_tab[iy*NX+ix] = exp(1i*fi)*exp(1i*teta1)/(1i*lam_z);
			h_z_cutab[iy*NX+ix].x = h_z_tab[iy*NX+ix].real();
			h_z_cutab[iy*NX+ix].y = h_z_tab[iy*NX+ix].imag();
			//printf("%.2f\t", h_z_cutab[iy*NX+ix].x);
		}
	}	
	free(h_z_tab);
}


void Q_roll(cufftDoubleComplex* u_in_fft, cufftDoubleComplex* data, int NX, int NY)
{
	for(int iy=0; iy<(NY/4); iy++)	//Petla na przepisanie tablicy koncowej
	{
		for(int jx=0; jx<(NX/4); jx++)
		{
			u_in_fft[(NX/2*NY/4+NY/4)+(jx+iy*NX/2)] = data[iy*(NX)+jx];		// Q1 -> Q4
			u_in_fft[(jx+NX/4)+(iy*NX/2)] = data[(iy*(NX)+jx)+(NX*NY*3/4)];		// Q3 -> Q2
			u_in_fft[(jx)+(iy*NX/2)] = data[((iy*NX)+jx)+(NX*3/4+NX*NY*3/4)];	// Q4 -> Q1
			u_in_fft[(jx)+(iy*NX/2)+NX*NY/2/4] = data[((iy*NX)+jx)+(NX*3/4)];	// Q2 -> Q3
		}
	}
}

void amplitude_print(cufftDoubleComplex* u_in_fft, int NX, int NY)
{
	// --- Przeliczanie Amplitudy --- //

	for(int ii=0; ii<(NX*NY/4); ii++)
	{	
		u_in_fft[ii].x = sqrt(pow(u_in_fft[ii].x, 2) + pow(u_in_fft[ii].y, 2));
	}
	
	double mini_data = u_in_fft[0].x;
	
	for(int ii=0; ii<(NX*NY/4); ii++)
	{		
		if (u_in_fft[ii].x < mini_data){ mini_data = u_in_fft[ii].x; }
	}
	
	double max_data = u_in_fft[0].x;
	mini_data = -mini_data;
	
	for(int ii=0; ii<(NX*NY/4); ii++)
	{		
		u_in_fft[ii].x = u_in_fft[ii].x + mini_data;
		if (u_in_fft[ii].x > max_data) { max_data = u_in_fft[ii].x; }
	}

	for(int ii=0; ii<(NX*NY/4); ii++)
	{	
		if (ii%(NX/2) == 0){printf("\n");}
		u_in_fft[ii].x = u_in_fft[ii].x / max_data * 255.0;
		printf ("%.0f\t", u_in_fft[ii].x);
	}
}

int FFT_Z2Z(cufftDoubleComplex* dData, int NX, int NY)
{
	// Create a 2D FFT plan. 
	int err = 0;
	cufftHandle plan1;
	if (cufftPlan2d(&plan1, NX, NY, CUFFT_Z2Z) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		err = -1;	
	}

	if (cufftExecZ2Z(plan1, dData, dData, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		err = -1;		
	}

	if (cudaDeviceSynchronize() != cudaSuccess){
  		fprintf(stderr, "Cuda error: Failed to synchronize\n");
   		err = -1;
	}	
	
	cufftDestroy(plan1);
	return err;
}

int IFFT_Z2Z(cufftDoubleComplex* dData, int NX, int NY)
{
	// Create a 2D FFT plan.
	int err = 0; 
	cufftHandle plan1;
	if (cufftPlan2d(&plan1, NX, NY, CUFFT_Z2Z) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		err = -1;	
	}

	if (cufftExecZ2Z(plan1, dData, dData, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		err = -1;		
	}

	if (cudaDeviceSynchronize() != cudaSuccess){
  		fprintf(stderr, "Cuda error: Failed to synchronize\n");
   		err = -1;
	}

	cufftDestroy(plan1);	
	return err;
}

/*
 * complie: nvcc -o prop.x prop.cu -O3 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_60,code=sm_60 -I/usr/local/cuda/inc -L/usr/local/cuda/lib -lcufft
 * start program: ./prop.x Tablica-1024x1024.txt 1024 1024 > 1024x1024.txt
 */

// --- Main Part --- //

int main(int *argc, char *argv[])
{
	ifstream inputFile;
	int COL = atoi(argv[2]);
	int ROW = atoi(argv[3]);
	double u_in[ROW][COL];
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
			}
		}
		cout << endl;
	} else {
		cout << "Error opening the file.\n";
	}
	inputFile.close();


// --- Liczenie propagacji i FFT --- //

	int NX = 2*COL;
	int NY = 2*ROW;

// --- Przeliczenie h_z --- //

	double sampling = 10.0 * pow(10.0, (-6)); 	// Sampling = 10 micro
	double lam = 633.0 * (pow(10.0,(-9))); 		// Lambda = 633 nm
	double k = 2.0 * M_PI / lam;			// Wektor falowy k
	double z = 1000.0*(pow(10.0,(-3)));		// Odleglosc propagacji = 1 metr

	printf("k = %.1f | lam = %.1f | z = %.1f", k, lam, z);

	cufftDoubleComplex* h_z_tab;
	h_z_tab = (cufftDoubleComplex *) malloc ( sizeof(cufftDoubleComplex)* NX * NY);
	h_z(lam, z, k, sampling, NX, NY, h_z_tab);	// Liczenie h_z

	printf("\n");


// --- FFT tablicy wejsciowej --- //
	
	cufftDoubleComplex* data;
	data = (cufftDoubleComplex *) malloc ( sizeof(cufftDoubleComplex)* NX * NY);

	cufftDoubleComplex* dData;
	cudaMalloc((void **) &dData, sizeof(cufftDoubleComplex)* NX * NY);

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
			data[(ii*NX+jj)+(NX*NY/4+NX/4)].x = (double)u_in[ii][jj];
		}
	}

// Liczenie U_in = FFT{u_in}
	
	size_t pitch1;
 	cudaMallocPitch(&dData, &pitch1, sizeof(cufftDoubleComplex)*NX, NY);
	cudaMemcpy2D(dData,pitch1,data,sizeof(cufftDoubleComplex)*NX,sizeof(cufftDoubleComplex)*NX,NX,cudaMemcpyHostToDevice);
 	
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return -1;	
	}

	if (FFT_Z2Z(dData, NX, NY) == -1) { return -1; }

// --- Liczenie H_Z = FFT{h_z_tab} --- //
	
	cufftDoubleComplex* H_Z;
	cudaMalloc((void **) &H_Z, sizeof(cufftDoubleComplex)* NX * NY);

	size_t pitch2;
 	cudaMallocPitch(&H_Z, &pitch2, sizeof(cufftDoubleComplex)*NX, NY);
	cudaMemcpy2D(H_Z,pitch2,h_z_tab,sizeof(cufftDoubleComplex)*NX,sizeof(cufftDoubleComplex)*NX,NX,cudaMemcpyHostToDevice);
 	
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return -1;	
	}

	if (FFT_Z2Z(H_Z, NX, NY) == -1) { return -1; }

	// Do the actual multiplication

	multiplyElementwise<<<NX*NY, 1>>>(dData, H_Z, NX*NY);
	

// --- Liczenie u_out = iFFT{dData = U_OUT} --- //

	if (IFFT_Z2Z(dData, NX, NY) == -1) { return -1; }


	cudaMemcpy(data, dData, sizeof(cufftDoubleComplex)*NX*NY, cudaMemcpyDeviceToHost);

	printf( "\nCUFFT vals: \n");
	
//TEST - wypisania
//Test do kasacji	
/*	int NX = 12;		//Pomoc
	int NY = 12;		//Test na mniejszej tablicy

	cufftComplex* data;
	data = (cufftDoubleComplex *) malloc ( sizeof(cufftDoubleComplex)* NX * NY);

	for(int ii=0; ii<NX*NY; ii++)
	{	
		data[ii].x = ii;
		data[ii].y = ii;
	}
*/	
//KONIEC TESTU

// Czytanie calosci


// --- ROLL cwiartek, zeby wszystko sie zgadzalo na koniec --- //

	cufftDoubleComplex* u_out;
	u_out = (cufftDoubleComplex *) malloc (sizeof(cufftDoubleComplex)* NX/2 * NY/2);

	Q_roll(u_out, data, NX, NY);

// --- Przeliczanie Amplitudy --- //

	amplitude_print(u_out, NX, NY);
		
	cudaFree(u_out);
	cudaFree(data);
	cudaFree(dData);
	cudaFree(h_z_tab);
	cudaFree(H_Z);

	return 0;
} 
