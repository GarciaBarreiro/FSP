#include <stdio.h>

void printDevProp(cudaDeviceProp devProp)
{
	// TODO: Completar esta función
	printf("\n Minor Capability:		%d\n",devProp.minor);
	printf("\n Major Capability:		%d\n",devProp.major);
	printf("\n MultiProcessor Count:	%d\n",devProp.multiProcessorCount);
	printf("\n Max threads per MultiPr.:	%d\n",devProp.maxThreadsPerMultiProcessor);
	printf("\n Max Grid Size:		%d\n",*devProp.maxGridSize);
	printf("\n Max Threads per Block:	%d\n",devProp.maxThreadsPerBlock);
	printf("\n Max Size for each Dim:	%d\n",*devProp.maxThreadsDim);
	printf("\n Num 32bits reg per SM:	%d\n",devProp.regsPerMultiprocessor);
	printf("\n Num 32bits reg pero block	%d\n",devProp.regsPerBlock);
	printf("\n Shared Mem per MultiPr.:	%ld\n",devProp.sharedMemPerMultiprocessor/1024);
	printf("\n Shared Mem per Block:	%ld\n",devProp.sharedMemPerBlock/1024);
	printf("\n Global Mem:	%ld\n",devProp.totalGlobalMem/(1024*1024));
	printf("\n Peak Mem Clock Frec:	%lf\n",devProp.memoryClockRate/1000.0);
	printf("\n Mem Bus With:	%d\n",devProp.memoryBusWidth);
	printf("\n BWPeak:	%lf\n",(devProp.memoryClockRate*1000*(devProp.memoryBusWidth/8.0)*2.0)/1000000000.0);

}

int main(int argc, char *argv[]) {
  int numDevs;
  cudaDeviceProp prop;
  cudaError_t error;

  // Obtiene el número de dispositivos (tarjetas GPUs disponibles)
  error = cudaGetDeviceCount(&numDevs);
  if(error != cudaSuccess) {
    fprintf(stderr, "Error obteniendo numero de dispositivos: %s en %s linea %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  printf("Numero de dispositivos = %d\n", numDevs);

  // Recorre las tarjetas disponibles y obtiene las propiedades de las mismas en prop.
  for(int i=0; i < numDevs; i++) {
    error = cudaGetDeviceProperties(&prop, i);
    if(error != cudaSuccess) {
      fprintf(stderr, "Error obteniendo propiedades del dispositivo %d: %s en %s linea %d\n", i, cudaGetErrorString(error), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    printf("\nDispositivo #%d\n", i);
    printDevProp(prop);
    if(!strcmp(argv[1],"T4")){
        printf("\n CUDACores:	%d\n",2560);
    }
    else if(!strcmp(argv[1],"A100")){
        printf("\n CUDACores:	%d\n",6912);
    }
  }

  return(EXIT_SUCCESS);
}
