#include <stdio.h>

void printDevProp(cudaDeviceProp devProp)
{
    printf(" Computer Capability:       %d.%d\n", devProp.major, devProp.minor);
	printf(" MultiProcessor Count:      %d\n",devProp.multiProcessorCount);
	printf(" Max threads per MultiPr.:	%d\n",devProp.maxThreadsPerMultiProcessor);
	printf(" Max Grid Size:		        %d\n",*devProp.maxGridSize);
	printf(" Max Threads per Block:     %d\n",devProp.maxThreadsPerBlock);
	printf(" Max Size for each Dim:     %d\n",*devProp.maxThreadsDim);
	printf(" Num 32bits reg per SM:     %d\n",devProp.regsPerMultiprocessor);
	printf(" Num 32bits reg per block	%d\n",devProp.regsPerBlock);
	printf(" Shared Mem per MultiPr.:	%ld\n",devProp.sharedMemPerMultiprocessor/1024);
	printf(" Shared Mem per Block:	    %ld\n",devProp.sharedMemPerBlock/1024);
	printf(" Global Mem:	            %lf\n",devProp.totalGlobalMem/(1024*1024*1024.0));
	printf(" Peak Mem Clock Frec:	    %lf\n",devProp.memoryClockRate/1000.0);
	printf(" Mem Bus With:	            %d\n",devProp.memoryBusWidth);
	printf(" BWPeak:	                %lf\n",(devProp.memoryClockRate*1000.0*(devProp.memoryBusWidth/8.0)*2)/1e9);
    printf(" CUDA Cores:                ");
    switch (devProp.major) {
        case 7:
            printf("%d\n", 64 * devProp.multiProcessorCount);
            break;
        case 8:
            if (devProp.minor == 0) printf("%d\n", 64 * devProp.multiProcessorCount);
            else printf("%d\n", 128 * devProp.multiProcessorCount);
            break;
        default:
            printf("NOT IMPLEMENTED\n");
    }

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
  }

  return(EXIT_SUCCESS);
}
