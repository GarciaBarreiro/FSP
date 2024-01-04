
/**
 * Multiplica dos matrices cuadradas: C = A * B.
 */
#include <stdio.h>
#include <time.h>

#define checkError(ans) { asserError((ans), __FILE__, __LINE__); }
inline void asserError(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define TSET(time)  clock_gettime( CLOCK_MONOTONIC, &(time) )
#define TINT(ts,te) { ( (double) 1000.*( (te).tv_sec - (ts).tv_sec ) + ( (te).tv_nsec - (ts).tv_nsec )/(double) 1.e6 ) }

// Numero maximo de threads por cada dimensión del bloque
// Consideramos threadsPerBlock.x == threadsPerBlock.y
//
#define MAX_TH_PER_BLOCK_DIM 34

// Tamanho por defecto de las matrices
#define MATDIMDEFX 1000
#define MATDIMDEFY 1000

// Numero de threads por cada dimensión bloque por defecto
#define TPBDIMDEFX 4
#define TPBDIMDEFY 4

// Tipo de datos
typedef float basetype;

void check_memoria(const unsigned int numElem_A, const unsigned int numElem_B, const unsigned int numElem_C);

/**
 * Codigo host
 */
__host__ void
h_matrizMul(const basetype *A, const basetype *B, basetype *C, 
        unsigned int A_x, unsigned int A_y, unsigned int B_x, unsigned int B_y)
{
  for (unsigned int i = 0; i < A_x; ++i)
    for (unsigned int j = 0; j < B_y; ++j) {
      basetype sum = (basetype) 0.0;
      for (unsigned int k = 0; k < A_y; ++k)
        sum += A[i*A_y + k]*B[k*B_y + j];
      C[i*B_y + j] = sum;
  }
}

/**
 * Codigo CUDA
 * Cada thread computa un elemento de C
 */
__global__ void
matrizMul(const basetype *A, const basetype *B, basetype *C,
        unsigned int A_x, unsigned int A_y, unsigned int B_x, unsigned int B_y)
{
  // TODO: Calcula el indice de la fila de C y A
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  // TODO Calcula el indice de la columna de C y B
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

  if ((i < A_x) && (j < B_y))
  {
    basetype sum = (basetype) 0.0;
    for(unsigned int k = 0; k < A_y; ++k)
    {
      sum += A[i*A_y + k]*B[k*B_y + j];
    }
    C[i*B_y + j] = sum;
  }
}

/**
 * Funcion main en el host
 * Parametros: nElementos threadsPerBlock
 */
int
main(int argc, char *argv[])
{
  basetype *h_A=NULL, *h_B=NULL, *h_C=NULL, *h_C2=NULL;
  basetype *d_A=NULL, *d_B=NULL, *d_C=NULL;
  unsigned int tpbdimx = 1, tpbdimy = 1;
  unsigned int A_x = 1, A_y = 1, B_x = 1, B_y = 1;
  unsigned int numElem_A = 1, numElem_B = 1, numElem_C = 1;
  size_t size_A = 0, size_B = 0, size_C = 0;
  // Valores para la medida de tiempos
  struct timespec tstart, tend;
  double tint;

  // Tamanho de los vectores
  A_x = (argc > 1) ? atoi(argv[1]):MATDIMDEFX;
  A_y = (argc > 2) ? atoi(argv[2]):MATDIMDEFY;
  B_x = (argc > 2) ? atoi(argv[2]):MATDIMDEFY;
  B_y = (argc > 3) ? atoi(argv[3]):MATDIMDEFX;
  // Número de elementos de las matrices
  numElem_A = A_x*A_y;
  numElem_B = B_x*B_y;
  numElem_C = A_x*B_y;
  // Tamanho de las matrices en bytes
  size_A = numElem_A * sizeof(basetype);
  size_B = numElem_B * sizeof(basetype);
  size_C = numElem_C * sizeof(basetype);

  // Numero de threads por cada dimension  del bloque
  tpbdimx = (argc > 4) ? atoi(argv[4]):TPBDIMDEFX;
  tpbdimy = (argc > 5) ? atoi(argv[5]):TPBDIMDEFY;
  // Comprueba si es superior al máximo
  tpbdimx = (tpbdimx > MAX_TH_PER_BLOCK_DIM) ? MAX_TH_PER_BLOCK_DIM:tpbdimx;
  tpbdimy = (tpbdimy > MAX_TH_PER_BLOCK_DIM) ? MAX_TH_PER_BLOCK_DIM:tpbdimy;

  check_memoria( numElem_A, numElem_B, numElem_C );

  // Caracteristicas del Grid
  // Hilos por bloque: primer parámetro dim_x, segundo dim_y
  dim3 threadsPerBlock( tpbdimx, tpbdimy, 1 );
  // TODO: Calcula el número de bloques en el Grid (bidimensional)
  dim3 blocksPerGrid( (A_x+tpbdimx) / threadsPerBlock.x, (B_y+tpbdimy) / threadsPerBlock.y, 1 );

  printf("Multiplicación de matrices de dimension (%u,%u) y (%u, %u), con (%u,%u) bloques de (%u,%u) threads\n",
    A_x, A_y, B_x, B_y, blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

  h_A = (basetype *) malloc(size_A);
  h_B = (basetype *) malloc(size_B);
  h_C = (basetype *) malloc(size_C);
  h_C2 = (basetype *) malloc(size_C);

  // Comprueba errores
  if (h_A == NULL || h_B == NULL || h_C == NULL || h_C2 == NULL)
  {
    fprintf(stderr, "Error reservando memoria en el host\n");
    exit(EXIT_FAILURE);
  }

  // Inicializa las matrices en el host
  for (unsigned int i = 0; i < numElem_A; ++i)
    h_A[i] = rand()/(basetype)RAND_MAX;
  for (unsigned int i = 0; i < numElem_B; ++i)
    h_B[i] = rand()/(basetype)RAND_MAX;

  // Inicio tiempo
  TSET(tstart);
  //clock_gettime( CLOCK_MONOTONIC, &tstart );
  // Multiplica las matrices en el host
  h_matrizMul( h_A, h_B, h_C, A_x, A_y, B_x, B_y );
  // Fin tiempo
  TSET( tend );
  tint = TINT(tstart, tend);
  printf( "HOST: Tiempo multiplicacion: %lf ms\n", tint );

  // Inicio tiempo multiplicacion GPU
  TSET( tstart );

  // Reserva memoria para las matrices en el dispositivo
  checkError( cudaMalloc((void **) &d_A, size_A) );
  checkError( cudaMalloc((void **) &d_B, size_B) );
  checkError( cudaMalloc((void **) &d_C, size_C) );

  // Copia las matrices h_A y h_B del host al dispositivo
  checkError( cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice) );
  checkError( cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice) );

  // TODO: Lanza el kernel CUDA
  matrizMul<<<blocksPerGrid, threadsPerBlock>>>( d_A, d_B, d_C, A_x, A_y, B_x, B_y );

  // Comprueba si hubo un error al el lanzamiento del kernel
  // Notar que el lanzamiento del kernel es asíncrono por lo que
  // este chequeo podría no detectar errores en la ejecución del mismo
  checkError( cudaPeekAtLastError() );
  // Sincroniza los hilos del kernel y chequea errores
  // Este chequeo detecta posibles errores en la ejecución
  // Notar que esta sincrinización puede degradar el rendimiento
  checkError( cudaDeviceSynchronize() );

  // Copia el vector resultado del dispositivo al host
  checkError( cudaMemcpy(h_C2, d_C, size_C, cudaMemcpyDeviceToHost) );

  // Fin tiempo multiplicacion GPU
  TSET( tend );
  // Calcula tiempo para la multiplicacion GPU
  tint = TINT(tstart, tend);
  printf( "DEVICE: Tiempo multiplicacion: %lf ms\n", tint );


  short error = 0;
  // Verifica que la multiplicacion es correcta
  for (unsigned int i = 0; i < numElem_C; ++i)
  {
    if (fabs(h_C2[i] - h_C[i]) > 1e-3)
    {
      printf("h_C2[%u] = %lf, h_C[%u] = %lf\n", i, h_C2[i], i, h_C[i]);
    }
  }

  if (error) {
    fprintf(stderr, "Verificacion de resultados falla\n");
    exit(EXIT_FAILURE);
  }

  printf("Multiplicacion correcta.\n");

  // Liberamos la memoria del dispositivo
  checkError( cudaFree(d_A) );
  checkError( cudaFree(d_B) );
  checkError( cudaFree(d_C) );

  // Liberamos la memoria del host
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Terminamos\n");
  return 0;
}

void
check_memoria(const unsigned int numElem_A, const unsigned int numElem_B, const unsigned int numElem_C)
{
  cudaDeviceProp prop;
  checkError( cudaGetDeviceProperties(&prop, 0) );

  size_t gmem = prop.totalGlobalMem;
  size_t bytes_arrays = (numElem_A + numElem_B + numElem_C)*sizeof(basetype);
  double gib = (double)(1073741824.0);

  printf( "GiB ocupados en la GPU: %g GiB, memoria global %g GiB\n", bytes_arrays/gib, gmem/gib );
  if( gmem >= bytes_arrays )
    printf( "GiB libres en la GPU: %g GiB\n", (gmem-bytes_arrays)/gib );
  else
    printf( "Los arrays no caben en la memoria de la GPU\n" );
}
