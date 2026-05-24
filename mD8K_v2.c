#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <strings.h>
#include <assert.h>
#include <mpi.h>

#define N 8000L
#define ND N*N/100

typedef struct {
    int i, j, v;
} tmd;

// Uso de variables globales como en la original (requerirá un stack/heap grande o depender de memoria virtual)
int A[N][N], B[N][N], C1[N][N], C2[N][N];
int jBD[N + 1];
tmd AD[ND], BD[ND], CD[N * N];

int cmp_fil ( const void *pa, const void *pb )
{
    tmd *a = (tmd *) pa;
    tmd *b = (tmd *) pb;

    if ( a->i > b->i )
        return 1;
    else if ( a->i < b->i )
        return -1;
    else
        return a->j - b->j;
}

int cmp_col ( const void *pa, const void *pb )
{
    tmd *a = (tmd *)pa;
    tmd *b = (tmd *)pb;

    if ( a->j > b->j )
        return 1;
    else if ( a->j < b->j )
        return -1;
    else
        return a->i - b->i;
}

int main(int argc, char** argv)
{
    int i, j, k;
    int rank, nodos;
    int i_ini, i_fin, cols_per_node;

    /* Inicializar entorno MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodos);

    bzero(C1, sizeof(int) * (N * N));
    bzero(C2, sizeof(int) * (N * N));

    /* IMPORTANTE: Todos los procesos generan los mismos datos iniciales. 
       Al no usar srand(), rand() generará la misma secuencia exacta en cada nodo.
       Esto actúa como un "Scatter" implícito de las matrices iniciales sin usar la red. */
    for ( k = 0; k < ND; k++ )
    {
        AD[k].i = rand() % (N - 1);
        AD[k].j = rand() % (N - 1);
        AD[k].v = rand() % 100 + 1;
        while ( A[AD[k].i][AD[k].j] )
        {
            if ( AD[k].i < AD[k].j ) AD[k].i = (AD[k].i + 1) % N;
            else AD[k].j = (AD[k].j + 1) % N;
        }
        A[AD[k].i][AD[k].j] = AD[k].v;
    }
    qsort(AD, ND, sizeof(tmd), cmp_fil);

    for ( k = 0; k < ND; k++ )
    {
        BD[k].i = rand() % (N - 1);
        BD[k].j = rand() % (N - 1);
        BD[k].v = rand() % 100 + 1;
        while ( B[BD[k].i][BD[k].j] )
        {
            if ( BD[k].i < BD[k].j ) BD[k].i = (BD[k].i + 1) % N;
            else BD[k].j = (BD[k].j + 1) % N;
        }
        B[BD[k].i][BD[k].j] = BD[k].v;
    }
    qsort(BD, ND, sizeof(tmd), cmp_col);

    // calcul dels index de les columnes
    k = 0;
    for ( j = 0; j < N + 1; j++ )
    {
        while ( k < ND && j > BD[k].j ) k++;
        jBD[j] = k;
    }

    /* Calcular Porciones de Datos para cada nodo */
    cols_per_node = N / nodos;
    i_ini = rank * cols_per_node;
    i_fin = (rank == nodos - 1) ? N : (rank + 1) * cols_per_node; // Asegurar el final exacto

    int VB_local[N];
    int VC_local[N];
    int neleC_local = 0;

    // Cálculo 1 -> Solo sobre el bloque i_ini a i_fin
    for ( i = i_ini; i < i_fin; i++ )
    {
        for ( j = 0; j < N; j++ )
        {
            VB_local[j] = B[j][i];
            VC_local[j] = 0;
        }

        for ( k = 0; k < ND; k++ )
            VC_local[AD[k].i] += AD[k].v * VB_local[AD[k].j];

        for ( j = 0; j < N; j++ )
            if ( VC_local[j] )
                C1[j][i] = VC_local[j];
    }

    // Cálculo 2 -> Solo sobre el bloque i_ini a i_fin
    for ( i = i_ini; i < i_fin; i++ )
    {
        for ( j = 0; j < N; j++ )
            VB_local[j] = 0;

        for ( k = jBD[i]; k < jBD[i + 1]; k++ )
            VB_local[BD[k].i] = BD[k].v;

        for ( k = 0; k < ND; k++ )
            C2[AD[k].i][i] += AD[k].v * VB_local[AD[k].j];
    }

    // Cálculo 3 -> Solo sobre el bloque i_ini a i_fin
    for ( i = i_ini; i < i_fin; i++ )
    {
        for ( j = 0; j < N; j++ )
            VB_local[j] = VC_local[j] = 0;

        for ( k = jBD[i]; k < jBD[i + 1]; k++ )
            VB_local[BD[k].i] = BD[k].v;

        for ( k = 0; k < ND; k++ )
            VC_local[AD[k].i] += AD[k].v * VB_local[AD[k].j];

        for ( j = 0; j < N; j++ )
        {
            VB_local[j] = 0;
            if ( VC_local[j] )
            {
                CD[neleC_local].i = j;
                CD[neleC_local].j = i;
                CD[neleC_local].v = VC_local[j];
                neleC_local++;
                VC_local[j] = 0;
            }
        }
    }

    /* Comprobaciones LOCAles */
    // Comprobamos solo las columnas procesadas por este nodo (desde i_ini hasta i_fin)
    for ( j = i_ini; j < i_fin; j++ )
        for ( i = 0; i < N; i++ )
            if ( C2[i][j] != C1[i][j] )
                printf("Rank %d - Diferencies C1 i C2 pos %d,%d: %d != %d\n", rank, i, j, C1[i][j], C2[i][j]);

    long long Suma_local = 0;
    for ( k = 0; k < neleC_local; k++ )
    {
        Suma_local += CD[k].v;
        if ( CD[k].v != C1[CD[k].i][CD[k].j] )
            printf("Rank %d - Diferencies C1 i CD a i:%d,j:%d,v%d, k:%d, vd:%d\n",
                   rank, CD[k].i, CD[k].j, C1[CD[k].i][CD[k].j], k, CD[k].v);
    }

    /* Reducir resultados locales a globales en el ROOT */
    long long Suma_global = 0;
    int neleC_global = 0;

    MPI_Reduce(&Suma_local, &Suma_global, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&neleC_local, &neleC_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if ( rank == 0 ) {
        printf("\nNumero elements de la matriu dispersa C %d\n", neleC_global);
        printf("Suma dels elements de C %lld \n", Suma_global);
    }

    /* Terminar entorno de MPI */
    MPI_Finalize();
    exit(0);
}