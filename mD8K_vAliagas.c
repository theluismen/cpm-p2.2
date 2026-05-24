#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#define N 8000L
#define ND (N*N/100)

typedef struct {
    int i, j, v;
} tmd;

int cmp_fil ( const void *pa, const void *pb )
{
    tmd *a = (tmd *) pa;
    tmd *b = (tmd *) pb;
    if ( a->i > b->i ) return 1;
    else if ( a->i < b->i ) return -1;
    else return a->j - b->j;
}

int cmp_col ( const void *pa, const void *pb )
{
    tmd *a = (tmd *)pa;
    tmd *b = (tmd *)pb;
    if ( a->j > b->j ) return 1;
    else if ( a->j < b->j ) return -1;
    else return a->i - b->i;
}

int main(int argc, char** argv)
{
    int i, j, k;
    int rank, nodos;
    int i_ini, i_fin, cols_per_node;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodos);

    int *A  = calloc(N * N, sizeof(int));
    int *B  = calloc(N * N, sizeof(int));
    int *C1 = calloc(N * N, sizeof(int));
    int *C2 = calloc(N * N, sizeof(int));
    
    int *jBD = malloc(sizeof(int) * (N + 1));
    tmd *AD  = malloc(sizeof(tmd) * ND);
    tmd *BD  = malloc(sizeof(tmd) * ND);
    tmd *CD  = malloc(sizeof(tmd) * N * N);

    for ( k = 0; k < ND; k++ )
    {
        AD[k].i = rand() % (N - 1);
        AD[k].j = rand() % (N - 1);
        AD[k].v = rand() % 100 + 1;
        while ( A[AD[k].i * N + AD[k].j] )
        {
            if ( AD[k].i < AD[k].j ) AD[k].i = (AD[k].i + 1) % N;
            else AD[k].j = (AD[k].j + 1) % N;
        }
        A[AD[k].i * N + AD[k].j] = AD[k].v;
    }
    qsort(AD, ND, sizeof(tmd), cmp_fil);

    free(A);

    for ( k = 0; k < ND; k++ )
    {
        BD[k].i = rand() % (N - 1);
        BD[k].j = rand() % (N - 1);
        BD[k].v = rand() % 100 + 1;
        while ( B[BD[k].i * N + BD[k].j] )
        {
            if ( BD[k].i < BD[k].j ) BD[k].i = (BD[k].i + 1) % N;
            else BD[k].j = (BD[k].j + 1) % N;
        }
        B[BD[k].i * N + BD[k].j] = BD[k].v;
    }
    qsort(BD, ND, sizeof(tmd), cmp_col);

    k = 0;
    for ( j = 0; j < N + 1; j++ )
    {
        while ( k < ND && j > BD[k].j ) k++;
        jBD[j] = k;
    }

    cols_per_node = N / nodos;
    i_ini = rank * cols_per_node;
    i_fin = (rank == nodos - 1) ? N : (rank + 1) * cols_per_node;

    int *VB_local = malloc(sizeof(int) * N);
    int *VC_local = malloc(sizeof(int) * N);
    int neleC_local = 0;

    for ( i = i_ini; i < i_fin; i++ )
    {
        for ( j = 0; j < N; j++ )
        {
            VB_local[j] = B[j * N + i];
            VC_local[j] = 0;
        }

        for ( k = 0; k < ND; k++ )
            VC_local[AD[k].i] += AD[k].v * VB_local[AD[k].j];

        for ( j = 0; j < N; j++ )
            if ( VC_local[j] )
                C1[j * N + i] = VC_local[j];
    }

    for ( i = i_ini; i < i_fin; i++ )
    {
        for ( j = 0; j < N; j++ )
            VB_local[j] = 0;

        for ( k = jBD[i]; k < jBD[i + 1]; k++ )
            VB_local[BD[k].i] = BD[k].v;

        for ( k = 0; k < ND; k++ )
            C2[AD[k].i * N + i] += AD[k].v * VB_local[AD[k].j];
    }

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

    for ( j = i_ini; j < i_fin; j++ )
        for ( i = 0; i < N; i++ )
            if ( C2[i * N + j] != C1[i * N + j] )
                printf("Rank %d - Diferencies C1 i C2 pos %d,%d: %d != %d\n", rank, i, j, C1[i * N + j], C2[i * N + j]);

    long long Suma_local = 0;
    for ( k = 0; k < neleC_local; k++ )
    {
        Suma_local += CD[k].v;
        if ( CD[k].v != C1[CD[k].i * N + CD[k].j] )
            printf("Rank %d - Diferencies C1 i CD a i:%d,j:%d,v%d, k:%d, vd:%d\n",
                   rank, CD[k].i, CD[k].j, C1[CD[k].i * N + CD[k].j], k, CD[k].v);
    }

    long long Suma_global = 0;
    int neleC_global = 0;

    MPI_Reduce(&Suma_local, &Suma_global, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&neleC_local, &neleC_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if ( rank == 0 ) {
        printf("\nNumero elements de la matriu dispersa C %d\n", neleC_global);
        printf("Suma dels elements de C %lld \n", Suma_global);
    }

    free(B);
    free(C1);
    free(C2);
    free(jBD);
    free(AD);
    free(BD);
    free(CD);
    free(VB_local);
    free(VC_local);

    MPI_Finalize();
    exit(0);
}