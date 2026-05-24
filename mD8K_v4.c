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
    int i_ini, i_fin, cols_per_node, local_cols;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodos);

    cols_per_node = N / nodos;
    i_ini = rank * cols_per_node;
    i_fin = (rank == nodos - 1) ? N : (rank + 1) * cols_per_node;
    local_cols = i_fin - i_ini;

    /* RESERVA DINÁMICA ÓPTIMA - Formato Column-Major */
    int *A  = calloc(N * N, sizeof(int));
    int *B  = calloc(N * N, sizeof(int));
    // Arrays locales agrupados por columna para indexar así: [offset_columna + fila]
    int *C1_local = calloc(local_cols * N, sizeof(int));
    int *C2_local = calloc(local_cols * N, sizeof(int));
    
    int *jBD = malloc(sizeof(int) * (N + 1));
    tmd *AD  = malloc(sizeof(tmd) * ND);
    tmd *BD  = malloc(sizeof(tmd) * ND);
    tmd *CD_local = malloc(sizeof(tmd) * N * local_cols); 

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

    int *VB_local = malloc(sizeof(int) * N);
    int *VC_local = malloc(sizeof(int) * N);
    int neleC_local = 0;

    // Cálculo 1
    for ( i = i_ini; i < i_fin; i++ )
    {
        int local_i = i - i_ini;
        int offset = local_i * N; // MAGIA: Calculamos el salto aquí, no en el bucle interno
        
        for ( j = 0; j < N; j++ )
        {
            VB_local[j] = B[j * N + i]; 
            VC_local[j] = 0;
        }

        for ( k = 0; k < ND; k++ )
            VC_local[AD[k].i] += AD[k].v * VB_local[AD[k].j];

        for ( j = 0; j < N; j++ )
            if ( VC_local[j] )
                C1_local[offset + j] = VC_local[j]; // Acceso súper limpio
    }

    free(B); // Liberar antes del Cálculo 2

    // Cálculo 2
    for ( i = i_ini; i < i_fin; i++ )
    {
        int local_i = i - i_ini;
        int offset = local_i * N; // MAGIA

        for ( j = 0; j < N; j++ )
            VB_local[j] = 0;

        for ( k = jBD[i]; k < jBD[i + 1]; k++ )
            VB_local[BD[k].i] = BD[k].v;

        for ( k = 0; k < ND; k++ )
            C2_local[offset + AD[k].i] += AD[k].v * VB_local[AD[k].j]; // ¡Cero multiplicaciones!
    }

    // Cálculo 3
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
                CD_local[neleC_local].i = j;
                CD_local[neleC_local].j = i; 
                CD_local[neleC_local].v = VC_local[j];
                neleC_local++;
                VC_local[j] = 0;
            }
        }
    }

    /* Comprobaciones LOCALES ajustadas al nuevo formato */
    for ( j = i_ini; j < i_fin; j++ ) {
        int local_j = j - i_ini;
        int offset = local_j * N;
        for ( i = 0; i < N; i++ ) {
            if ( C2_local[offset + i] != C1_local[offset + i] )
                printf("Rank %d - Diferencies C1 i C2 pos %d,%d: %d != %d\n", rank, i, j, C1_local[offset + i], C2_local[offset + i]);
        }
    }

    long long Suma_local = 0;
    for ( k = 0; k < neleC_local; k++ )
    {
        Suma_local += CD_local[k].v;
        int local_j = CD_local[k].j - i_ini;
        int offset = local_j * N;
        if ( CD_local[k].v != C1_local[offset + CD_local[k].i] )
            printf("Rank %d - Diferencies C1 i CD a i:%d,j:%d,v%d, k:%d, vd:%d\n",
                   rank, CD_local[k].i, CD_local[k].j, C1_local[offset + CD_local[k].i], k, CD_local[k].v);
    }

    /* Recolección en Proceso 0 */
    long long Suma_global = 0;
    int neleC_global = 0;

    MPI_Reduce(&Suma_local, &Suma_global, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&neleC_local, &neleC_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Datatype MPI_TMD;
    MPI_Type_contiguous(3, MPI_INT, &MPI_TMD);
    MPI_Type_commit(&MPI_TMD);

    int *recvcounts = NULL;
    int *displs = NULL;
    tmd *CD_global = NULL;

    if ( rank == 0 ) {
        recvcounts = malloc(nodos * sizeof(int));
        displs = malloc(nodos * sizeof(int));
    }

    MPI_Gather(&neleC_local, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if ( rank == 0 ) {
        int total_elements = 0;
        for ( i = 0; i < nodos; i++ ) {
            displs[i] = total_elements;
            total_elements += recvcounts[i];
        }
        CD_global = malloc(sizeof(tmd) * total_elements);
    }

    MPI_Gatherv(CD_local, neleC_local, MPI_TMD, CD_global, recvcounts, displs, MPI_TMD, 0, MPI_COMM_WORLD);

    if ( rank == 0 ) {
        printf("\nNumero elements de la matriu dispersa C %d\n", neleC_global);
        printf("Suma dels elements de C %lld \n", Suma_global);
    }

    free(C1_local);
    free(C2_local);
    free(jBD);
    free(AD);
    free(BD);
    free(CD_local);
    free(VB_local);
    free(VC_local);
    if ( rank == 0 ) {
        free(recvcounts);
        free(displs);
        free(CD_global);
    }

    MPI_Type_free(&MPI_TMD);
    MPI_Finalize();
    exit(0);
}