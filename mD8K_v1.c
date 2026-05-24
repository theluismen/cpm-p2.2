#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <strings.h>
#include <assert.h>
#include <mpi.h>

#define N  8000L
#define ND N*N/100

typedef struct {
    int i, j, v;
} tmd;

/* ---- Variables globales (todos los nodos las tienen, igual que kmean_v3) ---- */
int A[N][N], B[N][N], C1[N][N], C2[N][N];
int jBD[N + 1], VCcol[N], VBcol[N];
tmd AD[ND], BD[ND];

/* CD local de cada rank; rank 0 recoge todos con MPI_Gatherv              */
tmd CD_l[N * N];

long long Suma;

/* ----------------------------------------------------------------------------- */
/*  Comparadores para qsort                                                       */
/* ----------------------------------------------------------------------------- */
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
    tmd *a = (tmd *) pa;
    tmd *b = (tmd *) pb;

    if ( a->j > b->j )
        return 1;
    else if ( a->j < b->j )
        return -1;
    else
        return a->i - b->i;
}

/* ----------------------------------------------------------------------------- */
/*  main                                                                          */
/* ----------------------------------------------------------------------------- */
int main ( int argc, char **argv )
{
    int i, j, k;
    int rank, nodos;
    int n_cols_l;       /* Número de columnas locales (N/nodos)                   */
    int i_ini, i_fi;    /* Rango de columnas que procesa este rank                 */
    int neleC_l;        /* Elementos no nulos de CD producidos por este rank       */

    /* ---- Inicializar entorno MPI -------------------------------------------- */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    /* rank  : Identificador de nodo     */
    MPI_Comm_size(MPI_COMM_WORLD, &nodos);   /* nodos : Cantidad de nodos         */

    /* ---- Calcular porciones de columnas ----------------------------------------
       Igual que kmean_v3 calcula n_valores_l = N/nodos y apunta a su trozo,
       aquí cada rank procesa un bloque contiguo de columnas de C.                */
    n_cols_l = N / nodos;
    i_ini    = rank * n_cols_l;
    i_fi     = i_ini + n_cols_l;   /* extremo excluido                            */

    /* ---- Inicializar matrices (todos los nodos, igual que kmean_v3) ------------
       kmean_v3 comenta los Scatter/Bcast y hace que todos inicialicen valores_g.
       Aquí todos inicializan A, B, AD y BD con la misma semilla implícita de
       rand() para que sean idénticos en todos los ranks.                         */
    bzero(C1, sizeof(int) * (N * N));
    bzero(C2, sizeof(int) * (N * N));

    for ( k = 0; k < ND; k++ )
    {
        AD[k].i = rand() % (N - 1);
        AD[k].j = rand() % (N - 1);
        AD[k].v = rand() % 100 + 1;
        while ( A[AD[k].i][AD[k].j] )
        {
            if ( AD[k].i < AD[k].j )
                AD[k].i = (AD[k].i + 1) % N;
            else
                AD[k].j = (AD[k].j + 1) % N;
        }
        A[AD[k].i][AD[k].j] = AD[k].v;
    }
    qsort(AD, ND, sizeof(tmd), cmp_fil);    /* ordenat per files                  */

    for ( k = 0; k < ND; k++ )
    {
        BD[k].i = rand() % (N - 1);
        BD[k].j = rand() % (N - 1);
        BD[k].v = rand() % 100 + 1;
        while ( B[BD[k].i][BD[k].j] )
        {
            if ( BD[k].i < BD[k].j )
                BD[k].i = (BD[k].i + 1) % N;
            else
                BD[k].j = (BD[k].j + 1) % N;
        }
        B[BD[k].i][BD[k].j] = BD[k].v;
    }
    qsort(BD, ND, sizeof(tmd), cmp_col);    /* ordenat per columnes               */

    /* ---- Calcul dels index de les columnes (igual que mD8K.c) ---------------- */
    k = 0;
    for ( j = 0; j < N + 1; j++ )
    {
        while ( k < ND && j > BD[k].j )
            k++;
        jBD[j] = k;
    }

    /* ==========================================================================
       CÀLCUL 1 – Matriu dispersa per matriu densa -> C1
       Cada rank procesa sus columnas locales [i_ini, i_fi).
       Igual que kmean_v3 trabaja sobre valores_l = &valores_g[rank*n_valores_l].
       ========================================================================== */
    for ( i = i_ini; i < i_fi; i++ )
        for ( k = 0; k < ND; k++ )
            C1[AD[k].i][i] += AD[k].v * B[AD[k].j][i];

    /* ==========================================================================
       CÀLCUL 2 – Matriu dispersa per matriu dispersa -> C2
       Cada rank procesa sus columnas locales [i_ini, i_fi).
       ========================================================================== */
    for ( j = 0; j < N; j++ )
        VBcol[j] = 0;

    for ( i = i_ini; i < i_fi; i++ )
    {
        /* expandir Columna de B[*][i] */
        for ( k = jBD[i]; k < jBD[i + 1]; k++ )
            VBcol[BD[k].i] = BD[k].v;

        /* Calcul de tota una columna de C */
        for ( k = 0; k < ND; k++ )
            C2[AD[k].i][i] += AD[k].v * VBcol[AD[k].j];

        /* neteja vector de B[*][i] */
        for ( j = 0; j < N; j++ )
            VBcol[j] = 0;
    }

    /* ==========================================================================
       CÀLCUL 3 – Matriu dispersa per matriu dispersa -> CD (dispersa)
       Cada rank produce su trozo local de CD_l con neleC_l elementos.
       ========================================================================== */
    neleC_l = 0;
    for ( j = 0; j < N; j++ )
        VBcol[j] = VCcol[j] = 0;

    for ( i = i_ini; i < i_fi; i++ )
    {
        /* expandir Columna de B[*][i] */
        for ( k = jBD[i]; k < jBD[i + 1]; k++ )
            VBcol[BD[k].i] = BD[k].v;

        /* Calcul de tota una columna de C */
        for ( k = 0; k < ND; k++ )
            VCcol[AD[k].i] += AD[k].v * VBcol[AD[k].j];

        for ( j = 0; j < N; j++ )
        {
            /* neteja vector de B[*][i] */
            VBcol[j] = 0;

            /* Compressio de C */
            if ( VCcol[j] )
            {
                CD_l[neleC_l].i = j;
                CD_l[neleC_l].j = i;
                CD_l[neleC_l].v = VCcol[j];
                VCcol[j] = 0;
                neleC_l++;
            }
        }
    }

    /* ==========================================================================
       RECOLLIDA – Rank 0 recoge C1, C2 y CD de todos los ranks y verifica.
       Los tres Reduce/Gather se hacen al final juntos, sin bloqueos intermedios.
       Igual que en kmean_v3: solo rank 0 imprime los resultados finales.
       ========================================================================== */

    /* Reducir C1 y C2: cada rank solo escribió sus columnas [i_ini,i_fi),
       el resto son ceros, así que MPI_SUM reconstruye la matriz completa en rank 0. */
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : C1, C1, N * N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : C2, C2, N * N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Recoger neleC_l de cada rank en rank 0 */
    int counts_elems[nodos];
    MPI_Gather(&neleC_l, 1, MPI_INT, counts_elems, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if ( rank == 0 )
    {
        int neleC_g = 0;
        int counts[nodos];
        int displs[nodos];

        for ( i = 0; i < nodos; i++ )
        {
            displs[i]  = neleC_g * 3;      /* MPI_Gatherv trabaja en ints; tmd = 3 ints */
            neleC_g   += counts_elems[i];
            counts[i]  = counts_elems[i] * 3;
        }

        /* Buffer global para todos los CD_l */
        tmd *CD_g = malloc(neleC_g * sizeof(tmd));

        /* Recoger los fragmentos de CD de todos los ranks */
        MPI_Gatherv(CD_l, neleC_l * 3, MPI_INT,
                    CD_g, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

        /* ---- Comprovacio MD x M -> M i MD x MD -> M ---- */
        for ( i = 0; i < N; i++ )
            for ( j = 0; j < N; j++ )
                if ( C2[i][j] != C1[i][j] )
                    printf("Diferencies C1 i C2 pos %d,%d: %d != %d\n",
                           i, j, C1[i][j], C2[i][j]);

        /* ---- Comprovacio MD X MD -> M i MD x MD -> MD ---- */
        Suma = 0;
        for ( k = 0; k < neleC_g; k++ )
        {
            Suma += CD_g[k].v;
            if ( CD_g[k].v != C1[CD_g[k].i][CD_g[k].j] )
                printf("Diferencies C1 i CD a i:%d,j:%d,v%d, k:%d, vd:%d\n",
                       CD_g[k].i, CD_g[k].j, C1[CD_g[k].i][CD_g[k].j],
                       k, CD_g[k].v);
        }

        printf("\nNumero elements de la matriu dispersa C %d\n", neleC_g);
        printf("Suma dels elements de C %lld \n", Suma);

        free(CD_g);
    }
    else
    {
        /* Los ranks no-root solo envían su CD_l */
        MPI_Gatherv(CD_l, neleC_l * 3, MPI_INT,
                    NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }

    /* ---- Terminar entorno MPI (igual que kmean_v3) --------------------------- */
    MPI_Finalize();

    return 0;
}