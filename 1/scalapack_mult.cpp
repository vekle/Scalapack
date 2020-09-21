#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
    void Cblacs_pinfo(int*, int*);
    void Cblacs_get(int, int, int*);
    void Cblacs_gridinit(int*, char const*, int, int);
    void Cblacs_gridinfo(int, int*, int*, int*, int*);
    void Cblacs_barrier(int , char*);
    void Cblacs_gridexit(int);
    void Cblacs_exit(int);

    void descinit_( int *desc, int const& m, int const& n, int const& mb, int const& nb, int const& irsrc, int const& icsrc, int const& ictxt, int const& lld, int *info);
    void pdgemm_( char const *transa, char const *transb, int const& M, int const& N, int const& K, double const& ALPHA,  double * A, int const& IA, int const& JA, int * DESCA, double * B, int const& IB, int const& JB, int * DESCB, double const& BETA, double * C, int const& IC, int const& JC, int * DESCC );
}

int main(int argc, char **argv)
{
    int rank;
    int proc_num, proc_dim;
    int ictxt;

    MPI_Init(&argc, &argv);

    int n;
    if (argc != 2) {
        n = 64;
    } else {
        n = atoi(argv[1]);
        assert(n > 0);
    }    

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    proc_dim = sqrt(proc_num);
    int proc_n = n / proc_dim;
    assert(n % proc_dim == 0);

    Cblacs_get(0, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Grid", proc_dim, proc_dim);

    //Cblacs_gridinfo(ictxt,&nprow,&npcol,&myrow,&mycol);

    double *A = (double *)calloc(proc_n * proc_n, sizeof(double));
    double *B = (double *)calloc(proc_n * proc_n, sizeof(double));
    double *C = (double *)calloc(proc_n * proc_n, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < proc_n * proc_n; i++) {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }

    int descA[9], descB[9], descC[9];
    int info;
    descinit_(descA, n, n, proc_n, proc_n, 0, 0, ictxt, proc_n, &info);
    descinit_(descB, n, n, proc_n, proc_n, 0, 0, ictxt, proc_n, &info);
    descinit_(descC, n, n, proc_n, proc_n, 0, 0, ictxt, proc_n, &info);

    pdgemm_("N", "N", n, n, n, 1., A, 1, 1, descA, B, 1, 1, descB, 0., C, 1, 1, descC);

    return MPI_Finalize();
}
