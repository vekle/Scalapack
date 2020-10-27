#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex>
#include <iostream>

using namespace std;

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

    //void pdgeadd_(char const *TRANS, int const& M, int const& N, double const& ALPHA, double *A, int const& IA, int const& JA, int *DESCA, double const& BETA, double *C, int const& IC, int const& JC, int *DESCC);
    void pzgeadd_(char const *TRANS, int const& M, int const& N, complex<double> const& ALPHA, complex<double> *A, int const& IA, int const& JA, int *DESCA, complex<double> const& BETA, complex<double> *C, int const& IC, int const& JC, int *DESCC);
    //void pzgeadd_(char const *TRANS, int M, int N, double *ALPHA, double *A, int IA, int JA, int *DESCA, double *BETA, double *C, int IC, int JC, int *DESCC);
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

    int row, col;
    Cblacs_gridinfo(ictxt, &proc_dim,&proc_dim,&row,&col);

    complex<double> *A = (complex<double> *)calloc(proc_n * proc_n, sizeof(complex<double>));
    complex<double> *B = (complex<double> *)calloc(proc_n * proc_n, sizeof(complex<double>));
    complex<double> *C = (complex<double> *)calloc(proc_n * proc_n, sizeof(complex<double>));

    srand(time(NULL));
    for (int i = 0; i < proc_n * proc_n; i++) {
        A[i] = complex<double>(i + 1 + rank * proc_n * proc_n, 0); //rand() % 1000;
        B[i] = complex<double>(i + 1 + rank * proc_n * proc_n, 0); //rand() % 1000;
    }

    for (int r = 0; r < proc_num; r++) {
        if (rank == r) {
            for (int i = 0; i < proc_n * proc_n; i++) {
                cout << B[i];
            }
            cout << endl;
        }
    }

    int descA[9], descB[9], descC[9];
    int info;
    descinit_(descA, n, n, proc_n, proc_n, 0, 0, ictxt, proc_n, &info);
    descinit_(descB, n, n, proc_n, proc_n, 0, 0, ictxt, proc_n, &info);
    descinit_(descC, n, n, proc_n, proc_n, 0, 0, ictxt, proc_n, &info);

    ///////
    // int descB_local[9];
    // double *B_local = (double *)calloc(n * n, sizeof(double));
    // descinit_(descB_local, n, n, n, n, 0, 0, ictxt, proc_n, &info);
    // for (int i = 0; i < n * n; i++) {
    //     B[i] = i + 1 + rank * proc_n * proc_n; //rand() % 1000;
    // }

    //pdgemm_("N", "N", n, n, n, 1., A, 1, 1, descA, B, 1, 1, descB, 0., C, 1, 1, descC);

    pzgeadd_("N", n, n, complex<double>(1., 0.), A, 1, 1, descA, complex<double>(1., 0.), B, 1, 1, descB);

    cout << B[0];

    return MPI_Finalize();
}
