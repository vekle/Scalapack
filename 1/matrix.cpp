#ifndef __matrix_cpp__
#define __matrix_cpp__

#include <complex>
#include <iostream>

#include <assert.h>
#include <mpi.h>
#include <math.h>

using std::complex;
using std::max;
using std::cout;
using std::endl;

extern "C" {
    void Cblacs_pinfo(int*, int*);
    void Cblacs_get(int, int, int*);
    void Cblacs_gridinit(int*, char const*, int, int);
    void Cblacs_gridinfo(int, int*, int*, int*, int*);
    void Cblacs_barrier(int , char*);
    void Cblacs_gridexit(int);
    void Cblacs_exit(int);

    int numroc_(int const& n, int const& nb, int const& iproc, int const& isproc, int const& nprocs);
    void descinit_(int *desc, int const& m, int const& n, int const& mb, int const& nb, int const& irsrc, int const& icsrc, int const& ictxt, int const& lld, int *info);
    void pzgeadd_(char const *TRANS, int const& M, int const& N, 
            complex<double> const& ALPHA, complex<double> *A, int const& IA, int const& JA, int const *DESCA, 
            complex<double> const& BETA, complex<double> *C, int const& IC, int const& JC, int const *DESCC);
    void pzheevd_(char const *jobz, char const *uplo, int const& n, 
            complex<double> *a, int const& ia, int const& ja, int const *desca, 
            double *w, 
            complex<double> *z, int const& iz, int const& jz, int const *descz, 
            complex<double> *work, int const *lwork, double *rwork, int const *lrwork, int *iwork, int const *liwork, int *info);
    void pzgemm_(char const *transa, char const *transb, int const& M, int const& N, int const& K, 
            complex<double> const& ALPHA,  complex<double> * A, int const& IA, int const& JA, int * DESCA, 
            complex<double> * B, int const& IB, int const& JB, int * DESCB, 
            complex<double> const& BETA, complex<double> * C, int const& IC, int const& JC, int * DESCC);
}

extern int proc_rank, proc_num;
extern int col_rank, row_rank, col_num, row_num;
extern int ictxt;

static complex<double> zero(0.);
static complex<double> one(1.);
static complex<double> n_one(-1.);
static complex<double> im_one(0, 1);

static int izero = 0;
static int ione = 1;
static int in_one = -1;


class Matrix
{
public:

    int n = 0;
    int mp = 0, nq = 0;

    complex<double> *data = nullptr;
    complex<double> *data_proc = nullptr;

    int desc[9] = {};
    int desc_proc[9] = {};

    int lld = 0;
    int lld_proc = 0;

    int root_rank = 0;
    bool scattered = false;


    void init()
    {
        int m_proc = ceil((double)n / row_num);
        int n_proc = ceil((double)n / col_num);

        mp = numroc_(n, m_proc, row_rank, root_rank, row_num);
        nq = numroc_(n, n_proc, col_rank, root_rank, col_num);

        lld = max(numroc_(n, n, row_rank, root_rank, row_num), 1);
        lld_proc = max(mp, 1);

        int info;
        descinit_(desc, n, n, n, n, izero, izero, ictxt, lld, &info);
        descinit_(desc_proc, n, n, n_proc, m_proc, izero, izero, ictxt, lld_proc, &info);
    };

    Matrix(int _n, bool _scattered)
    {
        n = _n;
        scattered = _scattered;

        init();

        if (scattered) {
            data_proc = (complex<double> *)calloc(nq * mp, sizeof(complex<double>));
        } else {
            if (proc_rank == root_rank)
                data = (complex<double> *)calloc(n * n, sizeof(complex<double>));
        }
    };


    // complex<double> *operator[](int row)
    // {
    //     if (scattered) {
    //         return data_proc + row * mp;
    //     } else {
    //         return data + row * n;
    //     }
    // };

    // complex<double> *get_data()
    // {
    //     if (scattered) {
    //         return data_proc;
    //     } else {
    //         return data;
    //     }
    // };

    // int *get_desc()
    // {
    //     if (scattered) {
    //         return desc_proc;
    //     } else {
    //         return desc;
    //     }
    // };


    bool scatter()
    {
        if (scattered)
            return false;

        data_proc = (complex<double> *)calloc(nq * mp, sizeof(complex<double>));

        pzgeadd_("N", n, n, one, data, ione, ione, desc, zero, data_proc, ione, ione, desc_proc);

        if (proc_rank == root_rank) {
            free(data);
            data = nullptr;
        }

        scattered = true;

        Cblacs_barrier(ictxt, "All");

        return scattered;
    };

    // bool gather()
    // {
    //     return scattered;
    // };


    ~Matrix()
    {
        if (data)
            free(data);
        if (data_proc)
            free(data_proc);
    };
};


double *decompose_matrix(Matrix &E, Matrix &V)
{
    double *rwork;
    complex<double> *work;
    int *iwork;
    int lwork, lrwork, liwork;

    rwork = (double *)malloc(2 * sizeof(double));
    work = (complex<double> *)malloc(2 * sizeof(complex<double>));
    iwork = (int *)malloc(2 * sizeof(int));

    int info;
    double *w = (double*)calloc(E.n, sizeof(double));

    lwork = -1; lrwork = -1; liwork = -1;
    pzheevd_("V", "U", E.n, E.data_proc, 1, 1, E.desc_proc, w, V.data_proc, 1, 1, V.desc_proc, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    lwork = work[0].real(); lrwork = rwork[0]; liwork = iwork[0];
    free(work); free(rwork); free(iwork);

    rwork = (double *)malloc(lrwork * sizeof(double));
    work = (complex<double> *)malloc(lwork * sizeof(complex<double>));
    iwork = (int *)malloc(liwork * sizeof(int));
    pzheevd_("V", "U", E.n, E.data_proc, 1, 1, E.desc_proc, w, V.data_proc, 1, 1, V.desc_proc, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    free(work); free(rwork); free(iwork);

    return w;
};

Matrix *multiply_BAB_h(Matrix &A, Matrix &B)
{
    Matrix *tmp = new Matrix(A.n, true);
    Matrix *C = new Matrix(A.n, true);

    pzgemm_("N", "N", B.n, B.n, A.n, one, B.data_proc, 1, 1, B.desc_proc, A.data_proc, 1, 1, A.desc_proc, zero, tmp->data_proc, 1, 1, tmp->desc_proc);
    pzgemm_("N", "C", tmp->n, tmp->n, B.n, one, tmp->data_proc, 1, 1, tmp->desc_proc, B.data_proc, 1, 1, B.desc_proc, zero, C->data_proc, 1, 1, C->desc_proc);

    delete tmp;

    return C;
};


Matrix *make_diag_matrix(double *w, int n)
{
    Cblacs_barrier(ictxt, "All");

    Matrix *A = new Matrix(n, true);

    if (row_rank == col_rank) {
        for (int i = 0; i < A->nq; i++) {
            A->data_proc[i * A->nq + i] = w[A->nq * row_rank + i];
        }
    }

    Cblacs_barrier(ictxt, "All");

    return A;
};

Matrix *diag_matrix_exp(complex<double> coef, Matrix &A)
{
    Cblacs_barrier(ictxt, "All");

    Matrix *B = new Matrix(A.n, true);

    if (row_rank == col_rank) {
        for (int i = 0; i < A.nq; i++) {
            B->data_proc[i * B->nq + i] = exp(coef * A.data_proc[i * A.nq + i]);
        }
    }

    Cblacs_barrier(ictxt, "All");

    return B;
};

// Очень редко происходит гонка процессов
void print_diag(Matrix &A)
{
    Cblacs_barrier(ictxt, "All");

    for (int r = 0; r < proc_num; r++)
    {
        if (r == proc_rank && row_rank == col_rank) {
            for (int i = 0; i < A.nq; i++) {
                cout << A.data_proc[i * A.nq + i] << " ";
            }
            fflush(stdout);
        }
        Cblacs_barrier(ictxt, "All");
    }

    if (proc_rank == 0) {
        cout << endl;
        fflush(stdout);
    }

    Cblacs_barrier(ictxt, "All");
};

#endif