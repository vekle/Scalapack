#include <iostream>
#include <iomanip>

#include <assert.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.cpp"

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
}

int proc_rank, proc_num;
int col_rank, row_rank, col_num, row_num;
int ictxt;


int main(int argc, char **argv)
{
    cout << std::fixed << std::setprecision(3);

    MPI_Init(&argc, &argv);   

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    row_num = sqrt(proc_num);
    col_num = row_num;

    if (col_num * row_num != proc_num) {
        if (proc_rank == 0)
            cout << "sqrt(<proc_num>) not an integer" << endl;
        return MPI_Finalize();
    }


    Cblacs_get(0, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Grid", row_num, col_num);
    Cblacs_gridinfo(ictxt, &row_num, &col_num, &row_rank, &col_rank);


    int matrix_size = 4; 
    double h = 1, wa = 2, wc = 4, g = 10;
    int t_max = 5, t_delta = 1;

    Matrix *H = new Matrix(matrix_size, false);
    if (proc_rank == 0) {
        // H->data[0] = 2 * h * wc; H->data[1] = g; H->data[2] = g; H->data[3] = 0;
        // H->data[5] = h * (wa + wc); H->data[6] = 0; H->data[7] = g;
        // H->data[10] = h * (wa + wc); H->data[11] = g;
        // H->data[15] = 2 * h * wa;

        for (int i = 0; i < matrix_size * matrix_size; i++) {
            H->data[i] = 2;
        }
    }

    Matrix *P = new Matrix(matrix_size, false);
    if (proc_rank == 0) { 
        P->data[0] = 1;
    }

    H->scatter();
    P->scatter();

    Matrix *V = new Matrix(matrix_size, true);
    double *w = decompose_matrix(*H, *V);
    delete H;

    Matrix *E = make_diag_matrix(w, matrix_size);
    free(w);

    complex<double> coef = -im_one * complex<double>(t_delta / h);
    Matrix *E_exp = diag_matrix_exp(coef, *E);
    delete E;

    Matrix *U = multiply_BAB_h(*E_exp, *V);
    delete V;
    delete E_exp;

    for (int t = 0; t < t_max; t += t_delta) {

        Matrix *P_new = multiply_BAB_h(*P, *U);
        delete P;
        P = P_new;

        Cblacs_barrier(ictxt, "All");
        if (proc_rank == 0) {
            cout << "time: " << t + t_delta << endl;
            fflush(stdout);
        }

        print_diag(*P);
    }

    return MPI_Finalize();
}
