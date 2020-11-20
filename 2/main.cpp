#include <iostream>
#include <iomanip>

#include <assert.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.cpp"

using std::cout;
using std::cin;
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


unsigned int count_1bits(unsigned int x)
{
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x0000003F;
}


int main(int argc, char **argv)
{
    cout << std::fixed << std::setprecision(3);

    MPI_Init(&argc, &argv);   

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);


    int atom_num, photon_num;
    if (argc <= 1) {
        if (proc_rank == 0)
            cout << "enter number of atoms" << endl;

        cin >> atom_num;

        if (proc_rank == 0)
            cout << "enter number of photons" << endl;

        cin >> photon_num;
    } else {
        atom_num = atoi(argv[1]);   
        photon_num = atoi(argv[2]);
    }
    assert(atom_num > 0);
    assert(photon_num > 0);


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


    double h = 1, wa = 2, wc = 4, g = 10;
    int t_max = 5, t_delta = 1;

    // Построение Гамильтониана

    int matrix_size = 0;
    if (atom_num > photon_num) {

        double combination;
        for (int i = -1; i < photon_num; i++) {

            if (i == -1) {
                combination = 1;
            } else {
                combination = combination * (atom_num - i) / (i + 1);
            }

            matrix_size += combination;
        }
    } else {
        matrix_size = (1u << atom_num);
    }

    if (proc_rank == 0)
        cout << "Размер Гамильтониана: " << matrix_size << endl;

    Matrix *H = new Matrix(matrix_size, false);

    if (proc_rank == 0) {

        int cur_basis_size = 0;
        int next_basis_size = 1;

        int *cur_basis = nullptr;
        int *next_basis = (int *)calloc(next_basis_size, sizeof(int));
        next_basis[0] = 0;

        int cur_diag_elem = 0;

        for (int i = 0; i <= photon_num && i <= atom_num; i++) {

            int cur_photon_num = photon_num - i;
            int cur_atom_num = i;

            cur_diag_elem += cur_basis_size;
            cur_basis_size = next_basis_size;
            next_basis_size = (double)next_basis_size * (atom_num - cur_atom_num) / (cur_atom_num + 1);
            if (cur_photon_num == 0 || cur_atom_num == atom_num)
                    next_basis_size = 0;

            if (cur_basis)
                free(cur_basis);
            cur_basis = next_basis;
            if (next_basis_size > 0)
                next_basis = (int *)calloc(next_basis_size, sizeof(int));

            int cur_num = 0;
            int l = 0;
            while (l < next_basis_size) {
                if (count_1bits(cur_num) == cur_atom_num + 1)
                    next_basis[l++] = cur_num;
                cur_num++;
            }

            for (int k = cur_diag_elem; k < cur_diag_elem + cur_basis_size; k++) {

                H->data[matrix_size * k + k] = h * (cur_photon_num * wc + cur_atom_num * wa);

                for (int j = 0; j < next_basis_size; j++) {

                    if ((cur_basis[k - cur_diag_elem] & next_basis[j]) == cur_basis[k - cur_diag_elem]) {
                        H->data[matrix_size * k + k + (cur_basis_size - k + cur_diag_elem + j)] = sqrt(cur_photon_num) * g;
                        H->data[matrix_size * k + k + (cur_basis_size - k + cur_diag_elem + j) * matrix_size] = sqrt(cur_photon_num) * g;
                    }
                }
            }
        }
    }

    // Вывод матрицы H

    // if (proc_rank == 0) {
    //     for (int i = 0; i < matrix_size; i++) {
    //         for (int j = 0; j < matrix_size; j++) {
    //             cout << H->data[j * matrix_size + i];
    //         }
    //         cout << endl;
    //     }
    // }

    Matrix *P = new Matrix(matrix_size, false);
    if (proc_rank == 0) { 
        P->data[0] = 1;
    }

    H->scatter();
    P->scatter();

    // Вычисление матрицы U

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

    // Унитарная динамика

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
