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
    cout << std::fixed << std::setprecision(4);

    MPI_Init(&argc, &argv);   

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);


    int atom_num, photon_num_max, photon_num_min;
    double l_in, l_out, t_delta;
    if (argc < 7) {
        if (proc_rank == 0)
            cout << "wrong usage" << endl;
        return MPI_Finalize();
    } else {
        t_delta = atof(argv[1]);
        atom_num = atoi(argv[2]);   
        photon_num_max = atoi(argv[3]);
        photon_num_min = atoi(argv[4]);
        l_in = atof(argv[5]);
        l_out = atof(argv[6]);
    }
    assert(atom_num > 0);
    assert(photon_num_min > 0);
    assert(photon_num_max >= photon_num_min);
    

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


    double h = 1, wa = 2000, wc = 1000, g = 10;
    double t_max = t_delta * 100;

    // Построение Гамильтониана

    int matrix_size = 0;
    for (int photon_num = photon_num_max; photon_num >= photon_num_min; photon_num--) {

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
            matrix_size += (1u << atom_num);
        }
    }

    if (proc_rank == 0)
        cout << "Размер Гамильтониана: " << matrix_size << endl;

    Matrix *H = new Matrix(matrix_size, true);

    int cur_diag_elem = 0;
    for (int photon_num = photon_num_max; photon_num >= photon_num_min; photon_num--) {

        int cur_basis_size = 0;
        int *cur_basis = nullptr;

        int next_basis_size = 1;
        int *next_basis = (int *)calloc(next_basis_size, sizeof(int));

        for (int i = 0; i <= photon_num && i <= atom_num; i++) {

            int cur_photon_num = photon_num - i;
            int cur_atom_num = i;

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

                int global_idx = matrix_size * k + k;
                int local_idx = H->get_local_idx(global_idx);

                if (local_idx != -1)
                    H->data_proc[local_idx] = h * (cur_photon_num * wc + cur_atom_num * wa);

                for (int j = 0; j < next_basis_size; j++) {

                    if ((cur_basis[k - cur_diag_elem] & next_basis[j]) == cur_basis[k - cur_diag_elem]) {

                        global_idx = matrix_size * k + k + (cur_basis_size - k + cur_diag_elem + j);
                        local_idx = H->get_local_idx(global_idx);

                        if (local_idx != -1)
                            H->data_proc[local_idx] = sqrt(cur_photon_num) * g;

                        global_idx = matrix_size * k + k + (cur_basis_size - k + cur_diag_elem + j) * matrix_size;
                        local_idx = H->get_local_idx(global_idx);

                        if (local_idx != -1)
                            H->data_proc[local_idx] = sqrt(cur_photon_num) * g;
                    }
                }
            }

            cur_diag_elem += cur_basis_size;
        }

        free(cur_basis);
    }
    
    // Вывод матрицы H

    // Cblacs_barrier(ictxt, "All");
    // for (int i = 0; i < proc_num; i++) {
    //     if (i == proc_rank) {
    //         cout << proc_rank << " proc" << endl;
    //         for (int i = 0; i < H->nq; i++) {
    //             for (int j = 0; j < H->mp; j++) {
    //                 cout << H->data_proc[i * H->mp + j] << " ";
    //             }
    //             cout << endl;
    //         }
    //         cout << endl;
    //     }
    //     Cblacs_barrier(ictxt, "All");
    // }

    // Построение операторов Линдблада

    Matrix *L_in = new Matrix(matrix_size, true);
    Matrix *L_out = new Matrix(matrix_size, true);

    int cur_matrix_size = 0;
    if (atom_num > photon_num_max) {
        double combination;
        for (int i = -1; i < photon_num_max; i++) {

            if (i == -1) {
                combination = 1;
            } else {
                combination = combination * (atom_num - i) / (i + 1);
            }

            cur_matrix_size += combination;
        }
    } else {
        cur_matrix_size += (1u << atom_num);
    }

    cur_diag_elem = 0;
    for (int photon_num = photon_num_max; photon_num > photon_num_min; photon_num--) {

        int cur_basis_size = 1;

        double coef1 = sqrt(1 + photon_num_max - photon_num);

        int next_matrix_size = 0;
        if (atom_num > photon_num - 1) {
            double combination;
            for (int i = -1; i < photon_num - 1; i++) {

                if (i == -1) {
                    combination = 1;
                } else {
                    combination = combination * (atom_num - i) / (i + 1);
                }

                next_matrix_size += combination;
            }
        } else {
            next_matrix_size += (1u << atom_num);
        }

        int first_diag_elem = cur_diag_elem;

        for (int i = 0; i <= photon_num && i <= atom_num; i++) {

            int cur_photon_num = photon_num - i;
            int cur_atom_num = i;

            double coef2 = sqrt(1 + cur_photon_num);

            for (int k = cur_diag_elem; k < cur_diag_elem + cur_basis_size; k++) {

                if (k - first_diag_elem >= next_matrix_size)
                    break;

                int global_idx = matrix_size * k + k + cur_matrix_size;
                int local_idx = L_out->get_local_idx(global_idx);

                if (local_idx != -1)
                    L_out->data_proc[local_idx] = coef1 * coef2;

                global_idx = matrix_size * k + k + cur_matrix_size * matrix_size;
                local_idx = L_in->get_local_idx(global_idx);

                if (local_idx != -1)
                    L_in->data_proc[local_idx] = coef2;  
            }

            cur_diag_elem += cur_basis_size;
            cur_basis_size = (double)cur_basis_size * (atom_num - cur_atom_num) / (cur_atom_num + 1);
        }

        cur_matrix_size = next_matrix_size;
    }

    // Матрица P

    Matrix *P = new Matrix(matrix_size, false);
    if (proc_rank == 0) { 
        P->data[matrix_size + 1] = 0.5;
        P->data[matrix_size + 2] = -0.5;
        P->data[2 * matrix_size + 1] = -0.5;
        P->data[2 * matrix_size + 2] = 0.5;
    }
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

    Matrix *U = multiply_B_A_Bh(*E_exp, *V);
    delete V;
    delete E_exp;

    // Неунитарная динамика

    for (double t = 0; t < t_max; t += t_delta) {
        Matrix *Lh_L = multiply_Ah_A(*L_in); 

        Matrix *tmp1 = multiply_A_B(*P, *Lh_L);
        Matrix *tmp2 = multiply_A_B(*Lh_L, *P);
        Matrix *tmp3 = multiply_B_A_Bh(*P, *L_in);
        delete(Lh_L);

        sum_A_B(*tmp1, *tmp2, 0.5);
        delete(tmp1);

        subtract_A_B(*tmp3, *tmp2, l_in);
        delete(tmp3);

        Lh_L = multiply_Ah_A(*L_out); 

        Matrix *tmp4 = multiply_A_B(*P, *Lh_L);
        Matrix *tmp5 = multiply_A_B(*Lh_L, *P);
        Matrix *tmp6 = multiply_B_A_Bh(*P, *L_out);
        delete(Lh_L);

        sum_A_B(*tmp4, *tmp5, 0.5);
        delete(tmp4);

        subtract_A_B(*tmp6, *tmp5, l_out);
        delete(tmp6);

        sum_A_B(*tmp2, *tmp5, complex<double>(t_delta / h));
        delete(tmp2);

        sum_A_B(*tmp5, *P, 1.0); 
        delete(tmp5);

        Matrix *P_new = multiply_Bh_A_B(*P, *U);
        delete P;
        P = P_new;

        Cblacs_barrier(ictxt, "All");
        if (proc_rank == 0) {
            cout << "time: " << t + t_delta << endl;
            fflush(stdout);
        }

        complex<double> trace = print_diag(*P, photon_num_max, photon_num_min, atom_num);

        if (proc_rank == 0 && 1.0 - trace.real() > 0.001) {
            cout << "Плохая точность вычислений (след матрицы не равен 1)" << endl;
        }
    }

    return MPI_Finalize();
}
