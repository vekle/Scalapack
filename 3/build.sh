mpic++ -o main -O2 -w main.cpp -lscalapack -llapack -lblas -lgfortran -lm
echo Usage: mpirun -np \<proc_num\> ./main \<t_delta\> \<atom_num\> \<E_max\> \<E_min\> \<l_in\> \<l_out\> 
echo Supports only quadratic grids
