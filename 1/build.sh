mpic++ -o main -O2 -w main.cpp -lscalapack -llapack -lblas -lgfortran -lm
echo Usage: mpirun -np \<proc_num\> ./main
echo Supports only quadratic grids
