mpic++ -o main -O2 scalapack_mult.cpp -lscalapack -llapack -lblas -lgfortran -lm
echo Usage: mpirun -np proc_num ./main matrix_size
