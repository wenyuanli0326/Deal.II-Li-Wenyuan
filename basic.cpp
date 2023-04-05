#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/iterative_solvers.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <iostream>

using namespace dealii;

int main()
{
  // Create a matrix
  SparseMatrix<double> A(3, 3);
  A.add(0, 0, 1.0);
  A.add(1, 1, 2.0);
  A.add(2, 2, 3.0);

  // Print the matrix to the console
  std::cout << "A = " << std::endl << A << std::endl;

  // Visualize the matrix using the Gnuplot visualization tool
  Gnuplot gnuplot;
  gnuplot.set_title("Matrix Visualization");
  gnuplot.set_xlabel("Column");
  gnuplot.set_ylabel("Row");
  gnuplot.set_zlabel("Value");
  gnuplot.plot_matrix(A);
}
