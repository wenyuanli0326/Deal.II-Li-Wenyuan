/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */



#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <deal.II/base/logstream.h>
#include <ostream>


// packages for the eigen spectral problem
#include <deal.II/base/index_set.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>




// #include <Eigen/Dense>

using namespace dealii;


template <int dim>
class Step4
{
public:
  Step4();
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> Alocal;
  SparseMatrix<double> Slocal;

  // Vector<double> solution;
  Vector<double> system_rhs;
};



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  // f = 2 \pi^2 sin(\pi x) sin(\pi y)
  double return_value = 2.0 * M_PI;
  for (unsigned int i = 0; i < dim; ++i)
    return_value *= sin(M_PI * p(i));

  return return_value;
}


template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
  return 0.0;
}


template <int dim>
double kappa(const Point<dim> &p)
{
  if (p.square() < 0.5 * 0.5)
    return 20;
  else
    return 1;
}



// when is this being called?
template <int dim>
Step4<dim>::Step4()
  : fe(1)
  , dof_handler(triangulation)
{}



template <int dim>
void Step4<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(2);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void Step4<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  Alocal.reinit(sparsity_pattern);
  Slocal.reinit(sparsity_pattern);
  

  // solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



template <int dim>
void Step4<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);

  RightHandSide<dim> right_hand_side;

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrixA(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrixS(dofs_per_cell, dofs_per_cell);

  Vector<double>     cell_rhs(dofs_per_cell);  // rhs is not needed for the cell problem

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrixA = 0;
      cell_matrixS = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices()) {

        const double current_coefficient = kappa(fe_values.quadrature_point(q_index));

        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices()) {
              cell_matrixA(i, j) +=
                (current_coefficient *              // kappa(x_q)
                 fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

              cell_matrixS(i, j) +=
                (current_coefficient *               // kappa(x_q)
                 fe_values.shape_value(i, q_index) * // phi_i(x_q)
                 fe_values.shape_value(j, q_index) * // phi_j(x_q)
                 fe_values.JxW(q_index));            // dx 

            }

            const auto &x_q = fe_values.quadrature_point(q_index);
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            right_hand_side.value(x_q) *        // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }

      }

      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices()) {
            Alocal.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrixA(i, j));
            Slocal.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrixS(i, j));

          }
            
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

// remember to modify the matrix Slocal
  

  // need to iterate every boundary node
  // loop over golobal_dof_index
  // if it is a boundary index
  // set the value to 1
  // all other boundary values remains 0 (don't need to change)

  // std::cout << "   the row size of Alocal: " << Alocal.m()
  //           << std::endl
  //           << "   the column size of Alocal " << Alocal.n()
  //           << std::endl;

  // for (auto entry : boundary_values) {
  //   std::cout << " the first entry (index) is: " << entry.first
  //             << std::endl 
  //             << " the second entry (value) is: " << entry.second
  //             << std::endl;
    
  // }


}

/*
Ainterior = A(interior_index, interior_index)
AinteriorAll = A(interior_index, all_index)
*/

template <int dim>
void Step4<dim>::solve()
{
  

  std::map<types::global_dof_index, double> boundary_values;
  
  // this is needed to know the index of the boundary nodes
  VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            BoundaryValues<dim>(),
                                            boundary_values);

  FullMatrix<double> Rsnap(Alocal.m(), boundary_values.size());
  int j = 0;
  for (auto keyValuePair = boundary_values.begin(); keyValuePair != boundary_values.end(); keyValuePair++) {
    keyValuePair->second = 1.0;
    for (auto otherPair = boundary_values.begin(); otherPair != boundary_values.end(); otherPair++) {
      if (otherPair->first == keyValuePair->first) continue;
      otherPair->second = 0.0;
    }


  for (auto entry : boundary_values) {
    std::cout << " the first entry (index) is: " << entry.first
              << std::endl 
              << " the second entry (value) is: " << entry.second
              << std::endl;
    
  }

    Vector<double> solution;
    solution.reinit(dof_handler.n_dofs());

    MatrixTools::apply_boundary_values(boundary_values,
                                     Alocal,
                                     solution,
                                     system_rhs, false);

    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);


    solver.solve(Alocal, solution, system_rhs, PreconditionIdentity());

    std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;

    for (unsigned long i = 0; i < Rsnap.m(); i++) {
      std::cout << solution[i] << " ";
    }
    std::cout << std::endl;

   
    

    for (unsigned long i = 0; i < Rsnap.m(); i++) {
      Rsnap[i][j] = solution[i];
    }
    j++;

  }

  // for (unsigned long i = 0; i < Rsnap.m(); i++) {
  //   for (unsigned long j = 0; j < Rsnap.n(); j++) {
  //     std::cout << Rsnap[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }


    // std::cout << " the first dimension of Alocal is: " << Alocal.m()
    //           << std::endl 
    //           << " the second dimension of Alocal is: " << Alocal.n()
    //           << std::endl
    //           << " the dimension of  is: " << solution.size()
    //           << std::endl
    //           << " the dimension of  is: " << system_rhs.size()
    //           << std::endl;

 
  // FullMatrix<double> Asnap;
  // FullMatrix<double> Ssnap;
  // Rsnap.Tmmult(Asnap, Alocal, false);

  // Asnap = Rsnap.transpose() * Alocal * Rsnap;




// remember to modify the matrix Slocal

  PETScWrappers::SparseMatrix             Asnap, Ssnap;

// Asnap = Rsnap.transpose() * Alocal * Rsnap;
// Ssnap = Rsnap.transpose() * Slocal * Rsnap;

  


  std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
  std::vector<double>                     eigenvalues;

  SolverControl                    solver_control(dof_handler.n_dofs(), 1e-9);
  SLEPcWrappers::SolverKrylovSchur eigensolver(solver_control);

  eigensolver.set_which_eigenpairs(EPS_SMALLEST_REAL);
 
  eigensolver.set_problem_type(EPS_GHEP);
  
  eigensolver.solve(Asnap,
                    Ssnap,
                    eigenvalues,
                    eigenfunctions,
                    eigenfunctions.size());




}



template <int dim>
void Step4<dim>::output_results() const
{
  // DataOut<dim> data_out;

  // data_out.attach_dof_handler(dof_handler);
  // // data_out.add_data_vector(solution, "solution");

  // data_out.build_patches();

  // std::ofstream output(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
  // data_out.write_vtk(output);
}




template <int dim>
void Step4<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}



int main()
{
  {
    Step4<2> laplace_problem_2d;
    laplace_problem_2d.run();
  }

  // {
  //   Step4<3> laplace_problem_3d;
  //   laplace_problem_3d.run();
  // }


 

  return 0;
}
