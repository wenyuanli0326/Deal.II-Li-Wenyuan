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
#include <deal.II/fe/component_mask.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>


#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>


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
#include <vector>



 
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
 
#include <deal.II/base/index_set.h>
 
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
 
#include <deal.II/lac/slepc_solver.h>
 
#include <fstream>
#include <iostream>


#include <../../eigen-3147391d946bb4b6c68edd901f2add6ac1f31f8c/Eigen/Dense>
#include <../../eigen-3147391d946bb4b6c68edd901f2add6ac1f31f8c/Eigen/Eigenvalues>

#include "local_cell_problem.h"

using namespace dealii;







template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
{
  // f = 2 \pi sin(\pi x) sin(\pi y)
  double return_value = 2.0 * M_PI;
  for (unsigned int i = 0; i < dim; ++i)
    return_value *= sin(M_PI * p(i));

  return return_value;
}





template <int dim>
class Step4
{
public:
  Step4();
  void run();

private:
  void buildPOU();
  void global_grid();
  void fine_sol();


  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;


  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> Afine;
  SparseMatrix<double> Mfine;
  Vector<double> sol_fine;
  Vector<double> rhs_fine;


  const double cube_start = 0;
  const double cube_end = 1;
  int loc_refine_times = 2;
  int global_refine_times = 3;
  int total_refine_times = loc_refine_times + global_refine_times;

  unsigned int n_of_loc_basis = 1;



  Eigen::MatrixXd loc_basis;
  Eigen::MatrixXd Rms;
  
  Eigen::MatrixXd POU;
  
 
};







template <int dim>
Step4<dim>::Step4()
  : fe(1)
  , dof_handler(triangulation)
{}




// build bilinear basis functions
template <int dim>
void Step4<dim>:: buildPOU()
{
  int size1 = (int) round(pow(2, loc_refine_times + 1)) + 1;
  int size2 = (int) round(pow(2, loc_refine_times)) + 1;
  double side = 1 / pow(2, loc_refine_times - 1);

  POU.resize(size1, size1);
  Eigen::MatrixXd topleft(size2, size2);
  Eigen::MatrixXd topright(size2, size2);
  Eigen::MatrixXd botleft(size2, size2);
  Eigen::MatrixXd botright(size2, size2);
  for (int i = 0; i < size2; i++) {
    for (int j = 0; j < size2; j++) {
      double y = (size2 - 1 - i) * side;
      double x = j * side;
      topleft(i, j) = x * (1.0 - y);
      topright(i, j) = (1.0 - x) * (1.0 - y);
      botleft(i, j) = x * y;
      botright(i, j) = (1.0 - x) * y;
    }
  }

  // partion of unity

  POU.block(0, 0, size2, size2) = topleft;
  POU.block(0, size2, size2, size2 - 1) = topright.rightCols(size2 - 1);
  POU.block(size2, 0, size2 - 1, size2) = botleft.bottomRows(size2 - 1);
  POU.bottomRightCorner(size2 - 1, size2 - 1) = botright.bottomRightCorner(size2 - 1, size2 - 1);

}




template <int dim>
void Step4<dim>:: global_grid()
{


// global + loc = total
  int Nx = (int) pow(2, global_refine_times);     // number of coarse element in one row
  double coarse_side = (cube_end - cube_start) / Nx;
  int nx = (int) pow(2, loc_refine_times);        // number of fine element in a coarse element in a side
  double fine_side = coarse_side / nx; 

// get the interior coarse degrees of freedom
  std::vector<Point<dim>> coarse_centers;
  for (int i = 1; i < Nx; i++) {
    for (int j = 1; j < Nx; j++) {
      Point<dim> coarse_center(i * coarse_side, j * coarse_side);
      coarse_centers.push_back(coarse_center);
      // std::cout << "coarse center: " << coarse_center << std::endl;
    }
  }


  using iterator_type = Triangulation<2>::cell_iterator;
  using active_type   = Triangulation<2>::active_cell_iterator;

  std::vector<std::vector<active_type>> coarse_patches;
  coarse_patches.resize(coarse_centers.size());


  for (active_type cell : triangulation.active_cell_iterators()) {
    // std::cout << "cell center: " << cell->center() << std::endl;
    Point<dim> cell_center = cell->center();
    for (unsigned long i = 0; i < coarse_centers.size(); i++ ) {
      Point<dim> coarse_center = coarse_centers[i];
      if (abs(cell_center[0] - coarse_center[0]) < coarse_side && 
        abs(cell_center[1] - coarse_center[1]) < coarse_side ) {
          coarse_patches[i].push_back(cell);
          
        }
    }
  }

  Rms.resize((Nx * nx + 1) * (Nx * nx + 1), n_of_loc_basis * (Nx - 1) * (Nx - 1));

  for (unsigned long i = 0; i < coarse_centers.size(); i++)
    {

      std::vector<active_type> coarse_patch = coarse_patches[i];
      Point<dim> coarse_center = coarse_centers[i];
      // why we use iterator_type instead of active_type
      std::map<iterator_type, active_type> patch_to_global_triangulation_map;
      Triangulation<dim> patch_triangulation;

      std::map<active_type, active_type> patch_to_global_triangulation_map_temporary;

      GridTools::build_triangulation_from_patch<Triangulation<2>>(
        coarse_patch, patch_triangulation, patch_to_global_triangulation_map_temporary);

      // // do we need this?
      // for (const auto &it : patch_to_global_triangulation_map_temporary)
      // {
      //   patch_to_global_triangulation_map[it.first] = it.second;
      //   std::cout << it.first << " " << it.second << std::endl;
      // }
    
      // // for checking the patches
      // {
      //   std::ofstream out("patch" + std::to_string(i) + ".svg");
      //   GridOut       grid_out;
      //   grid_out.write_svg(patch_triangulation, out);
      // }
      // i++;


      // call the local cell problem solver with patch_triangulation
      Local<dim> local_cell_problem;
      local_cell_problem.setUp(patch_triangulation, n_of_loc_basis, POU, coarse_center, fine_side);
      Eigen::MatrixXd loc_basis = local_cell_problem.run();


      // map the loc_basis to Rms !!!
      // patch_to_global_triangulation_map_temporary.begin()->first->get_triangulation().


      

    }

}






// get fine grid matrices and solutions
template <int dim>
void Step4<dim>::fine_sol()
{

  // make fine grid
  {

    GridGenerator::hyper_cube(triangulation, cube_start, cube_end);
    triangulation.refine_global(total_refine_times);

    std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;

    // print out the total triangulation
    std::ofstream out("grid.svg");
    GridOut       grid_out;
    grid_out.write_svg(triangulation, out);
  
  }

  // set up system
  {

    dof_handler.distribute_dofs(fe);
 
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
  
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
  
    Afine.reinit(sparsity_pattern);
    Mfine.reinit(sparsity_pattern);
  
    sol_fine.reinit(dof_handler.n_dofs());
    rhs_fine.reinit(dof_handler.n_dofs());


  }


  // assemble_system()
  {
    QGauss<dim> quadrature_formula(fe.degree + 1);
  
    RightHandSide<dim> right_hand_side;
  
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  
    FullMatrix<double> cell_matrixA(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrixM(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
  
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_matrixA = 0;
        cell_matrixM = 0;
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

                cell_matrixM(i, j) +=
                  (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                   fe_values.shape_value(j, q_index) * // phi_j(x_q)
                   fe_values.JxW(q_index));           // dx
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
              Afine.add(local_dof_indices[i],
                        local_dof_indices[j],
                        cell_matrixA(i, j));
              Mfine.add(local_dof_indices[i],
                        local_dof_indices[j],
                        cell_matrixM(i, j));
              }
  
            rhs_fine(local_dof_indices[i]) += cell_rhs(i);
          }
      }
  
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            BoundaryValues<dim>(),
                                            boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                      Afine,
                                      sol_fine,
                                      rhs_fine);
  }




  // solve()
  {

    SolverControl            solver_control(5000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(Afine, sol_fine, rhs_fine, PreconditionIdentity());
  
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence." << std::endl;

  }
  



}
 






template <int dim>
void Step4<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  buildPOU();
  fine_sol();
  global_grid();

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

  // RightHandSide<2> right_hand_side;
  // Point<2> p(1,2);
  // std::cout << "value = " << right_hand_side.value(p) << std::endl;


  return 0;

}
