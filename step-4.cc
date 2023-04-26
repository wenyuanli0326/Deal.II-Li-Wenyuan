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
class Step4
{
public:
  Step4();
  void run();

private:
  void global_grid();

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

  const double cube_start = 0;
  const double cube_end = 1;
  int loc_refine_times = 2;
  int global_refine_times = 2;
  int total_refine_times = loc_refine_times + global_refine_times;

  unsigned int n_of_loc_basis = 5;



  Eigen::MatrixXd loc_basis;
 
};







template <int dim>
Step4<dim>::Step4()
  : fe(1)
  , dof_handler(triangulation)
{}


template <int dim>
void Step4<dim>:: global_grid()
{
  GridGenerator::hyper_cube(triangulation, cube_start, cube_end);
  triangulation.refine_global(total_refine_times);

// print out the global triangulation
  {
    std::ofstream out("grid.svg");
    GridOut       grid_out;
    grid_out.write_svg(triangulation, out);
  }


  int Nx = (int) pow(2, global_refine_times);     // number of coarse element in one row
  double coarse_side = (cube_end - cube_start) / Nx;

// get the interior coarse degrees of freedom
  std::vector<Point<dim>> coarse_centers;
  for (int i = 1; i < Nx; i++) {
    for (int j = 1; j < Nx; j++) {
      Point<dim> coarse_center(i * coarse_side, j * coarse_side);
      coarse_centers.push_back(coarse_center);
      std::cout << "coarse center: " << coarse_center << std::endl;
    }
  }


  using iterator_type = Triangulation<2>::cell_iterator;
  using active_type   = Triangulation<2>::active_cell_iterator;

  std::vector<std::vector<active_type>> coarse_patches;
  coarse_patches.resize(coarse_centers.size());


  for (active_type cell : triangulation.active_cell_iterators()) {
    std::cout << "cell center: " << cell->center() << std::endl;
    Point<dim> cell_center = cell->center();
    for (unsigned long i = 0; i < coarse_centers.size(); i++ ) {
      Point<dim> coarse_center = coarse_centers[i];
      if (abs(cell_center[0] - coarse_center[0]) < coarse_side && 
        abs(cell_center[1] - coarse_center[1]) < coarse_side ) {
          coarse_patches[i].push_back(cell);
          
        }
    }
  }

  
  for (std::vector<active_type> coarse_patch : coarse_patches) 
    {
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
      local_cell_problem.setUp(patch_triangulation, n_of_loc_basis);
      local_cell_problem.run();

      break;

    }


 


}



template <int dim>
void Step4<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  global_grid();
  // make_grid();
  // setup_system();
  // assemble_system();
  // solve();
  // output_results();
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
