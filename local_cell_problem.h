

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

// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>


using namespace dealii;




template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                  const unsigned int /*component*/) const
{
  return 0.0;
}


template <int dim>
double kappa(const Point<dim> &p)
{
  double high_value = 10000.0;
  double low_value = 1.0;
  if (0.105 <= p(0) and p(0) <= 0.205 and 0.135 <= p(1) and p(1) <= 0.185) {
    return high_value;
  } else if (0.205 <= p(0) and p(0) <= 0.305 and 0.125 <= p(1) and p(1) <= 0.165) {
    return high_value;
  } else if (0.205 <= p(0) and p(0) <= 0.305 and 0.125 <= p(1) and p(1) <= 0.165) {
    return high_value;
  } else if (0.405 <= p(0) and p(0) <= 0.505 and 0.155 <= p(1) and p(1) <= 0.195) {
    return high_value;
  } else if (0.505 <= p(0) and p(0) <= 0.605 and 0.135 <= p(1) and p(1) <= 0.185) {
    return high_value;
  } else if (0.605 <= p(0) and p(0) <= 0.705 and 0.125 <= p(1) and p(1) <= 0.165) {
    return high_value;
  } else if (0.805 <= p(0) and p(0) <= 0.905 and 0.155 <= p(1) and p(1) <= 0.195) {
    return high_value;

  } else if (0.135 <= p(0) and p(0) <= 0.165 and 0.245 <= p(1) and p(1) <= 0.275) {
    return high_value;
  } else if (0.205 <= p(0) and p(0) <= 0.305 and 0.255 <= p(1) and p(1) <= 0.295) {
    return high_value;
  } else if (0.235 <= p(0) and p(0) <= 0.265 and 0.245 <= p(1) and p(1) <= 0.255) {
    return high_value;
  } else if (0.305 <= p(0) and p(0) <= 0.405 and 0.245 <= p(1) and p(1) <= 0.285) {
    return high_value;
  } else if (0.405 <= p(0) and p(0) <= 0.505 and 0.235 <= p(1) and p(1) <= 0.275) {
    return high_value;
  } else if (0.535 <= p(0) and p(0) <= 0.565 and 0.245 <= p(1) and p(1) <= 0.275) {
    return high_value;
  } else if (0.635 <= p(0) and p(0) <= 0.665 and 0.245 <= p(1) and p(1) <= 0.275) {
    return high_value;
  } else if (0.735 <= p(0) and p(0) <= 0.765 and 0.245 <= p(1) and p(1) <= 0.275) {
    return high_value;
  } else if (0.835 <= p(0) and p(0) <= 0.865 and 0.245 <= p(1) and p(1) <= 0.275) {
    return high_value;

  } else if (0.105 <= p(0) and p(0) <= 0.205 and 0.335 <= p(1) and p(1) <= 0.385) {
    return high_value;
  } else if (0.205 <= p(0) and p(0) <= 0.305 and 0.325 <= p(1) and p(1) <= 0.365) {
    return high_value;
  } else if (0.205 <= p(0) and p(0) <= 0.305 and 0.325 <= p(1) and p(1) <= 0.365) {
    return high_value;
  } else if (0.405 <= p(0) and p(0) <= 0.505 and 0.355 <= p(1) and p(1) <= 0.395) {
    return high_value;
  } else if (0.505 <= p(0) and p(0) <= 0.605 and 0.335 <= p(1) and p(1) <= 0.385) {
    return high_value;
  } else if (0.605 <= p(0) and p(0) <= 0.705 and 0.325 <= p(1) and p(1) <= 0.365) {
    return high_value;
  } else if (0.805 <= p(0) and p(0) <= 0.905 and 0.355 <= p(1) and p(1) <= 0.395) {
    return high_value;

  } else if (0.105 <= p(0) and p(0) <= 0.135 and 0.575 <= p(1) and p(1) <= 0.595) {
    return high_value;
  } else if (0.135 <= p(0) and p(0) <= 0.295 and 0.565 <= p(1) and p(1) <= 0.585) {
    return high_value;
  } else if (0.295 <= p(0) and p(0) <= 0.525 and 0.555 <= p(1) and p(1) <= 0.575) {
    return high_value;
  } else if (0.525 <= p(0) and p(0) <= 0.825 and 0.545 <= p(1) and p(1) <= 0.565) {
    return high_value;
  } else if (0.825 <= p(0) and p(0) <= 0.905 and 0.535 <= p(1) and p(1) <= 0.555) {
    return high_value;

  } else if (0.135 <= p(0) and p(0) <= 0.165 and 0.645 <= p(1) and p(1) <= 0.675) {
    return high_value;
  } else if (0.205 <= p(0) and p(0) <= 0.305 and 0.655 <= p(1) and p(1) <= 0.695) {
    return high_value;
  } else if (0.235 <= p(0) and p(0) <= 0.265 and 0.645 <= p(1) and p(1) <= 0.655) {
    return high_value;
  } else if (0.305 <= p(0) and p(0) <= 0.405 and 0.645 <= p(1) and p(1) <= 0.685) {
    return high_value;
  } else if (0.405 <= p(0) and p(0) <= 0.505 and 0.635 <= p(1) and p(1) <= 0.675) {
    return high_value;
  } else if (0.535 <= p(0) and p(0) <= 0.565 and 0.645 <= p(1) and p(1) <= 0.675) {
    return high_value;
  } else if (0.635 <= p(0) and p(0) <= 0.665 and 0.645 <= p(1) and p(1) <= 0.675) {
    return high_value;
  } else if (0.735 <= p(0) and p(0) <= 0.765 and 0.645 <= p(1) and p(1) <= 0.675) {
    return high_value;
  } else if (0.835 <= p(0) and p(0) <= 0.865 and 0.645 <= p(1) and p(1) <= 0.675) {
    return high_value;

  } else if (0.105 <= p(0) and p(0) <= 0.205 and 0.735 <= p(1) and p(1) <= 0.785) {
    return high_value;
  } else if (0.205 <= p(0) and p(0) <= 0.305 and 0.725 <= p(1) and p(1) <= 0.765) {
    return high_value;
  } else if (0.205 <= p(0) and p(0) <= 0.305 and 0.725 <= p(1) and p(1) <= 0.765) {
    return high_value;
  } else if (0.405 <= p(0) and p(0) <= 0.505 and 0.755 <= p(1) and p(1) <= 0.795) {
    return high_value;
  } else if (0.505 <= p(0) and p(0) <= 0.605 and 0.735 <= p(1) and p(1) <= 0.785) {
    return high_value;
  } else if (0.605 <= p(0) and p(0) <= 0.705 and 0.725 <= p(1) and p(1) <= 0.765) {
    return high_value;
  } else if (0.805 <= p(0) and p(0) <= 0.905 and 0.755 <= p(1) and p(1) <= 0.795) {
    return high_value;

  } else if (0.135 <= p(0) and p(0) <= 0.165 and 0.845 <= p(1) and p(1) <= 0.875) {
    return high_value;
  } else if (0.205 <= p(0) and p(0) <= 0.305 and 0.855 <= p(1) and p(1) <= 0.895) {
    return high_value;
  } else if (0.235 <= p(0) and p(0) <= 0.265 and 0.845 <= p(1) and p(1) <= 0.855) {
    return high_value;
  } else if (0.305 <= p(0) and p(0) <= 0.405 and 0.845 <= p(1) and p(1) <= 0.885) {
    return high_value;
  } else if (0.405 <= p(0) and p(0) <= 0.505 and 0.835 <= p(1) and p(1) <= 0.875) {
    return high_value;
  } else if (0.535 <= p(0) and p(0) <= 0.565 and 0.845 <= p(1) and p(1) <= 0.875) {
    return high_value;
  } else if (0.635 <= p(0) and p(0) <= 0.665 and 0.845 <= p(1) and p(1) <= 0.875) {
    return high_value;
  } else if (0.735 <= p(0) and p(0) <= 0.765 and 0.845 <= p(1) and p(1) <= 0.875) {
    return high_value;
  } else if (0.835 <= p(0) and p(0) <= 0.865 and 0.845 <= p(1) and p(1) <= 0.875) {
    return high_value;


  } else {
    return low_value;
  }

}







template <int dim>
class Local
{
public:
  Local();
  Eigen::MatrixXd run();
  void setUp(Triangulation<dim> &input_triangulation, unsigned int input_n_of_loc_basis,
             Eigen::MatrixXd input_POU, Point<dim> input_coarse_center, 
             double input_fine_side, double input_coarse_side) 
    {
      triangulation.copy_triangulation(input_triangulation);
      // triangulation = &input_triangulation;
      n_of_loc_basis = input_n_of_loc_basis;
      POU = input_POU;
      coarse_center = input_coarse_center;
      fine_side = input_fine_side;
      coarse_side = input_coarse_side;
    }

  const Triangulation<dim> &get_triangulation() const { return triangulation; }
  const DoFHandler<dim> &get_dof_handler() const { return dof_handler; }

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;  // consider &
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> Alocal;
  SparseMatrix<double> Slocal;

  // Vector<double> solution;
  Vector<double> system_rhs;

  Eigen::MatrixXd POU;
  Point<dim> coarse_center;
  double fine_side;
  double coarse_side;

  unsigned int n_of_loc_basis;

  Eigen::MatrixXd loc_basis;
  
 
};



// // check this !! the &;
// template <int dim>
// void Local<dim>::setUp(Triangulation<dim> &input_triangulation, unsigned int input_n_of_loc_basis,
//                         Eigen::MatrixXd input_POU, Point<dim> input_coarse_center, 
//                         double input_fine_side, double input_coarse_side)
// {

// }


template <int dim>
Local<dim>::Local()
  : fe(1)
  , dof_handler(triangulation)
{}



template <int dim>
void Local<dim>::make_grid()
{

  // std::cout << "   Number of active cells: " << triangulation.n_active_cells()
  //           << std::endl
  //           << "   Total number of cells: " << triangulation.n_cells()
  //           << std::endl;

}


template <int dim>
void Local<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  // std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
  //           << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  Alocal.reinit(sparsity_pattern);
  Slocal.reinit(sparsity_pattern);
  

  // solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



template <int dim>
void Local<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrixA(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrixS(dofs_per_cell, dofs_per_cell);

  Vector<double>     cell_rhs(dofs_per_cell);  // rhs is 0 for the cell problem

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

            // can just use cell_rhs(i) += 0;
            cell_rhs(i) += 0;
            // const auto &x_q = fe_values.quadrature_point(q_index);
            // cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
            //                 right_hand_side.value(x_q) *        // f(x_q)
            //                 fe_values.JxW(q_index));            // dx
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
  



}



template <int dim>
void Local<dim>::solve()
{
  

  std::map<types::global_dof_index, double> boundary_values;

    // try to put this inside the for loop;
  SparseMatrix<double> Alocaltemp;
  Alocaltemp.reinit(sparsity_pattern);
  Alocaltemp.copy_from(Alocal);

    // //try to check if this code make a difference or not, should be no difference
    // //tried, made no difference
//   Vector<double> system_rhs_temp;
//   system_rhs_temp.reinit(dof_handler.n_dofs());
//   system_rhs_temp = system_rhs;
  


  
  // this is needed to know the index of the boundary nodes
  // we can also use Functions::ZeroFunction<2>(), for BoundaryValues<dim>()
  VectorTools::interpolate_boundary_values(dof_handler,
                                            0,
                                            BoundaryValues<dim>(),
                                            boundary_values);

  
  Eigen::MatrixXd Rsnap(Alocal.m(), boundary_values.size());
  int j = 0;    // column index for Rsnap
  for (auto keyValuePair = boundary_values.begin(); keyValuePair != boundary_values.end(); keyValuePair++) {
    keyValuePair->second = 1.0;
    for (auto otherPair = boundary_values.begin(); otherPair != boundary_values.end(); otherPair++) {
      if (otherPair->first == keyValuePair->first) continue;
      otherPair->second = 0.0;
    }



    Vector<double> solution;
    solution.reinit(dof_handler.n_dofs());

    MatrixTools::apply_boundary_values(boundary_values,
                                     Alocaltemp,
                                     solution,
                                     system_rhs, false);

    SolverControl            solver_control(2000, 1e-5);
    SolverCG<Vector<double>> solver(solver_control);


    solver.solve(Alocaltemp, solution, system_rhs, PreconditionIdentity());

    // std::cout << "   " << solver_control.last_step()
    //         << " CG iterations needed to obtain convergence." << std::endl;



    for (auto i = 0; i < Rsnap.rows(); i++) {
      Rsnap(i,j) = solution[i];
    }
    j++;

  }


  FullMatrix<double> AlocalDense(Alocal.m(), Alocal.n());
  AlocalDense.copy_from(Alocal);	
  FullMatrix<double> SlocalDense(Slocal.m(), Slocal.n());
  SlocalDense.copy_from(Slocal);
  Eigen::MatrixXd Alocal0(Alocal.m(), Alocal.n());
  Eigen::MatrixXd Slocal0(Alocal.m(), Alocal.n());
  for (unsigned long i = 0; i < AlocalDense.m(); i++) {
    for (unsigned long j = 0; j < AlocalDense.n(); j++) {

      Alocal0(i, j) = AlocalDense[i][j];
      Slocal0(i, j) = SlocalDense[i][j];
    }
  }

  // remember to modify Slocal !!!!
  Slocal0 = Slocal0 / coarse_side / coarse_side;

//   Eigen::MatrixXd Asnap = Rsnap.transpose() * Alocal0 * Rsnap;
//   Eigen::MatrixXd Ssnap = Rsnap.transpose() * Slocal0 * Rsnap;


//   // to ensure the matrices are symmetric
//   // they are symmetric originally, only some machine error difference
//   Asnap = (Asnap + Asnap.transpose()) / 2;
//   Ssnap = (Ssnap + Ssnap.transpose()) / 2;


//   Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;

//   ges.compute(Asnap, Ssnap);


//   // std::cout << "The (complex) numerators of the generalzied eigenvalues are: " << ges.alphas().transpose() << std::endl;
//   // std::cout << "The (real) denominatore of the generalzied eigenvalues are: " << ges.betas().transpose() << std::endl;
// //   std::cout << "The (complex) generalzied eigenvalues are (alphas./beta): " << ges.eigenvalues().transpose() << std::endl;
//   // std::cout << "The (complex) generalzied eigenvectors are: " << ges.eigenvectors().transpose() << std::endl;

//   // // remember to modify the matrix Slocal


//   Eigen::MatrixXd loc_basis0;
//   loc_basis0 = Rsnap * ges.eigenvectors().leftCols(n_of_loc_basis);




  // for testing snapshot space
  Alocal0 = (Alocal0 + Alocal0.transpose()) / 2;
  Slocal0 = (Slocal0 + Slocal0.transpose()) / 2;
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
  ges.compute(Alocal0, Slocal0);
  Eigen::MatrixXd loc_basis0;
  loc_basis0 = ges.eigenvectors().leftCols(n_of_loc_basis);

  // for testing 

  

  // for testing POU and also the mapping between local and global
  for (int i = 0; i < loc_basis0.rows(); i++) {
    for (int j = 0; j < loc_basis0.cols(); j++) {
      loc_basis0(i, j) = 1.0;
    }
  }

  // for testing 




  MappingQ<dim> mapping(1);
  std::map<types::global_dof_index, Point<dim>> support_points;
  auto fe_collection = dof_handler.get_fe_collection();

  
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);


  Eigen::VectorXd POUvector(support_points.size());
  
// does the order in the support points match the order in dof numbering?
// yes, they match. I checked the support points and the boundary_values.

  int move_position = POU.cols() / 2;


  
  for (auto support_point : support_points) {
    Point<dim> coordinates = support_point.second;
    int positionx = (int) round((coordinates(0) - coarse_center[0]) / fine_side) + move_position;
    int positiony = (int) round((coordinates(1) - coarse_center[1]) / fine_side) + move_position;
    POUvector(support_point.first) = POU(POU.rows() - 1 - positiony, positionx);

  }

  loc_basis = loc_basis0.array().colwise() * POUvector.array();
  




  // for testing 
  DataOut<dim> data_out1;

  data_out1.attach_dof_handler(dof_handler);



  Vector<double> POUsolution;
  POUsolution.reinit(dof_handler.n_dofs());
  

  for (int i = 0; i < loc_basis.rows(); i++) {
    POUsolution[i] = POUvector(i, 0);
  }
//   std::cout << solution << std::endl;

  data_out1.add_data_vector(POUsolution, "POUsolution");
  

  data_out1.build_patches();

  std::ofstream output1("solution-POU-local.vtk");

  data_out1.write_vtk(output1);

  // for testing 
}



template <int dim>
void Local<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);



  Vector<double> solution;
  solution.reinit(dof_handler.n_dofs());
  

  for (int i = 0; i < loc_basis.rows(); i++) {
    solution[i] = loc_basis(i, 0);
  }
//   std::cout << solution << std::endl;

  data_out.add_data_vector(solution, "solution");
  

  data_out.build_patches();

  std::ofstream output("solution-basis.vtk");

  data_out.write_vtk(output);

  // DataOut<dim> data_out;
  // data_out.add_data_vector(POU.reshaped(dof_handler.n_dofs(),1), "solution");
  // std::ofstream output(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
  // data_out.write_vtk(output);

}




template <int dim>
Eigen::MatrixXd Local<dim>::run()
{


  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();

  return loc_basis;
}


