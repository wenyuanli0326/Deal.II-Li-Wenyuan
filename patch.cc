#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;


void grid()
{
  Triangulation<2> triangulation;

  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(4);

  {
    std::ofstream out("grid.svg");
    GridOut       grid_out;
    grid_out.write_svg(triangulation, out);
  }

  Triangulation<2>::active_cell_iterator cell = triangulation.begin_active();
  std::advance(cell, 30);

  std::cout << cell->center() << std::endl;
  const auto patch = GridTools::get_patch_around_cell<Triangulation<2>>(cell);

  for(auto patch_cell : patch)
    std::cout << "PATCH: " << patch_cell->center() << std::endl;

  using iterator_type = Triangulation<2>::cell_iterator;
  using active_type   = Triangulation<2>::active_cell_iterator;

  std::map<iterator_type, active_type> patch_to_global_triangulation_map;
  Triangulation<2> patch_triangulation;

  {
    std::map<active_type, active_type>
      patch_to_global_triangulation_map_temporary;

    GridTools::build_triangulation_from_patch<Triangulation<2>>(
      patch, patch_triangulation, patch_to_global_triangulation_map_temporary);

    for (const auto &it : patch_to_global_triangulation_map_temporary)
      patch_to_global_triangulation_map[it.first] = it.second;

    patch_triangulation.refine_global(5);
  }


  {
    std::ofstream out("patch.svg");
    GridOut       grid_out;
    grid_out.write_svg(patch_triangulation, out);
  }


}


int main()
{
  grid();
}
