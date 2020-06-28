#
# MoFEM is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# MoFEM is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MoFEM. If not, see <http://www.gnu.org/licenses/>

include_directories(
  ${UM_SOURCE_DIR}/basic_finite_elements/
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(MFrontGenericInterface SHARED IMPORTED)
set_target_properties(MFrontGenericInterface PROPERTIES IMPORTED_LOCATION /home/karol/spack/opt/spack/linux-ubuntu18.04-zen/gcc-7.4.0/mgis-1.1-imbfil7ff53kka2qik7iprmun7rjfupi/lib/libMFrontGenericInterface.so)


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)
  
  add_executable(mfront_interface
    ${CMAKE_CURRENT_SOURCE_DIR}/mfront_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imp/Operators.cpp
  )
  
# link_directories(/home/karol/spack/opt/spack/linux-ubuntu18.04-zen/gcc-7.4.0/mgis-1.1-imbfil7ff53kka2qik7iprmun7rjfupi/lib)
include_directories(/home/karol/spack/opt/spack/linux-ubuntu18.04-zen/gcc-7.4.0/mgis-1.1-imbfil7ff53kka2qik7iprmun7rjfupi/include)
include_directories(/home/karol/spack/opt/spack/linux-ubuntu18.04-zen/gcc-7.4.0/mgis-1.1-imbfil7ff53kka2qik7iprmun7rjfupi/lib)
# find_library(MFrontGenericInterface NAMES MFrontGenericInterface /home/karol/spack/opt/spack/linux-ubuntu18.04-zen/gcc-7.4.0/mgis-1.1-imbfil7ff53kka2qik7iprmun7rjfupi/lib)
# find_package(MFrontGenericInterface REQUIRED)

add_dependencies(mfront_interface MFrontGenericInterface)
target_link_libraries(mfront_interface
  MFrontGenericInterface
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  mofem_cblas
  ${MoFEM_PROJECT_LIBS}
)

