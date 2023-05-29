/**
 * @file MFrontOperators.cpp
 * @brief
 * @date 2023-01-25
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <MoFEM.hpp>

using namespace MoFEM;
using namespace FTensor;

// using EntData = EntitiesFieldData::EntData;
// using DomainEle = VolumeElementForcesAndSourcesCore;
// using DomainEleOp = DomainEle::UserDataOperator;

#include <BasicFiniteElements.hpp>
#include <quad.h>
#include <MGIS/Behaviour/Behaviour.hxx>
#include <MGIS/Behaviour/BehaviourData.hxx>
#include "MGIS/Behaviour/Integrate.hxx"
using namespace mgis;
using namespace mgis::behaviour;

#include <MFrontOperators.hpp>
#include <MFrontMoFEMInterface.hpp>

double mfront_dt = 1;
double mfront_dt_prop = 1;

namespace MFrontInterface {

#define VOIGT_VEC_SYMM(VEC)                                                    \
  VEC[0], inv_sqr2 *VEC[3], inv_sqr2 *VEC[4], VEC[1], inv_sqr2 *VEC[5], VEC[2]

#define VOIGT_VEC_SYMM_PLANE_STRAIN(VEC) VEC[0], inv_sqr2 *VEC[3], VEC[1]

#define VOIGT_VEC_SYMM_AXISYMMETRICAL(VEC)                                     \
  VEC[0], inv_sqr2 *VEC[3], 0, VEC[1], 0, VEC[2]

#define VOIGT_VEC_FULL(VEC)                                                    \
  VEC[0], VEC[3], VEC[5], VEC[4], VEC[1], VEC[7], VEC[6], VEC[8], VEC[2]

#define VOIGT_VEC_FULL_PLANE_STRAIN(VEC) VEC[0], VEC[3], VEC[4], VEC[1]

Index<'i', 3> i;
Index<'j', 3> j;
Index<'k', 3> k;
Index<'l', 3> l;
Index<'m', 3> m;
Index<'n', 3> n;

Index<'I', 2> I;
Index<'J', 2> J;

boost::shared_ptr<CommonData> commonDataPtr;

template <>
Tensor4Pack<3> get_tangent_tensor<Tensor4Pack<3>>(MatrixDouble &mat) {
  return getFTensor4FromMat<3, 3, 3, 3>(mat);
}

template <>
Tensor4Pack<2> get_tangent_tensor<Tensor4Pack<2>>(MatrixDouble &mat) {
  return getFTensor4FromMat<2, 2, 2, 2>(mat);
}

template <> DdgPack<3> get_tangent_tensor<DdgPack<3>>(MatrixDouble &mat) {
  return getFTensor4DdgFromMat<3, 3>(mat);
}

template <> DdgPack<2> get_tangent_tensor<DdgPack<2>>(MatrixDouble &mat) {
  return getFTensor4DdgFromMat<2, 2>(mat);
}

template <bool UPDATE, bool IS_LARGE_STRAIN, Hypothesis H>
OpStressTmp<UPDATE, IS_LARGE_STRAIN, H>::OpStressTmp(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : MFrontMoFEMInterface<Hypothesis::AXISYMMETRICAL>::DomainEleOp(
          field_name,
          MFrontMoFEMInterface<Hypothesis::AXISYMMETRICAL>::DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
template <bool UPDATE, bool IS_LARGE_STRAIN, Hypothesis H>
MoFEMErrorCode OpStressTmp<UPDATE, IS_LARGE_STRAIN, H>::doWork(int side,
                                                               EntityType type,
                                                               EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  dAta.setTag(RHS);
  dAta.behDataPtr->dt = mfront_dt;
  dAta.bView.dt = mfront_dt;

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, dAta.sizeIntVar,
                                       dAta.sizeGradVar);

  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
  MatrixDouble &mat_grad0 = *commonDataPtr->mPrevGradPtr;
  MatrixDouble &mat_stress0 = *commonDataPtr->mPrevStressPtr;

  auto t_grad = getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->mGradPtr));
  auto t_disp = getFTensor1FromMat<DIM>(*(commonDataPtr->mDispPtr));
  auto t_coords = getFTensor1CoordsAtGaussPts();

  commonDataPtr->mStressPtr->resize(DIM * DIM, nb_gauss_pts);
  commonDataPtr->mStressPtr->clear();
  auto t_stress = getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->mStressPtr));

  commonDataPtr->mFullStressPtr->resize(3 * 3, nb_gauss_pts);
  commonDataPtr->mFullStressPtr->clear();
  auto t_full_stress =
      getFTensor2FromMat<3, 3>(*(commonDataPtr->mFullStressPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    CHKERR mgis_integration<IS_LARGE_STRAIN, DIM, H>(
        gg, t_grad, t_disp, t_coords, *commonDataPtr, dAta);

    if constexpr (IS_LARGE_STRAIN) {
      // FIXME: plane strain
      if (DIM == 3) {
        Tensor2<double, 3, 3> forces(
            VOIGT_VEC_FULL(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
        t_stress(i, j) = forces(i, j);
      } else if (DIM == 2) {
        Tensor2<double, 2, 2> forces(VOIGT_VEC_FULL_PLANE_STRAIN(
            getThermodynamicForce(dAta.behDataPtr->s1, 0)));
        t_stress(I, J) = forces(I, J);
      }
    } else {
      if (DIM == 3) {
        Tensor2_symmetric<double, 3> nstress(
            VOIGT_VEC_SYMM(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
        auto forces = to_non_symm(nstress);
        t_stress(i, j) = forces(i, j);
      } else if (DIM == 2) {
        Tensor2_symmetric<double, 2> nstress(VOIGT_VEC_SYMM_PLANE_STRAIN(
            getThermodynamicForce(dAta.behDataPtr->s1, 0)));
        auto forces = to_non_symm_plane_strain(nstress);
        t_stress(I, J) = forces(I, J);

        if (H == Hypothesis::AXISYMMETRICAL) {
          Tensor2_symmetric<double, 3> fstress(VOIGT_VEC_SYMM_AXISYMMETRICAL(
              getThermodynamicForce(dAta.behDataPtr->s1, 0)));
          auto full_forces = to_non_symm(fstress);
          t_full_stress(i, j) = full_forces(i, j);
        }
      }
    }

    if constexpr (UPDATE) {
      for (int dd = 0; dd != dAta.sizeIntVar; ++dd) {
        mat_int(gg, dd) = *getInternalStateVariable(dAta.behDataPtr->s1, dd);
      }
      for (int dd = 0; dd != dAta.sizeGradVar; ++dd) {
        mat_grad0(gg, dd) = *getGradient(dAta.behDataPtr->s1, dd);
      }
      for (int dd = 0; dd != dAta.sizeStressVar; ++dd) {
        mat_stress0(gg, dd) = *getThermodynamicForce(dAta.behDataPtr->s1, dd);
      }
    }

    ++t_stress;
    ++t_full_stress;
    ++t_grad;
    ++t_disp;
    ++t_coords;
  }

  if constexpr (UPDATE) {
    CHKERR commonDataPtr->setInternalVar(fe_ent);
    // mfront_dt_prop = mfront_dt * b_view.rdt;
  }

  MoFEMFunctionReturn(0);
}

template <typename T, Hypothesis H>
OpTangent<T, H>::OpTangent(const std::string field_name,
                           boost::shared_ptr<CommonData> common_data_ptr)
    : MFrontMoFEMInterface<Hypothesis::AXISYMMETRICAL>::DomainEleOp(
          field_name,
          MFrontMoFEMInterface<Hypothesis::AXISYMMETRICAL>::DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
template <typename T, Hypothesis H>
MoFEMErrorCode OpTangent<T, H>::doWork(int side, EntityType type,
                                       EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  dAta.setTag(LHS);
  dAta.behDataPtr->dt = mfront_dt;
  dAta.bView.dt = mfront_dt;

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, dAta.sizeIntVar,
                                       dAta.sizeGradVar);

  MatrixDouble &S_E = *(commonDataPtr->materialTangentPtr);
  MatrixDouble &F_E = *(commonDataPtr->mFullTangentPtr);

  size_t tens_size = 36;
  constexpr bool IS_LARGE_STRAIN = std::is_same<T, Tensor4Pack<3>>::value ||
                                   std::is_same<T, Tensor4Pack<2>>::value;
  // constexpr int DIM = (std::is_same<T, Tensor4Pack<3>>::value ||
  //                      std::is_same<T, DdgPack<3>>::value)
  //                         ? 3
  //                         : 2;
  if constexpr (DIM == 2) {
    // plane strain
    if constexpr (IS_LARGE_STRAIN) 
      tens_size = 16;
    else
      tens_size = 9;
  } else {
    if constexpr (IS_LARGE_STRAIN) // for finite strains
      tens_size = 81;
  }
  // FIXME plain strain
  S_E.resize(tens_size, nb_gauss_pts, false);
  auto D1 = get_tangent_tensor<T>(S_E);

  size_t full_tens_size = 36;
  F_E.resize(full_tens_size, nb_gauss_pts, false);
  F_E.clear();
  //FIXME: LARGE STRAIN
  auto D2 = get_tangent_tensor<DdgPack<3>>(F_E);

  auto t_grad = getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->mGradPtr));
  auto t_disp = getFTensor1FromMat<DIM>(*(commonDataPtr->mDispPtr));
  auto t_coords = getFTensor1CoordsAtGaussPts();

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    CHKERR mgis_integration<IS_LARGE_STRAIN, DIM, H>(
        gg, t_grad, t_disp, t_coords, *commonDataPtr, dAta);

    CHKERR get_tensor4_from_voigt(&*dAta.behDataPtr->K.begin(), D1);
    CHKERR get_full_tensor4_from_voigt(&*dAta.behDataPtr->K.begin(), D2);

    ++D1;
    ++D2;
    ++t_grad;
    ++t_disp;
    ++t_coords;
  }

  MoFEMFunctionReturn(0);
}

OpAxisymmetricRhs::OpAxisymmetricRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : OpBaseImpl<PETSC,
                 MFrontMoFEMInterface<Hypothesis::AXISYMMETRICAL>::DomainEleOp>(
          field_name, field_name,
          MFrontMoFEMInterface<Hypothesis::AXISYMMETRICAL>::DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr){};

MoFEMErrorCode OpAxisymmetricRhs::iNtegrate(EntData &row_data) {
  MoFEMFunctionBegin;

  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);

  // get element volume
  const double vol = getMeasure();
  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();
  // get coordinate at integration points
  auto t_coords = getFTensor1CoordsAtGaussPts();

  auto t_full_stress =
      getFTensor2FromMat<3, 3>(*(commonDataPtr->mFullStressPtr));

  // loop over integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {

    // cout << "2, 2: " <<  t_full_stress(2, 2) << endl;

    auto t_nf = getNf<2>();

    FTensor::Tensor0<double *> t_base(&row_data.getN()(gg, 0));

    // Cylinder radius
    const double r_cylinder = t_coords(0);

    // take into account Jacobian
    const double alpha = t_w * vol * (2 * M_PI * r_cylinder);
    // loop over rows base functions
    for (int rr = 0; rr != nbRows / 2; ++rr) {

      t_nf(0) += alpha * t_full_stress(2, 2) / r_cylinder * t_base;

      ++t_base;
      ++t_nf;
    }

    ++t_full_stress;
    ++t_coords;
    ++t_w; // move to another integration weight
  }

  MoFEMFunctionReturn(0);
}

OpAxisymmetricLhs::OpAxisymmetricLhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : OpBaseImpl<PETSC,
                 MFrontMoFEMInterface<Hypothesis::AXISYMMETRICAL>::DomainEleOp>(
          field_name, field_name,
          MFrontMoFEMInterface<
              Hypothesis::AXISYMMETRICAL>::DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr){};

MoFEMErrorCode OpAxisymmetricLhs::iNtegrate(EntData &row_data,
                                            EntData &col_data) {
  MoFEMFunctionBegin;

  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);

  // get element volume
  const double vol = getMeasure();
  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor0N();
  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<2>();
  // get coordinate at integration points
  auto t_coords = getFTensor1CoordsAtGaussPts();

  Number<0> N0;
  Number<1> N1;
  Number<2> N2;

  // Elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  auto t_D = getFTensor4DdgFromMat<3, 3, 0>(*(commonDataPtr->mFullTangentPtr));

  // loop over integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {

    // cout << "0, 0, 2, 2: " << t_D(N0, N0, N2, N2) << endl;
    // cout << "1, 1, 2, 2: " << t_D(N1, N1, N2, N2) << endl;
    // cout << "2, 2, 0, 0: " << t_D(N2, N2, N0, N0) << endl;
    // cout << "2, 2, 1, 1: " << t_D(N2, N2, N1, N1) << endl;
    // cout << "2, 2, 2, 2: " << t_D(N2, N2, N2, N2) << endl;
    // cout << "2, 2, 0, 1: " << t_D(N2, N2, N0, N1) << endl;
    // cout << "0, 1, 2, 2: " << t_D(N0, N1, N2, N2) << endl;

    // Cylinder radius
    const double r_cylinder = t_coords(0);

    // take into account Jacobean
    const double alpha = t_w * vol * (2 * M_PI * r_cylinder);

    // loop over rows base functions
    int rr = 0;
    for (; rr != nbRows / 2; ++rr) {

      // get sub matrix for the row
      auto t_m = getLocMat<2>(2 * rr);

      // get derivatives of base functions for columns
      auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);
      // get base functions for columns
      auto t_col_base = col_data.getFTensor0N(gg, 0);

      // loop over columns
      for (int cc = 0; cc != nbCols / 2; cc++) {

        t_m(0, 0) += alpha * t_D(N0, N0, N2, N2) * t_col_base / r_cylinder *
                     t_row_diff_base(0);

        t_m(1, 0) += alpha * t_D(N1, N1, N2, N2) * t_col_base / r_cylinder *
                     t_row_diff_base(1);

        t_m(0, 0) += alpha * t_D(N2, N2, N0, N0) * t_col_diff_base(0) *
                     t_row_base / r_cylinder;

        t_m(0, 1) += alpha * t_D(N2, N2, N1, N1) * t_col_diff_base(1) *
                     t_row_base / r_cylinder;
                     
        t_m(0, 0) += alpha * t_D(N2, N2, N2, N2) * t_col_base / r_cylinder *
                     t_row_base / r_cylinder;


        t_m(0, 0) += alpha * t_D(N2, N2, N0, N1) * t_col_diff_base(1) *
                     t_row_base / r_cylinder;

        t_m(0, 1) += alpha * t_D(N2, N2, N0, N1) * t_col_diff_base(0) *
                     t_row_base / r_cylinder;

        t_m(0, 0) += alpha * t_D(N0, N1, N2, N2) * t_col_base / r_cylinder *
                     t_row_diff_base(1);
                     
        t_m(1, 0) += alpha * t_D(N0, N1, N2, N2) * t_col_base / r_cylinder *
                     t_row_diff_base(0);

        ++t_col_base;
        ++t_col_diff_base;
        ++t_m;
      }
      
      ++t_row_base;
      ++t_row_diff_base;
    }

    for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
        ++t_row_diff_base;
    }

    ++t_coords;
    ++t_w; // move to another integration weight
    ++t_D;
  }

  MoFEMFunctionReturn(0);
}

// OpPostProcElastic::OpPostProcElastic(
//     const std::string field_name, moab::Interface &post_proc_mesh,
//     std::vector<EntityHandle> &map_gauss_pts,
//     boost::shared_ptr<CommonData> common_data_ptr)
//     : DomainEleOp(field_name, DomainEleOp::OPROW),
//     postProcMesh(post_proc_mesh),
//       mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
//   // Operator is only executed for vertices
//   std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
// }

//! [Postprocessing]
// MoFEMErrorCode OpPostProcElastic::doWork(int side, EntityType type,
//                                          EntData &data) {
//   MoFEMFunctionBegin;
//   auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
//   auto id = commonDataPtr->blocksIDmap.at(fe_ent);
//   auto &dAta = commonDataPtr->setOfBlocksData.at(id);
//   auto &mgis_bv = *dAta.mGisBehaviour;

//   int &size_of_vars = dAta.sizeIntVar;
//   int &size_of_grad = dAta.sizeGradVar;
//   auto get_tag = [&](std::string name, size_t size) {
//     std::array<double, 9> def;
//     std::fill(def.begin(), def.end(), 0);
//     Tag th;
//     CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE,
//     th,
//                                        MB_TAG_CREAT | MB_TAG_SPARSE,
//                                        def.data());
//     return th;
//   };

//   MatrixDouble3by3 mat(3, 3);

//   auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
//     mat.clear();
//     for (size_t r = 0; r != 3; ++r)
//       for (size_t c = 0; c != 3; ++c)
//         mat(r, c) = t(r, c);
//     return mat;
//   };

//   auto set_tag = [&](auto th, auto gg, auto &mat) {
//     return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
//                                      &*mat.data().begin());
//   };

//   auto th_grad = get_tag(mgis_bv.gradients[0].name, 9);

//   size_t nb_gauss_pts1 = data.getN().size1();
//   size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
//   auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
//   Tensor2<double, 3, 3> t_stress;

//   for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

//     CHKERR set_tag(th_grad, gg, set_matrix(t_grad));

//     ++t_grad;
//   }

//   MoFEMFunctionReturn(0);
// }

// OpSaveGaussPts::OpSaveGaussPts(const std::string field_name,
//                                moab::Interface &moab_mesh,
//                                boost::shared_ptr<CommonData> common_data_ptr)
//     : DomainEleOp(field_name, DomainEleOp::OPROW),
//     internalVarMesh(moab_mesh),
//       commonDataPtr(common_data_ptr) {
//   // Operator is only executed for vertices
//   std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
// }

//! [Postprocessing]
// MoFEMErrorCode OpSaveGaussPts::doWork(int side, EntityType type,
//                                       EntData &data) {
//   MoFEMFunctionBegin;

//   auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
//   auto id = commonDataPtr->blocksIDmap.at(fe_ent);
//   auto &dAta = commonDataPtr->setOfBlocksData.at(id);
//   auto &mgis_bv = *dAta.mGisBehaviour;

//   int &size_of_vars = dAta.sizeIntVar;
//   int &size_of_grad = dAta.sizeGradVar;

//   auto get_tag = [&](std::string name, size_t size) {
//     std::array<double, 9> def;
//     std::fill(def.begin(), def.end(), 0);
//     Tag th;
//     CHKERR internalVarMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE,
//                                           th, MB_TAG_CREAT | MB_TAG_SPARSE,
//                                           def.data());
//     return th;
//   };

//   auto t_stress = getFTensor2FromMat<3, 3>(*(commonDataPtr->mStressPtr));

//   MatrixDouble3by3 mat(3, 3);
//   //FIXME: this should not be hard-coded
//   auto th_disp = get_tag("U", 3);
//   auto th_stress = get_tag(mgis_bv.thermodynamic_forces[0].name, 9);
//   auto th_grad = get_tag(mgis_bv.gradients[0].name, 9);

//   auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
//     mat.clear();
//     for (size_t r = 0; r != 3; ++r)
//       for (size_t c = 0; c != 3; ++c)
//         mat(r, c) = t(r, c);
//     return mat;
//   };

//   size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
//   auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
//   auto t_disp = getFTensor1FromMat<3>(*(commonDataPtr->mDispPtr));
//   CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, size_of_vars,
//                                        size_of_grad);
//   MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
//   vector<Tag> tags_vec;
//   bool is_large_strain = dAta.isFiniteStrain;

//   for (auto c : mgis_bv.isvs) {
//     auto vsize = getVariableSize(c, mgis_bv.hypothesis);
//     const size_t parav_siz = get_paraview_size(vsize);
//     tags_vec.emplace_back(get_tag(c.name, parav_siz));
//   }

//   if (!(side == 0 && type == MBVERTEX))
//     MoFEMFunctionReturnHot(0);

//   for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

//     double coords[] = {0, 0, 0};
//     EntityHandle vertex;
//     for (int dd = 0; dd != 3; dd++)
//       coords[dd] = getCoordsAtGaussPts()(gg, dd);

//     CHKERR internalVarMesh.create_vertex(coords, vertex);
//     VectorDouble disps({t_disp(0), t_disp(1), t_disp(2)});

//     auto it = tags_vec.begin();
//     for (auto c : mgis_bv.isvs) {
//       auto vsize = getVariableSize(c, mgis_bv.hypothesis);
//       const size_t parav_siz = get_paraview_size(vsize);
//       const auto offset =
//           getVariableOffset(mgis_bv.isvs, c.name, mgis_bv.hypothesis);
//       auto vec =
//           getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
//       VectorDouble tag_vec = getVectorAdaptor(&vec[offset], vsize);
//       tag_vec.resize(parav_siz);

//       CHKERR internalVarMesh.tag_set_data(*it, &vertex, 1,
//       &*tag_vec.begin());

//       it++;
//     }

//     // keep the convention consistent for postprocessing!
//     array<double, 9> my_stress_vec{
//         t_stress(0, 0), t_stress(1, 1), t_stress(2, 2),
//         t_stress(0, 1), t_stress(1, 0), t_stress(0, 2),
//         t_stress(2, 0), t_stress(1, 2), t_stress(2, 1)};

//     array<double, 9> grad1_vec;
//     if (is_large_strain)
//       grad1_vec = get_voigt_vec(t_grad);
//     else
//       grad1_vec = get_voigt_vec_symm(t_grad);

//     CHKERR internalVarMesh.tag_set_data(th_stress, &vertex, 1,
//                                         my_stress_vec.data());
//     CHKERR internalVarMesh.tag_set_data(th_grad, &vertex, 1,
//     grad1_vec.data()); CHKERR internalVarMesh.tag_set_data(th_disp, &vertex,
//     1, &*disps.begin());

//     ++t_grad;
//     ++t_stress;
//     ++t_disp;
//   }

//   MoFEMFunctionReturn(0);
// }

// // MoFEMErrorCode saveOutputMesh(int step, bool print_gauss) {
// //   MoFEMFunctionBegin;
// //   MoFEMFunctionReturn(0);
// // }

// OpPostProcInternalVariables::OpPostProcInternalVariables(
//     const std::string field_name, moab::Interface &post_proc_mesh,
//     std::vector<EntityHandle> &map_gauss_pts,
//     boost::shared_ptr<CommonData> common_data_ptr, int global_rule)
//     : DomainEleOp(field_name, DomainEleOp::OPROW),
//     postProcMesh(post_proc_mesh),
//       mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr),
//       globalRule(global_rule) {
//   // Operator is only executed for vertices
//   std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
// }

// MoFEMErrorCode OpPostProcInternalVariables::doWork(int side, EntityType type,
//                                                    EntData &row_data) {
//   MoFEMFunctionBegin;
//   auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
//   auto id = commonDataPtr->blocksIDmap.at(fe_ent);
//   auto &dAta = commonDataPtr->setOfBlocksData.at(id);
//   auto &mgis_bv = *dAta.mGisBehaviour;

//   dAta.setTag(RHS);
//   dAta.behDataPtr->dt = mfront_dt;
//   dAta.bView.dt = mfront_dt;

//   int &size_of_vars = dAta.sizeIntVar;
//   int &size_of_grad = dAta.sizeGradVar;
//   bool is_large_strain = dAta.isFiniteStrain;

//   int nb_rows = row_data.getIndices().size() / 3;
//   int nb_cols = row_data.getIndices().size() / 3;

//   auto get_tag = [&](std::string name, size_t size) {
//     std::array<double, 9> def;
//     std::fill(def.begin(), def.end(), 0);
//     Tag th;
//     CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE,
//     th,
//                                        MB_TAG_CREAT | MB_TAG_SPARSE,
//                                        def.data());
//     return th;
//   };
//   vector<Tag> tags_vec;
//   for (auto c : mgis_bv.isvs) {
//     auto vsize = getVariableSize(c, mgis_bv.hypothesis);
//     const size_t parav_siz = get_paraview_size(vsize);
//     tags_vec.emplace_back(get_tag(c.name, parav_siz));
//   }

//   MatrixDouble3by3 mat(3, 3);

//   auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
//     mat.clear();
//     for (size_t r = 0; r != 3; ++r)
//       for (size_t c = 0; c != 3; ++c)
//         mat(r, c) = t(r, c);
//     return mat;
//   };

//   auto inverse = [](double *A, int N) {
//     int *ipv = new int[N];
//     int lwork = N * N;
//     double *work = new double[lwork];
//     int info;
//     info = lapack_dgetrf(N, N, A, N, ipv);
//     info = lapack_dgetri(N, A, N, ipv, work, lwork);

//     delete[] ipv;
//     delete[] work;
//   };

//   int rule = globalRule;
//   auto &gaussPts = commonDataPtr->gaussPts;
//   auto &myN = commonDataPtr->myN;
//   size_t nb_gauss_pts = QUAD_3D_TABLE[rule]->npoints;

//   //FIXME: implement this for hex as well
//   if (gaussPts.size2() != nb_gauss_pts || myN.size1() != nb_gauss_pts) {

//     gaussPts.resize(4, nb_gauss_pts, false);
//     gaussPts.clear();
//     cblas_dcopy(nb_gauss_pts, &QUAD_3D_TABLE[rule]->points[1], 4,
//                 &gaussPts(0, 0), 1);
//     cblas_dcopy(nb_gauss_pts, &QUAD_3D_TABLE[rule]->points[2], 4,
//                 &gaussPts(1, 0), 1);
//     cblas_dcopy(nb_gauss_pts, &QUAD_3D_TABLE[rule]->points[3], 4,
//                 &gaussPts(2, 0), 1);
//     cblas_dcopy(nb_gauss_pts, QUAD_3D_TABLE[rule]->weights, 1, &gaussPts(3,
//     0),
//                 1);
//     myN.resize(nb_gauss_pts, 4, false);
//     myN.clear();
//     double *shape_ptr = &*myN.data().begin();
//     cblas_dcopy(4 * nb_gauss_pts, QUAD_3D_TABLE[rule]->points, 1, shape_ptr,
//     1);

//   }

//   auto set_tag = [&](auto th, auto gg, auto &mat) {
//     return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
//                                      &*mat.data().begin());
//   };

//   MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
//   auto th_stress = get_tag(mgis_bv.thermodynamic_forces[0].name, 9);
//   commonDataPtr->mStressPtr->resize(9, nb_gauss_pts, false);
//   auto t_stress = getFTensor2FromMat<3, 3>(*(commonDataPtr->mStressPtr));
//   auto &stress_mat = *commonDataPtr->mStressPtr;
//   auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));

//   auto nbg = mat_int.size1();
//   CHKERR commonDataPtr->getInternalVar(fe_ent, nbg, dAta.sizeIntVar,
//                                        dAta.sizeGradVar);

//   MatrixDouble mat_int_fs(size_of_vars, nb_rows, false);
//   MatrixDouble mat_stress_fs(9, nb_rows, false);
//   mat_int_fs.clear();
//   mat_stress_fs.clear();
//   auto &L = commonDataPtr->lsMat;

//   // calculate inverse of N^T N   only once
//   if (L.size1() != nb_rows || L.size2() != nb_cols) {
//     MatrixDouble LU(nb_rows, nb_cols, false);
//     LU.clear();
//     L = LU;

//     for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
//       double alpha = getMeasure() * gaussPts(3, gg);
//       auto shape_function =
//           getVectorAdaptor(&myN.data()[gg * nb_rows], nb_rows);

//       LU += alpha * outer_prod(shape_function, shape_function);
//     }

//     cholesky_decompose(LU, L);
//   }

//   for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
//     double alpha = getMeasure() * gaussPts(3, gg);
//     auto shape_function = getVectorAdaptor(&myN.data()[gg * nb_rows],
//     nb_rows); if (size_of_vars != 0) {

//       auto internal_var =
//           getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
//       int cc = 0;
//       for (auto int_var : internal_var) {

//         auto ls_vec_c = row(mat_int_fs, cc);
//         ls_vec_c += alpha * int_var * shape_function;
//         cc++;
//       }
//     }

//     if (is_large_strain) {
//       CHKERR mgis_integration<true, 3>(gg, t_grad, *commonDataPtr, dAta);
//       Tensor2<double, 3, 3> forces(
//           VOIGT_VEC_FULL(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
//       t_stress(i, j) = forces(i, j);

//     } else {
//       CHKERR mgis_integration<false, 3>(gg, t_grad, *commonDataPtr, dAta);
//       Tensor2_symmetric<double, 3> nstress(
//           VOIGT_VEC_SYMM(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
//       auto forces = to_non_symm(nstress);
//       t_stress(i, j) = forces(i, j);
//     }

//     for (int ii = 0; ii != 3; ++ii)
//       for (int jj = 0; jj != 3; ++jj) {
//         auto ls_vec_c = row(mat_stress_fs, 3 * ii + jj);
//         ls_vec_c += alpha * t_stress(ii, jj) * shape_function;
//       }

//     ++t_stress;
//   }

//   MatrixDouble stress_at_gauss(mapGaussPts.size(), 9, false);
//   MatrixDouble internal_var_at_gauss(mapGaussPts.size(), size_of_vars,
//   false); internal_var_at_gauss.clear(); stress_at_gauss.clear(); for (size_t
//   gg = 0; gg != mapGaussPts.size(); ++gg) {

//     if (size_of_vars != 0) {
//       for (int cc = 0; cc != size_of_vars; ++cc) {
//         auto ls_vec_c = row(mat_int_fs, cc);
//         VectorDouble field_c_vec = ls_vec_c;
//         cholesky_solve(L, field_c_vec, ublas::lower());
//         internal_var_at_gauss(gg, cc) +=
//             inner_prod(trans(row_data.getN(gg)), field_c_vec);
//       }
//     }

//     for (int ii = 0; ii != 3; ++ii)
//       for (int jj = 0; jj != 3; ++jj) {
//         auto ls_vec_c = row(mat_stress_fs, 3 * ii + jj);
//         // VectorDouble field_c_vec = prod(ls_mat, ls_vec_c);
//         VectorDouble field_c_vec = ls_vec_c;
//         cholesky_solve(L, field_c_vec, ublas::lower());
//         stress_at_gauss(gg, 3 * ii + jj) +=
//             inner_prod(trans(row_data.getN(gg)), ls_vec_c);
//         ;
//       }
//   }
//   for (size_t gg = 0; gg != mapGaussPts.size(); ++gg) {

//     auto it = tags_vec.begin();
//     for (auto c : mgis_bv.isvs) {
//       auto vsize = getVariableSize(c, mgis_bv.hypothesis);
//       const size_t parav_siz = get_paraview_size(vsize);
//       const auto offset =
//           getVariableOffset(mgis_bv.isvs, c.name, mgis_bv.hypothesis);
//       auto vec = getVectorAdaptor(
//           &internal_var_at_gauss.data()[gg * size_of_vars], size_of_vars);
//       VectorDouble tag_vec = getVectorAdaptor(&vec[offset], vsize);
//       tag_vec.resize(parav_siz);
//       CHKERR postProcMesh.tag_set_data(*it, &mapGaussPts[gg], 1,
//                                        &*tag_vec.begin());
//       it++;
//     }

//     // keep the convention consistent for postprocessing
//     array<double, 9> my_stress_vec{
//         stress_at_gauss(gg, 0), stress_at_gauss(gg, 1), stress_at_gauss(gg,
//         2), stress_at_gauss(gg, 3), stress_at_gauss(gg, 4),
//         stress_at_gauss(gg, 5), stress_at_gauss(gg, 6), stress_at_gauss(gg,
//         7), stress_at_gauss(gg, 8)};

//     CHKERR postProcMesh.tag_set_data(th_stress, &mapGaussPts[gg], 1,
//                                      my_stress_vec.data());
//   }

//   MoFEMFunctionReturn(0);
// }

//! [Body force]

// template struct OpStressTmp<true, true, 3>;
// template struct OpStressTmp<true, false, 3>;
// template struct OpStressTmp<false, false, 3>;
// template struct OpStressTmp<false, true, 3>;

template struct OpStressTmp<true, true, Hypothesis::AXISYMMETRICAL>;
template struct OpStressTmp<true, false, Hypothesis::AXISYMMETRICAL>;
template struct OpStressTmp<false, false, Hypothesis::AXISYMMETRICAL>;
template struct OpStressTmp<false, true, Hypothesis::AXISYMMETRICAL>;

// template struct OpTangent<Tensor4Pack<3>,3>;
// template struct OpTangent<DdgPack<3>,3>;

template struct OpTangent<Tensor4Pack<2>, Hypothesis::AXISYMMETRICAL>;
template struct OpTangent<DdgPack<2>, Hypothesis::AXISYMMETRICAL>;

} // namespace MFrontInterface