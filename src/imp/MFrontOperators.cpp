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

#include <MFrontOperators.hpp>
#include <MFrontMoFEMInterface.hpp>

namespace MFrontInterface {

#define VOIGT_VEC_SYMM_3D(VEC)                                                 \
  VEC[0], inv_sqr2 *VEC[3], inv_sqr2 *VEC[4], VEC[1], inv_sqr2 *VEC[5], VEC[2]

#define VOIGT_VEC_SYMM_2D(VEC) VEC[0], inv_sqr2 *VEC[3], VEC[1]

#define VOIGT_VEC_SYMM_2D_FULL(VEC)                                            \
  VEC[0], inv_sqr2 *VEC[3], 0, VEC[1], 0, VEC[2]

#define VOIGT_VEC_3D(VEC)                                                      \
  VEC[0], VEC[3], VEC[5], VEC[4], VEC[1], VEC[7], VEC[6], VEC[8], VEC[2]

#define VOIGT_VEC_2D(VEC) VEC[0], VEC[3], VEC[4], VEC[1]

#define VOIGT_VEC_2D_FULL(VEC)                                                 \
  VEC[0], VEC[3], 0, VEC[4], VEC[1], 0, 0, 0, VEC[2]

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

template struct OpStressTmp<true, true, TRIDIMENSIONAL>;
template struct OpStressTmp<true, false, TRIDIMENSIONAL>;
template struct OpStressTmp<false, false, TRIDIMENSIONAL>;
template struct OpStressTmp<false, true, TRIDIMENSIONAL>;

template struct OpStressTmp<true, true, PLANESTRAIN>;
template struct OpStressTmp<true, false, PLANESTRAIN>;
template struct OpStressTmp<false, false, PLANESTRAIN>;
template struct OpStressTmp<false, true, PLANESTRAIN>;

template struct OpStressTmp<true, true, AXISYMMETRICAL>;
template struct OpStressTmp<true, false, AXISYMMETRICAL>;
template struct OpStressTmp<false, false, AXISYMMETRICAL>;
template struct OpStressTmp<false, true, AXISYMMETRICAL>;

template struct OpSaveStress<false, TRIDIMENSIONAL>;
template struct OpSaveStress<true, TRIDIMENSIONAL>;

template struct OpSaveStress<false, PLANESTRAIN>;
template struct OpSaveStress<true, PLANESTRAIN>;

template struct OpSaveStress<false, AXISYMMETRICAL>;
template struct OpSaveStress<true, AXISYMMETRICAL>;

template struct OpSaveGaussPts<TRIDIMENSIONAL>;
template struct OpSaveGaussPts<PLANESTRAIN>;
template struct OpSaveGaussPts<AXISYMMETRICAL>;

template struct OpTangent<Tensor4Pack<3>, TRIDIMENSIONAL>;
template struct OpTangent<DdgPack<3>, TRIDIMENSIONAL>;

template struct OpTangent<Tensor4Pack<2>, PLANESTRAIN>;
template struct OpTangent<DdgPack<2>, PLANESTRAIN>;

template struct OpTangent<Tensor4Pack<2>, AXISYMMETRICAL>;
template struct OpTangent<DdgPack<2>, AXISYMMETRICAL>;

template <bool IS_LARGE_STRAIN, ModelHypothesis H>
MoFEMErrorCode OpSaveStress<IS_LARGE_STRAIN, H>::doWork(int side,
                                                        EntityType type,
                                                        EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto fe_ent =
      MFrontEleType<H>::DomainEleOp::getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  int &size_of_stress = dAta.sizeStressVar;
  int &size_of_grad = dAta.sizeGradVar;

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, dAta.sizeIntVar,
                                       dAta.sizeGradVar, dAta.sizeStressVar,
                                       IS_LARGE_STRAIN);

  MatrixDouble &mat_stress = *commonDataPtr->mPrevStressPtr;
  MatrixDouble &mat_grad = *commonDataPtr->mPrevGradPtr;

  commonDataPtr->mFullStressPtr->resize(3 * 3, nb_gauss_pts);
  commonDataPtr->mFullStressPtr->clear();
  auto t_full_stress =
      getFTensor2FromMat<3, 3>(*(commonDataPtr->mFullStressPtr));

  commonDataPtr->mFullStrainPtr->resize(3 * 3, nb_gauss_pts);
  commonDataPtr->mFullStrainPtr->clear();
  auto t_full_strain =
      getFTensor2FromMat<3, 3>(*(commonDataPtr->mFullStrainPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    auto stress_vec = getVectorAdaptor(&mat_stress.data()[gg * size_of_stress],
                                       size_of_stress);
    auto grad_vec =
        getVectorAdaptor(&mat_grad.data()[gg * size_of_grad], size_of_grad);

    Tensor2<double, 3, 3> t_stress;
    Tensor2<double, 3, 3> t_grad;

    if constexpr (IS_LARGE_STRAIN) {
      if constexpr (DIM == 3) {
        t_stress = Tensor2<double, 3, 3>(VOIGT_VEC_3D(stress_vec));
        t_grad = Tensor2<double, 3, 3>(VOIGT_VEC_3D(grad_vec));
      } else if constexpr (DIM == 2) {
        t_stress = Tensor2<double, 3, 3>(VOIGT_VEC_2D_FULL(stress_vec));
        t_grad = Tensor2<double, 3, 3>(VOIGT_VEC_2D_FULL(grad_vec));
      }
    } else {
      if constexpr (DIM == 3) {
        t_stress = to_non_symm_3d(
            Tensor2_symmetric<double, 3>(VOIGT_VEC_SYMM_3D(stress_vec)));
        t_grad = to_non_symm_3d(
            Tensor2_symmetric<double, 3>(VOIGT_VEC_SYMM_3D(grad_vec)));
      } else if constexpr (DIM == 2) {
        t_stress = to_non_symm_3d(
            Tensor2_symmetric<double, 3>(VOIGT_VEC_SYMM_2D_FULL(stress_vec)));
        t_grad = to_non_symm_3d(
            Tensor2_symmetric<double, 3>(VOIGT_VEC_SYMM_2D_FULL(grad_vec)));
      }
    }

    t_full_stress(i, j) = t_stress(i, j);
    t_full_strain(i, j) = t_grad(i, j);

    ++t_full_stress;
    ++t_full_strain;
  }

  MoFEMFunctionReturn(0);
}

template <bool UPDATE, bool IS_LARGE_STRAIN, ModelHypothesis H>
MoFEMErrorCode OpStressTmp<UPDATE, IS_LARGE_STRAIN, H>::doWork(int side,
                                                               EntityType type,
                                                               EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto fe_ent =
      MFrontEleType<H>::DomainEleOp::getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  if (monitorPtr == nullptr)
    SETERRQ(PETSC_COMM_WORLD, MOFEM_NOT_FOUND,
            "Time Monitor (FEMethod) has not been set for MFrontInterface. "
            "Make sure to call setMonitorPtr(monitor_ptr) before calling "
            "setOperators().");

  dAta.setTag(RHS);
  dAta.behDataPtr->dt = monitorPtr->ts_dt;
  dAta.bView.dt = monitorPtr->ts_dt;

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, dAta.sizeIntVar,
                                       dAta.sizeGradVar, dAta.sizeStressVar,
                                       IS_LARGE_STRAIN);

  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
  MatrixDouble &mat_grad0 = *commonDataPtr->mPrevGradPtr;
  MatrixDouble &mat_stress0 = *commonDataPtr->mPrevStressPtr;

  auto t_grad = getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->mGradPtr));
  auto t_disp = getFTensor1FromMat<DIM>(*(commonDataPtr->mDispPtr));
  auto t_coords = MFrontEleType<H>::DomainEleOp::getFTensor1CoordsAtGaussPts();

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

    if constexpr (DIM == 3) {
      Tensor2<double, 3, 3> t_force;
      if constexpr (IS_LARGE_STRAIN) {
        t_force = Tensor2<double, 3, 3>(
            VOIGT_VEC_3D(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
      } else {
        t_force = to_non_symm_3d(Tensor2_symmetric<double, 3>(
            VOIGT_VEC_SYMM_3D(getThermodynamicForce(dAta.behDataPtr->s1, 0))));
      }
      t_stress(i, j) = t_force(i, j);
    } else if constexpr (DIM == 2) {
      Tensor2<double, 2, 2> t_force;
      Tensor2<double, 3, 3> t_full_force;
      if constexpr (IS_LARGE_STRAIN) {
        t_force = Tensor2<double, 2, 2>(
            VOIGT_VEC_2D(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
        t_full_force = Tensor2<double, 3, 3>(
            VOIGT_VEC_2D_FULL(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
      } else {
        t_force = to_non_symm_2d(Tensor2_symmetric<double, 2>(
            VOIGT_VEC_SYMM_2D(getThermodynamicForce(dAta.behDataPtr->s1, 0))));
        t_full_force =
            to_non_symm_3d(Tensor2_symmetric<double, 3>(VOIGT_VEC_SYMM_2D_FULL(
                getThermodynamicForce(dAta.behDataPtr->s1, 0))));
      }
      t_stress(I, J) = t_force(I, J);
      t_full_stress(i, j) = t_full_force(i, j);
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

template <typename T, ModelHypothesis H>
MoFEMErrorCode OpTangent<T, H>::doWork(int side, EntityType type,
                                       EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto fe_ent =
      MFrontEleType<H>::DomainEleOp::getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  if (monitorPtr == nullptr)
    SETERRQ(PETSC_COMM_WORLD, MOFEM_NOT_FOUND,
            "Time Monitor (FEMethod) has not been set for MFrontInterface. "
            "Make sure to call setMonitorPtr(monitor_ptr) before calling "
            "setOperators().");

  dAta.setTag(LHS);
  dAta.behDataPtr->dt = monitorPtr->ts_dt;
  dAta.bView.dt = monitorPtr->ts_dt;

  constexpr bool IS_LARGE_STRAIN = std::is_same<T, Tensor4Pack<3>>::value ||
                                   std::is_same<T, Tensor4Pack<2>>::value;

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, dAta.sizeIntVar,
                                       dAta.sizeGradVar, dAta.sizeStressVar,
                                       IS_LARGE_STRAIN);

  MatrixDouble &S_E = *(commonDataPtr->materialTangentPtr);
  MatrixDouble &F_E = *(commonDataPtr->mFullTangentPtr);

  size_t tens_size = 36;

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

  size_t full_tens_size = 81;
  F_E.resize(full_tens_size, nb_gauss_pts, false);
  F_E.clear();

  auto D2 = get_tangent_tensor<Tensor4Pack<3>>(F_E);

  auto t_grad = getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->mGradPtr));
  auto t_disp = getFTensor1FromMat<DIM>(*(commonDataPtr->mDispPtr));
  auto t_coords = MFrontEleType<H>::DomainEleOp::getFTensor1CoordsAtGaussPts();

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    CHKERR mgis_integration<IS_LARGE_STRAIN, DIM, H>(
        gg, t_grad, t_disp, t_coords, *commonDataPtr, dAta);

    CHKERR get_tensor4_from_voigt(&*dAta.behDataPtr->K.begin(), D1);
    CHKERR get_full_tensor4_from_voigt<IS_LARGE_STRAIN>(
        &*dAta.behDataPtr->K.begin(), D2);

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
    : OpBaseImpl<PETSC, MFrontMoFEMInterface<AXISYMMETRICAL>::DomainEleOp>(
          field_name, field_name,
          MFrontMoFEMInterface<AXISYMMETRICAL>::DomainEleOp::OPROW),
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

  auto t_full_stress =
      getFTensor2FromMat<3, 3>(*(commonDataPtr->mFullStressPtr));

  // loop over integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {

    auto t_nf = getNf<2>();

    FTensor::Tensor0<double *> t_base(&row_data.getN()(gg, 0));

    // take into account Jacobian
    const double alpha = t_w * vol * 2 * M_PI;
    // loop over rows base functions
    for (int rr = 0; rr != nbRows / 2; ++rr) {

      t_nf(0) += alpha * t_full_stress(2, 2) * t_base;

      ++t_base;
      ++t_nf;
    }

    ++t_full_stress;
    ++t_w; // move to another integration weight
  }

  MoFEMFunctionReturn(0);
}

OpAxisymmetricLhs::OpAxisymmetricLhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : OpBaseImpl<PETSC, MFrontMoFEMInterface<AXISYMMETRICAL>::DomainEleOp>(
          field_name, field_name,
          MFrontMoFEMInterface<AXISYMMETRICAL>::DomainEleOp::OPROWCOL),
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

  auto t_D =
      getFTensor4FromMat<3, 3, 3, 3, 1>(*(commonDataPtr->mFullTangentPtr));

  // loop over integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {

    // Cylinder radius
    const double r_cylinder = t_coords(0);

    // take into account Jacobean
    const double alpha = t_w * vol * 2. * M_PI;

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

        t_m(0, 0) +=
            alpha * t_D(N0, N0, N2, N2) * t_col_base * t_row_diff_base(0);

        t_m(1, 0) +=
            alpha * t_D(N1, N1, N2, N2) * t_col_base * t_row_diff_base(1);

        t_m(0, 0) +=
            alpha * t_D(N2, N2, N0, N0) * t_col_diff_base(0) * t_row_base;

        t_m(0, 1) +=
            alpha * t_D(N2, N2, N1, N1) * t_col_diff_base(1) * t_row_base;

        t_m(0, 0) += alpha * (t_D(N2, N2, N2, N2) / r_cylinder) * t_col_base *
                     t_row_base;

        t_m(0, 0) +=
            alpha * t_D(N2, N2, N0, N1) * t_col_diff_base(1) * t_row_base;

        t_m(0, 1) +=
            alpha * t_D(N2, N2, N1, N0) * t_col_diff_base(0) * t_row_base;

        t_m(0, 0) +=
            alpha * t_D(N0, N1, N2, N2) * t_col_base * t_row_diff_base(1);

        t_m(1, 0) += alpha * t_D(N1, N0, N2, N2) * t_col_base *
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

template <ModelHypothesis H>
MoFEMErrorCode OpSaveGaussPts<H>::doWork(int side, EntityType type,
                                         EntData &data) {
  MoFEMFunctionBegin;

  auto fe_ent =
      MFrontEleType<H>::DomainEleOp::getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  int &size_of_vars = dAta.sizeIntVar;
  int &size_of_grad = dAta.sizeGradVar;
  int &size_of_stress = dAta.sizeStressVar;

  auto get_tag = [&](std::string name, size_t size) {
    std::array<double, 9> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR internalVarMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE,
                                          th, MB_TAG_CREAT | MB_TAG_SPARSE,
                                          def.data());
    return th;
  };

  auto t_stress = getFTensor2FromMat<3, 3>(*(commonDataPtr->mStressPtr));

  MatrixDouble3by3 mat(3, 3);
  // FIXME: this should not be hard-coded
  auto th_disp = get_tag("U", 3);
  auto th_stress = get_tag(mgis_bv.thermodynamic_forces[0].name, 9);
  auto th_grad = get_tag(mgis_bv.gradients[0].name, 9);

  // auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
  //   mat.clear();
  //   for (size_t r = 0; r != 3; ++r)
  //     for (size_t c = 0; c != 3; ++c)
  //       mat(r, c) = t(r, c);
  //   return mat;
  // };

  size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  auto t_disp = getFTensor1FromMat<3>(*(commonDataPtr->mDispPtr));
  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, size_of_vars,
                                       size_of_grad, size_of_stress);

  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
  vector<Tag> tags_vec;
  bool is_large_strain = dAta.isFiniteStrain;

  for (auto c : mgis_bv.isvs) {
    auto vsize = getVariableSize(c, mgis_bv.hypothesis);
    const size_t parav_siz = get_paraview_size(vsize);
    tags_vec.emplace_back(get_tag(c.name, parav_siz));
  }

  if (!(side == 0 && type == MBVERTEX))
    MoFEMFunctionReturnHot(0);

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    double coords[] = {0, 0, 0};
    EntityHandle vertex;
    for (int dd = 0; dd != 3; dd++)
      coords[dd] = MFrontEleType<H>::DomainEleOp::getCoordsAtGaussPts()(gg, dd);

    CHKERR internalVarMesh.create_vertex(coords, vertex);
    VectorDouble disps({t_disp(0), t_disp(1), t_disp(2)});

    auto it = tags_vec.begin();
    for (auto c : mgis_bv.isvs) {
      auto vsize = getVariableSize(c, mgis_bv.hypothesis);
      const size_t parav_siz = get_paraview_size(vsize);
      const auto offset =
          getVariableOffset(mgis_bv.isvs, c.name, mgis_bv.hypothesis);
      auto vec =
          getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
      VectorDouble tag_vec = getVectorAdaptor(&vec[offset], vsize);
      tag_vec.resize(parav_siz);

      CHKERR internalVarMesh.tag_set_data(*it, &vertex, 1, &*tag_vec.begin());

      it++;
    }

    // keep the convention consistent for postprocessing!
    array<double, 9> my_stress_vec{
        t_stress(0, 0), t_stress(1, 1), t_stress(2, 2),
        t_stress(0, 1), t_stress(1, 0), t_stress(0, 2),
        t_stress(2, 0), t_stress(1, 2), t_stress(2, 1)};

    array<double, 9> grad1_vec;
    if (is_large_strain)
      grad1_vec = get_voigt_vec(t_grad);
    else
      grad1_vec = get_voigt_vec_symm(t_grad);

    CHKERR internalVarMesh.tag_set_data(th_stress, &vertex, 1,
                                        my_stress_vec.data());
    CHKERR internalVarMesh.tag_set_data(th_grad, &vertex, 1, grad1_vec.data());
    CHKERR internalVarMesh.tag_set_data(th_disp, &vertex, 1, &*disps.begin());

    ++t_grad;
    ++t_stress;
    ++t_disp;
  }

  MoFEMFunctionReturn(0);
}

} // namespace MFrontInterface