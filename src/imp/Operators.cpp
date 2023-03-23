/**
 * @file Operators.cpp
 * @brief 
 * @date 2023-01-25
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <MoFEM.hpp>

using namespace MoFEM;
using namespace FTensor;

using EntData = EntitiesFieldData::EntData;
using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;

#include <BasicFiniteElements.hpp>
#include <quad.h>
#include <MGIS/Behaviour/Behaviour.hxx>
#include <MGIS/Behaviour/BehaviourData.hxx>
#include "MGIS/Behaviour/Integrate.hxx"
using namespace mgis;
using namespace mgis::behaviour;

#include <Operators.hpp>

namespace MFrontInterface {

#define VOIGT_VEC_SYMM(VEC)                                                    \
  VEC[0], inv_sqr2 *VEC[3], inv_sqr2 *VEC[4], VEC[1], inv_sqr2 *VEC[5], VEC[2]

#define VOIGT_VEC_FULL(VEC)                                                    \
  VEC[0], VEC[3], VEC[5], VEC[4], VEC[1], VEC[7], VEC[6], VEC[8], VEC[2]


Index<'i', 3> i;
Index<'j', 3> j;
Index<'k', 3> k;
Index<'l', 3> l;
Index<'m', 3> m;
Index<'n', 3> n;

double t_dt = 1.0;
double t_dt_prop = 1.0;

boost::shared_ptr<CommonData> commonDataPtr;

template <> Tensor4Pack get_tangent_tensor<Tensor4Pack>(MatrixDouble &mat) {
  return getFTensor4FromMat<3, 3, 3, 3>(mat);
}

template <> DdgPack get_tangent_tensor<DdgPack>(MatrixDouble &mat) {
  return getFTensor4DdgFromMat<3, 3>(mat);
}

template <bool UPDATE, bool IS_LARGE_STRAIN>
OpStressTmp<UPDATE, IS_LARGE_STRAIN>::OpStressTmp(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
template <bool UPDATE, bool IS_LARGE_STRAIN>
MoFEMErrorCode OpStressTmp<UPDATE, IS_LARGE_STRAIN>::doWork(int side,
                                                            EntityType type,
                                                            EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  dAta.setTag(RHS);
  dAta.behDataPtr->dt = t_dt;

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, dAta.sizeIntVar,
                                       dAta.sizeGradVar);

  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
  MatrixDouble &mat_grad0 = *commonDataPtr->mPrevGradPtr;
  MatrixDouble &mat_stress0 = *commonDataPtr->mPrevStressPtr;

  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  commonDataPtr->mStressPtr->resize(9, nb_gauss_pts);
  commonDataPtr->mStressPtr->clear();
  auto t_stress = getFTensor2FromMat<3, 3>(*(commonDataPtr->mStressPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    CHKERR mgis_integration<IS_LARGE_STRAIN>(gg, t_grad, *commonDataPtr, dAta);

    if constexpr (IS_LARGE_STRAIN) {
      Tensor2<double, 3, 3> forces(
          VOIGT_VEC_FULL(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
      t_stress(i, j) = forces(i, j);
    } else {
      Tensor2_symmetric<double, 3> nstress(
          VOIGT_VEC_SYMM(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
      auto forces = to_non_symm(nstress);
      t_stress(i, j) = forces(i, j);
    }

    if constexpr (UPDATE) {
      for (int dd = 0; dd != dAta.sizeIntVar; ++dd) {
        mat_int(gg, dd) = *getInternalStateVariable(dAta.behDataPtr->s1, dd);
      }
      for (int dd = 0; dd != dAta.sizeGradVar; ++dd) {
        mat_grad0(gg, dd) = *getGradient(dAta.behDataPtr->s1, dd);
        mat_stress0(gg, dd) = *getThermodynamicForce(dAta.behDataPtr->s1, dd);
      }
    }

    ++t_stress;
    ++t_grad;
  }

  if constexpr (UPDATE) {
    CHKERR commonDataPtr->setInternalVar(fe_ent);
    // t_dt_prop = t_dt * b_view.rdt;
  }

  MoFEMFunctionReturn(0);
}

template <typename T>
OpTangent<T>::OpTangent(const std::string field_name,
                        boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
template <typename T>
MoFEMErrorCode OpTangent<T>::doWork(int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  dAta.setTag(LHS);
  dAta.behDataPtr->dt = t_dt;

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, dAta.sizeIntVar,
                                       dAta.sizeGradVar);

  MatrixDouble &S_E = *(commonDataPtr->materialTangentPtr);

  size_t tens_size = 36;
  constexpr bool IS_LARGE_STRAIN = std::is_same<T, Tensor4Pack>::value;
  if (IS_LARGE_STRAIN) // for finite strains
    tens_size = 81;
  S_E.resize(tens_size, nb_gauss_pts, false);
  auto D1 = get_tangent_tensor<T>(S_E);

  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    CHKERR mgis_integration<IS_LARGE_STRAIN>(gg, t_grad, *commonDataPtr, dAta);

    CHKERR get_tensor4_from_voigt(&*dAta.behDataPtr->K.begin(), D1);

    ++D1;
    ++t_grad;
  }

  MoFEMFunctionReturn(0);
};

OpPostProcElastic::OpPostProcElastic(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
  // Operator is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode OpPostProcElastic::doWork(int side, EntityType type,
                                         EntData &data) {
  MoFEMFunctionBegin;
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  int &size_of_vars = dAta.sizeIntVar;
  int &size_of_grad = dAta.sizeGradVar;
  auto get_tag = [&](std::string name, size_t size) {
    std::array<double, 9> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);

  auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 3; ++r)
      for (size_t c = 0; c != 3; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, auto &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_grad = get_tag(mgis_bv.gradients[0].name, 9);

  size_t nb_gauss_pts1 = data.getN().size1();
  size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  Tensor2<double, 3, 3> t_stress;

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    CHKERR set_tag(th_grad, gg, set_matrix(t_grad));

    ++t_grad;
  }

  MoFEMFunctionReturn(0);
}

OpSaveGaussPts::OpSaveGaussPts(const std::string field_name,
                               moab::Interface &moab_mesh,
                               boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), internalVarMesh(moab_mesh),
      commonDataPtr(common_data_ptr) {
  // Operator is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode OpSaveGaussPts::doWork(int side, EntityType type,
                                      EntData &data) {
  MoFEMFunctionBegin;

  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  int &size_of_vars = dAta.sizeIntVar;
  int &size_of_grad = dAta.sizeGradVar;

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
  //FIXME: this should not be hard-coded
  auto th_disp = get_tag("U", 3);
  auto th_stress = get_tag(mgis_bv.thermodynamic_forces[0].name, 9);
  auto th_grad = get_tag(mgis_bv.gradients[0].name, 9);

  auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 3; ++r)
      for (size_t c = 0; c != 3; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  auto t_disp = getFTensor1FromMat<3>(*(commonDataPtr->mDispPtr));
  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, size_of_vars,
                                       size_of_grad);
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
      coords[dd] = getCoordsAtGaussPts()(gg, dd);

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

// MoFEMErrorCode saveOutputMesh(int step, bool print_gauss) {
//   MoFEMFunctionBegin;
//   MoFEMFunctionReturn(0);
// }

OpPostProcInternalVariables::OpPostProcInternalVariables(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr, int global_rule)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr),
      globalRule(global_rule) {
  // Operator is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}


MoFEMErrorCode OpPostProcInternalVariables::doWork(int side, EntityType type,
                                                   EntData &row_data) {
  MoFEMFunctionBegin;
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  auto id = commonDataPtr->blocksIDmap.at(fe_ent);
  auto &dAta = commonDataPtr->setOfBlocksData.at(id);
  auto &mgis_bv = *dAta.mGisBehaviour;

  dAta.setTag(RHS);
  dAta.behDataPtr->dt = t_dt;

  int &size_of_vars = dAta.sizeIntVar;
  int &size_of_grad = dAta.sizeGradVar;
  bool is_large_strain = dAta.isFiniteStrain;

  int nb_rows = row_data.getIndices().size() / 3;
  int nb_cols = row_data.getIndices().size() / 3;

  auto get_tag = [&](std::string name, size_t size) {
    std::array<double, 9> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };
  vector<Tag> tags_vec;
  for (auto c : mgis_bv.isvs) {
    auto vsize = getVariableSize(c, mgis_bv.hypothesis);
    const size_t parav_siz = get_paraview_size(vsize);
    tags_vec.emplace_back(get_tag(c.name, parav_siz));
  }

  MatrixDouble3by3 mat(3, 3);

  auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 3; ++r)
      for (size_t c = 0; c != 3; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto inverse = [](double *A, int N) {
    int *ipv = new int[N];
    int lwork = N * N;
    double *work = new double[lwork];
    int info;
    info = lapack_dgetrf(N, N, A, N, ipv);
    info = lapack_dgetri(N, A, N, ipv, work, lwork);

    delete[] ipv;
    delete[] work;
  };

  int rule = globalRule;
  auto &gaussPts = commonDataPtr->gaussPts;
  auto &myN = commonDataPtr->myN;
  size_t nb_gauss_pts = QUAD_3D_TABLE[rule]->npoints;

  //FIXME: implement this for hex as well
  if (gaussPts.size2() != nb_gauss_pts || myN.size1() != nb_gauss_pts) {

    gaussPts.resize(4, nb_gauss_pts, false);
    gaussPts.clear();
    cblas_dcopy(nb_gauss_pts, &QUAD_3D_TABLE[rule]->points[1], 4,
                &gaussPts(0, 0), 1);
    cblas_dcopy(nb_gauss_pts, &QUAD_3D_TABLE[rule]->points[2], 4,
                &gaussPts(1, 0), 1);
    cblas_dcopy(nb_gauss_pts, &QUAD_3D_TABLE[rule]->points[3], 4,
                &gaussPts(2, 0), 1);
    cblas_dcopy(nb_gauss_pts, QUAD_3D_TABLE[rule]->weights, 1, &gaussPts(3, 0),
                1);
    myN.resize(nb_gauss_pts, 4, false);
    myN.clear();
    double *shape_ptr = &*myN.data().begin();
    cblas_dcopy(4 * nb_gauss_pts, QUAD_3D_TABLE[rule]->points, 1, shape_ptr, 1);

  }

  auto set_tag = [&](auto th, auto gg, auto &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
  auto th_stress = get_tag(mgis_bv.thermodynamic_forces[0].name, 9);
  commonDataPtr->mStressPtr->resize(9, nb_gauss_pts, false);
  auto t_stress = getFTensor2FromMat<3, 3>(*(commonDataPtr->mStressPtr));
  auto &stress_mat = *commonDataPtr->mStressPtr;
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));

  auto nbg = mat_int.size1();
  CHKERR commonDataPtr->getInternalVar(fe_ent, nbg, dAta.sizeIntVar,
                                       dAta.sizeGradVar);

  MatrixDouble mat_int_fs(size_of_vars, nb_rows, false);
  MatrixDouble mat_stress_fs(9, nb_rows, false);
  mat_int_fs.clear();
  mat_stress_fs.clear();
  auto &L = commonDataPtr->lsMat;

  // calculate inverse of N^T N   only once
  if (L.size1() != nb_rows || L.size2() != nb_cols) {
    MatrixDouble LU(nb_rows, nb_cols, false);
    LU.clear();
    L = LU;

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      double alpha = getMeasure() * gaussPts(3, gg);
      auto shape_function =
          getVectorAdaptor(&myN.data()[gg * nb_rows], nb_rows);

      LU += alpha * outer_prod(shape_function, shape_function);
    }

    cholesky_decompose(LU, L);
  }

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    double alpha = getMeasure() * gaussPts(3, gg);
    auto shape_function = getVectorAdaptor(&myN.data()[gg * nb_rows], nb_rows);
    if (size_of_vars != 0) {

      auto internal_var =
          getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
      int cc = 0;
      for (auto int_var : internal_var) {

        auto ls_vec_c = row(mat_int_fs, cc);
        ls_vec_c += alpha * int_var * shape_function;
        cc++;
      }
    }

    if (is_large_strain) {
      CHKERR mgis_integration<true>(gg, t_grad, *commonDataPtr, dAta);
      Tensor2<double, 3, 3> forces(
          VOIGT_VEC_FULL(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
      t_stress(i, j) = forces(i, j);

    } else {
      CHKERR mgis_integration<false>(gg, t_grad, *commonDataPtr, dAta);
      Tensor2_symmetric<double, 3> nstress(
          VOIGT_VEC_SYMM(getThermodynamicForce(dAta.behDataPtr->s1, 0)));
      auto forces = to_non_symm(nstress);
      t_stress(i, j) = forces(i, j);
    }

    for (int ii = 0; ii != 3; ++ii)
      for (int jj = 0; jj != 3; ++jj) {
        auto ls_vec_c = row(mat_stress_fs, 3 * ii + jj);
        ls_vec_c += alpha * t_stress(ii, jj) * shape_function;
      }

    ++t_stress;
  }

  MatrixDouble stress_at_gauss(mapGaussPts.size(), 9, false);
  MatrixDouble internal_var_at_gauss(mapGaussPts.size(), size_of_vars, false);
  internal_var_at_gauss.clear();
  stress_at_gauss.clear();
  for (size_t gg = 0; gg != mapGaussPts.size(); ++gg) {

    if (size_of_vars != 0) {
      for (int cc = 0; cc != size_of_vars; ++cc) {
        auto ls_vec_c = row(mat_int_fs, cc);
        VectorDouble field_c_vec = ls_vec_c;
        cholesky_solve(L, field_c_vec, ublas::lower());
        internal_var_at_gauss(gg, cc) +=
            inner_prod(trans(row_data.getN(gg)), field_c_vec);
      }
    }

    for (int ii = 0; ii != 3; ++ii)
      for (int jj = 0; jj != 3; ++jj) {
        auto ls_vec_c = row(mat_stress_fs, 3 * ii + jj);
        // VectorDouble field_c_vec = prod(ls_mat, ls_vec_c);
        VectorDouble field_c_vec = ls_vec_c;
        cholesky_solve(L, field_c_vec, ublas::lower());
        stress_at_gauss(gg, 3 * ii + jj) +=
            inner_prod(trans(row_data.getN(gg)), ls_vec_c);
        ;
      }
  }
  for (size_t gg = 0; gg != mapGaussPts.size(); ++gg) {

    auto it = tags_vec.begin();
    for (auto c : mgis_bv.isvs) {
      auto vsize = getVariableSize(c, mgis_bv.hypothesis);
      const size_t parav_siz = get_paraview_size(vsize);
      const auto offset =
          getVariableOffset(mgis_bv.isvs, c.name, mgis_bv.hypothesis);          
      auto vec = getVectorAdaptor(
          &internal_var_at_gauss.data()[gg * size_of_vars], size_of_vars);
      VectorDouble tag_vec = getVectorAdaptor(&vec[offset], vsize);      
      tag_vec.resize(parav_siz);
      CHKERR postProcMesh.tag_set_data(*it, &mapGaussPts[gg], 1,
                                       &*tag_vec.begin());
      it++;
    }

    // keep the convention consistent for postprocessing
    array<double, 9> my_stress_vec{
        stress_at_gauss(gg, 0), stress_at_gauss(gg, 1), stress_at_gauss(gg, 2),
        stress_at_gauss(gg, 3), stress_at_gauss(gg, 4), stress_at_gauss(gg, 5),
        stress_at_gauss(gg, 6), stress_at_gauss(gg, 7), stress_at_gauss(gg, 8)};

    CHKERR postProcMesh.tag_set_data(th_stress, &mapGaussPts[gg], 1,
                                     my_stress_vec.data());
  }

  MoFEMFunctionReturn(0);
}


//! [Body force]

template struct OpStressTmp<true, true>;
template struct OpStressTmp<true, false>;
template struct OpStressTmp<false, false>;
template struct OpStressTmp<false, true>;

template struct OpTangent<Tensor4Pack>;
template struct OpTangent<DdgPack>;

} // namespace MFrontInterface