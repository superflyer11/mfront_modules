
#include <MoFEM.hpp>

using namespace MoFEM;
using namespace FTensor;

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;

#include <BasicFiniteElements.hpp>

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

#define DDG_MAT_PTR(MAT)                                                       \
  &MAT(0, 0), &MAT(1, 0), &MAT(2, 0), &MAT(3, 0), &MAT(4, 0), &MAT(5, 0),      \
      &MAT(6, 0), &MAT(7, 0), &MAT(8, 0), &MAT(9, 0), &MAT(10, 0),             \
      &MAT(11, 0), &MAT(12, 0), &MAT(13, 0), &MAT(14, 0), &MAT(15, 0),         \
      &MAT(16, 0), &MAT(17, 0), &MAT(18, 0), &MAT(19, 0), &MAT(20, 0),         \
      &MAT(21, 0), &MAT(22, 0), &MAT(23, 0), &MAT(24, 0), &MAT(25, 0),         \
      &MAT(26, 0), &MAT(27, 0), &MAT(28, 0), &MAT(29, 0), &MAT(30, 0),         \
      &MAT(31, 0), &MAT(32, 0), &MAT(33, 0), &MAT(34, 0), &MAT(35, 0)

#define TENSOR4_MAT_PTR2(MAT)                                                  \
  &MAT(0, 0), &MAT(1, 0), &MAT(2, 0), &MAT(3, 0), &MAT(4, 0), &MAT(5, 0),      \
      &MAT(6, 0), &MAT(7, 0), &MAT(8, 0), &MAT(9, 0), &MAT(10, 0),             \
      &MAT(11, 0), &MAT(12, 0), &MAT(13, 0), &MAT(14, 0), &MAT(15, 0),         \
      &MAT(16, 0), &MAT(17, 0), &MAT(18, 0), &MAT(19, 0), &MAT(20, 0),         \
      &MAT(21, 0), &MAT(22, 0), &MAT(23, 0), &MAT(24, 0), &MAT(25, 0),         \
      &MAT(26, 0), &MAT(27, 0), &MAT(28, 0), &MAT(29, 0), &MAT(30, 0),         \
      &MAT(31, 0), &MAT(32, 0), &MAT(33, 0), &MAT(34, 0), &MAT(35, 0),         \
      &MAT(36, 0), &MAT(37, 0), &MAT(38, 0), &MAT(39, 0), &MAT(40, 0),         \
      &MAT(41, 0), &MAT(42, 0), &MAT(43, 0), &MAT(44, 0), &MAT(45, 0),         \
      &MAT(46, 0), &MAT(47, 0), &MAT(48, 0), &MAT(49, 0), &MAT(50, 0),         \
      &MAT(51, 0), &MAT(52, 0), &MAT(53, 0), &MAT(54, 0), &MAT(55, 0),         \
      &MAT(56, 0), &MAT(57, 0), &MAT(58, 0), &MAT(59, 0), &MAT(60, 0),         \
      &MAT(61, 0), &MAT(62, 0), &MAT(63, 0), &MAT(64, 0), &MAT(65, 0),         \
      &MAT(66, 0), &MAT(67, 0), &MAT(68, 0), &MAT(69, 0), &MAT(70, 0),         \
      &MAT(71, 0), &MAT(72, 0), &MAT(73, 0), &MAT(74, 0), &MAT(75, 0),         \
      &MAT(76, 0), &MAT(77, 0), &MAT(78, 0), &MAT(79, 0), &MAT(80, 0)

// #define TENSOR4_MAT_PTR2(MAT) &MAT(0, 0), MAT.size2()

Index<'i', 3> i;
Index<'j', 3> j;
Index<'k', 3> k;
Index<'l', 3> l;
Index<'m', 3> m;
Index<'n', 3> n;

vector<shared_ptr<BodyForceData>> body_force_vec;
double t_dt = 0;
double t_dt_prop = 0;

template <> Tensor4Pack get_tangent_tensor<Tensor4Pack>(MatrixDouble &mat) {
  return Tensor4Pack(TENSOR4_MAT_PTR2(mat));
}

template <> DdgPack get_tangent_tensor<DdgPack>(MatrixDouble &mat) {
  return DdgPack(DDG_MAT_PTR(mat));
}

template <bool UPDATE, bool IS_LARGE_STRAIN>
OpStressTmp<UPDATE, IS_LARGE_STRAIN>::OpStressTmp(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr,
    BlockData &block_data)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), dAta(block_data) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
template <bool UPDATE, bool IS_LARGE_STRAIN>
MoFEMErrorCode OpStressTmp<UPDATE, IS_LARGE_STRAIN>::doWork(int side,
                                                            EntityType type,
                                                            EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  int check_integration = 0;
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.tEts.find(fe_ent) == dAta.tEts.end())
    MoFEMFunctionReturnHot(0);
  auto &mgis_bv = *dAta.mGisBehaviour;
  auto b_view = commonDataPtr->getBlockDataView(dAta, RHS);
  int &size_of_vars = dAta.sizeIntVar;

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, size_of_vars);
  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;

  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  commonDataPtr->mStressPtr->resize(9, nb_gauss_pts);
  auto t_stress = getFTensor2FromMat<3, 3>(*(commonDataPtr->mStressPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    auto mgis_integration = [&] {
      MoFEMFunctionBeginHot;

      array<double, 9> vec;
      if (IS_LARGE_STRAIN) {
        vec = get_voigt_vec(t_grad);
        b_view.s0.gradients = vec.data();
      } else {
        vec = get_voigt_vec_symm(t_grad);
        b_view.s0.gradients = vec.data();
      }

      if (size_of_vars) {
        auto internal_var =
            getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
        b_view.s0.internal_state_variables = &*internal_var.begin();
        check_integration = integrate(b_view, mgis_bv);
      } else
        check_integration = integrate(b_view, mgis_bv);

      auto &st1 = b_view.s1.thermodynamic_forces;

      if (IS_LARGE_STRAIN) {
        Tensor2<double, 3, 3> forces(VOIGT_VEC_FULL(st1));
        t_stress(i, j) = forces(i, j);
      } else {
        Tensor2_symmetric<double, 3> nstress(VOIGT_VEC_SYMM(st1));
        auto forces = to_non_symm(nstress);
        t_stress(i, j) = forces(i, j);
      }
      MoFEMFunctionReturnHot(0);
    };
    CHKERR mgis_integration();

    if (UPDATE) // template
      for (int dd = 0; dd != size_of_vars; ++dd)
        mat_int(gg, dd) = b_view.s1.internal_state_variables[dd];

    // if (UPDATE)
    //   for (auto c : mgis_bv.isvs) {
    //     auto vsize = getVariableSize(c, mgis_bv.hypothesis);
    //     const size_t parav_siz = get_paraview_size(vsize);
    //     const auto offset =
    //         getVariableOffset(mgis_bv.isvs, c.name, mgis_bv.hypothesis);
    //     auto vec =
    //         getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
    //     auto tag_vec = getVectorAdaptor(&vec[offset], vsize);
    //     tag_vec.clear();
    //     break;
    //   }
    
    ++t_stress;
    ++t_grad;
  }

  if (UPDATE) // template
    CHKERR commonDataPtr->setInternalVar(fe_ent);

  if (check_integration < 0)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Something went wrong with MGIS integration.");

  MoFEMFunctionReturn(0);
}

OpAssembleRhs::OpAssembleRhs(const std::string field_name,
                             boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpAssembleRhs::doWork(int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  const size_t nb_gauss_pts = data.getN().size1();

  std::array<double, MAX_DOFS_ON_ENTITY> nf;
  std::fill(&nf[0], &nf[nb_dofs], 0);

  const size_t nb_base_functions = data.getN().size2();
  if (3 * nb_base_functions < nb_dofs)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Number of DOFs is larger than number of base functions on entity");

  if (nb_dofs) {

    auto t_w = getFTensor0IntegrationWeight();
    auto t_diff_base = data.getFTensor1DiffN<3>();
    auto t_stress = getFTensor2FromMat<3, 3>(*(commonDataPtr->mStressPtr));
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      double alpha = getMeasure() * t_w;
      Tensor1<PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1], &nf[2]};

      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i) += alpha * t_diff_base(j) * t_stress(i, j);
        ++t_diff_base;
        ++t_nf;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_diff_base;

      ++t_stress;
      ++t_w;
    }

    CHKERR VecSetValues(getKSPf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

template <typename T>
OpTangent<T>::OpTangent(const std::string field_name,
                        boost::shared_ptr<CommonData> common_data_ptr,
                        BlockData &block_data)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), dAta(block_data) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
template <typename T>
MoFEMErrorCode OpTangent<T>::doWork(int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  int check_integration = 0;
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.tEts.find(fe_ent) == dAta.tEts.end())
    MoFEMFunctionReturnHot(0);
  auto &mgis_bv = *dAta.mGisBehaviour;
  auto b_view = commonDataPtr->getBlockDataView(dAta, LHS);
  int &size_of_vars = dAta.sizeIntVar;
  int &size_of_evars = dAta.sizeExtVar;

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, size_of_vars);
  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
  MatrixDouble &S_E = *(commonDataPtr->materialTangentPtr);

  size_t tens_size = 36;
  if (std::is_same<T, Tensor4Pack>::value) // for finite strains
    tens_size = 81;
  S_E.resize(tens_size, nb_gauss_pts, false);
  auto D1 = get_tangent_tensor<T>(S_E);

  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    auto mgis_integration = [&] {
      MoFEMFunctionBeginHot;

      array<double, 9> vec;
      if (std::is_same<T, Tensor4Pack>::value)  {
        vec = get_voigt_vec(t_grad);
        b_view.s0.gradients = vec.data();
      } else {
        vec = get_voigt_vec_symm(t_grad);
        b_view.s0.gradients = vec.data();
      }

      if (size_of_vars) {
        auto internal_var =
            getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
        b_view.s0.internal_state_variables = &*internal_var.begin();
        check_integration = integrate(b_view, mgis_bv);
      } else
        check_integration = integrate(b_view, mgis_bv);

      MoFEMFunctionReturnHot(0);
    };
    CHKERR mgis_integration();

    if (size_of_vars) {
      auto internal_var =
          getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
      b_view.s0.internal_state_variables = &*internal_var.begin();
    }
    check_integration = integrate(b_view, mgis_bv);

    CHKERR get_tensor4_from_voigt(b_view.K, D1);

    ++D1;
    ++t_grad;
  }

  if (check_integration < 0)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Something went wrong with MGIS integration.");

  MoFEMFunctionReturn(0);
};

template <typename T>
OpAssembleLhs<T>::OpAssembleLhs(const std::string row_field_name,
                                const std::string col_field_name,
                                boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = true; // use symmetry
}
template <typename T>
MoFEMErrorCode OpAssembleLhs<T>::doWork(int row_side, int col_side,
                                        EntityType row_type,
                                        EntityType col_type, EntData &row_data,
                                        EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();

  if (nb_row_dofs && nb_col_dofs) {

    locK.resize(nb_row_dofs, nb_col_dofs, false);

    const size_t nb_gauss_pts = row_data.getN().size1();
    const size_t nb_row_base_funcs = row_data.getN().size2();
    auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
    auto t_w = getFTensor0IntegrationWeight();

    MatrixDouble &S_E = *(commonDataPtr->materialTangentPtr);
    auto D1 = get_tangent_tensor<T>(S_E);

    locK.clear();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        Tensor2<PackPtr<double *, 3>, 3, 3> t_a{

            &locK(3 * rr + 0, 0), &locK(3 * rr + 0, 1), &locK(3 * rr + 0, 2),
            &locK(3 * rr + 1, 0), &locK(3 * rr + 1, 1), &locK(3 * rr + 1, 2),
            &locK(3 * rr + 2, 0), &locK(3 * rr + 2, 1), &locK(3 * rr + 2, 2)};

        auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

        for (size_t cc = 0; cc != nb_col_dofs / 3; ++cc) {
          t_a(i, k) += alpha * (D1(i, j, k, l) *
                                (t_row_diff_base(j) * t_col_diff_base(l)));
          ++t_col_diff_base;
          ++t_a;
        }

        ++t_row_diff_base;
      }
      for (; rr != nb_row_base_funcs; ++rr)
        ++t_row_diff_base;

      ++D1;
      ++t_w;
    }

    // use symmetry
    CHKERR MatSetValues(getKSPB(), row_data, col_data, &locK(0, 0), ADD_VALUES);

    if (row_side != col_side || row_type != col_type) {
      MatrixDouble locK_trans;
      locK_trans.resize(nb_col_dofs, nb_row_dofs, false);
      noalias(locK_trans) = trans(locK);
      CHKERR MatSetValues(getKSPB(), col_data, row_data, &locK_trans(0, 0),
                          ADD_VALUES);
    }
  }
  MoFEMFunctionReturn(0);
}

OpPostProcElastic::OpPostProcElastic(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr, BlockData &block_data)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr),
      dAta(block_data) {
  // Operator is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode OpPostProcElastic::doWork(int side, EntityType type,
                                         EntData &data) {
  MoFEMFunctionBegin;
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.tEts.find(fe_ent) == dAta.tEts.end())
    MoFEMFunctionReturnHot(0);

  auto &mgis_bv = *dAta.mGisBehaviour;
  auto b_view = commonDataPtr->getBlockDataView(dAta, RHS);
  int &size_of_vars = dAta.sizeIntVar;

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
  auto th_stress =
      get_tag(mgis_bv.thermodynamic_forces[0].name + string("_AVG"), 9);

  size_t nb_gauss_pts1 = data.getN().size1();
  size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  Tensor2<double, 3, 3> t_stress;
  Tensor2<double, 3, 3> stress_avg(0., 0., 0., 0., 0., 0., 0., 0., 0.);
  int check_integration;
  // FIXME: this is risky part, it assumes that number of gauss pts is not
  // changed
  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
  auto nbg = mat_int.size1();
  CHKERR commonDataPtr->getInternalVar(fe_ent, nbg, size_of_vars);
  auto get_internal_val_avg = [&]() {
    MoFEMFunctionBeginHot;

    int &size_of_vars = dAta.sizeIntVar;

    for (auto c : mgis_bv.isvs) {
      auto vsize = getVariableSize(c, mgis_bv.hypothesis);
      // for paraview we can only use 1 3 or 9
      const size_t parav_siz = get_paraview_size(vsize);
      auto th_tag = get_tag(c.name + string("_AVG"), parav_siz);
      const auto offset =
          getVariableOffset(mgis_bv.isvs, c.name, mgis_bv.hypothesis);
      VectorDouble avg(vsize, false);
      avg.clear();
      for (int gg = 0; gg != nbg; ++gg) {
        VectorDouble vec =
            getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
        auto tag_vec = getVectorAdaptor(&vec[offset], vsize);
        avg += tag_vec;
      }
      avg /= nbg;
      avg.resize(parav_siz);

      for (int gg = 0; gg != nb_gauss_pts; ++gg)
        CHKERR set_tag(th_tag, gg, avg);
    }

    auto t_grads = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));

    for (int gg = 0; gg != nbg; ++gg) {

      auto mgis_integration = [&] {
        MoFEMFunctionBeginHot;

        array<double, 9> vec;
        if (dAta.isFiniteStrain) {
          vec = get_voigt_vec(t_grad);
          b_view.s0.gradients = vec.data();
        } else {
          vec = get_voigt_vec_symm(t_grad);
          b_view.s0.gradients = vec.data();
        }

        if (size_of_vars) {
          auto internal_var = getVectorAdaptor(
              &mat_int.data()[gg * size_of_vars], size_of_vars);
          b_view.s0.internal_state_variables = &*internal_var.begin();
          check_integration = integrate(b_view, mgis_bv);
        } else
          check_integration = integrate(b_view, mgis_bv);

        auto &st1 = b_view.s1.thermodynamic_forces;

        if (dAta.isFiniteStrain) {
          Tensor2<double, 3, 3> forces(VOIGT_VEC_FULL(st1));
          t_stress(i, j) = forces(i, j);
        } else {
          Tensor2_symmetric<double, 3> nstress(VOIGT_VEC_SYMM(st1));
          auto forces = to_non_symm(nstress);
          t_stress(i, j) = forces(i, j);
        }
        MoFEMFunctionReturnHot(0);
      };
      CHKERR mgis_integration();
      stress_avg(i, j) += t_stress(i, j);
      ++t_grads;
    }

    stress_avg(i, j) /= nbg;
    MoFEMFunctionReturnHot(0);
  };

  CHKERR get_internal_val_avg();

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    CHKERR set_tag(th_grad, gg, set_matrix(t_grad));
    CHKERR set_tag(th_stress, gg, set_matrix(stress_avg));

    ++t_grad;
  }

  MoFEMFunctionReturn(0);
}

OpSaveGaussPts::OpSaveGaussPts(const std::string field_name,
                               moab::Interface &moab_mesh,
                               boost::shared_ptr<CommonData> common_data_ptr,
                               BlockData &block_data)
    : DomainEleOp(field_name, DomainEleOp::OPROW), internalVarMesh(moab_mesh),
      commonDataPtr(common_data_ptr), dAta(block_data) {
  // Operator is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode OpSaveGaussPts::doWork(int side, EntityType type,
                                      EntData &data) {
  MoFEMFunctionBegin;

  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.tEts.find(fe_ent) == dAta.tEts.end())
    MoFEMFunctionReturnHot(0);
  auto &mgis_bv = *dAta.mGisBehaviour;
  auto b_view = commonDataPtr->getBlockDataView(dAta, RHS);
  int &size_of_vars = dAta.sizeIntVar;

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
  auto th_disp = get_tag("DISPLACEMENT", 3);
  auto th_stress = get_tag(mgis_bv.thermodynamic_forces[0].name, 9);

  auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 3; ++r)
      for (size_t c = 0; c != 3; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_tag = [&](Tag th, EntityHandle vertex, auto &mat) {
    return internalVarMesh.tag_set_data(th, &vertex, 1, &*mat.data().begin());
  };

  size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  auto t_disp = getFTensor1FromMat<3>(*(commonDataPtr->mDispPtr));
  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, size_of_vars);
  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
  vector<Tag> tags_vec;

  for (auto c : mgis_bv.isvs) {
    auto vsize = getVariableSize(c, mgis_bv.hypothesis);
    const size_t parav_siz = get_paraview_size(vsize);
    tags_vec.emplace_back(get_tag(c.name, parav_siz));
  }

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

      CHKERR set_tag(*it, vertex, tag_vec);

      it++;
    }

    CHKERR set_tag(th_disp, vertex, disps);
    CHKERR set_tag(th_stress, vertex, set_matrix(t_stress));

    ++t_grad;
    ++t_stress;
    ++t_disp;
  }

  MoFEMFunctionReturn(0);
}

OpBodyForceRhs::OpBodyForceRhs(const std::string field_name,
                               boost::shared_ptr<CommonData> common_data_ptr,
                               BodyForceData &body_data)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), bodyData(body_data) {}

//! [Body force]
MoFEMErrorCode OpBodyForceRhs::doWork(int side, EntityType type,
                                      EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
    if (bodyData.tEts.find(fe_ent) == bodyData.tEts.end())
      MoFEMFunctionReturnHot(0);

    const size_t nb_base_functions = data.getN().size2();
    if (3 * nb_base_functions < nb_dofs)
      SETERRQ(
          PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
          "Number of DOFs is larger than number of base functions on entity");

    const size_t nb_gauss_pts = data.getN().size1();
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      double alpha = getMeasure() * t_w * bodyData.dEnsity;
      auto t_acc = Tensor1<double, 3>(bodyData.accValuesScaled(0),
                                      bodyData.accValuesScaled(1),
                                      bodyData.accValuesScaled(2));

      Tensor1<PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1], &nf[2]};
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i) += alpha * t_base * t_acc(i);
        ++t_base;
        ++t_nf;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_w;
    }

    CHKERR VecSetValues(getKSPf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

template struct OpStressTmp<true, true>;
template struct OpStressTmp<true, false>;
template struct OpStressTmp<false, false>;
template struct OpStressTmp<false, true>;

template struct OpTangent<Tensor4Pack>;
template struct OpTangent<DdgPack>;

template struct OpAssembleLhs<Tensor4Pack>;
template struct OpAssembleLhs<DdgPack>;

} // namespace MFrontInterface