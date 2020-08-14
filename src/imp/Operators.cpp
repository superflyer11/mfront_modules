
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

#define VOIGHT_VEC_SYMM(VEC)                                                   \
  VEC[0], inv_sqr2 *VEC[3], inv_sqr2 *VEC[4], VEC[1], inv_sqr2 *VEC[5], VEC[2]

#define DDG_MAT_PTR(MAT)                                                       \
  &MAT(0, 0), &MAT(1, 0), &MAT(2, 0), &MAT(3, 0), &MAT(4, 0), &MAT(5, 0),      \
      &MAT(6, 0), &MAT(7, 0), &MAT(8, 0), &MAT(9, 0), &MAT(10, 0),             \
      &MAT(11, 0), &MAT(12, 0), &MAT(13, 0), &MAT(14, 0), &MAT(15, 0),         \
      &MAT(16, 0), &MAT(17, 0), &MAT(18, 0), &MAT(19, 0), &MAT(20, 0),         \
      &MAT(21, 0), &MAT(22, 0), &MAT(23, 0), &MAT(24, 0), &MAT(25, 0),         \
      &MAT(26, 0), &MAT(27, 0), &MAT(28, 0), &MAT(29, 0), &MAT(30, 0),         \
      &MAT(31, 0), &MAT(32, 0), &MAT(33, 0), &MAT(34, 0), &MAT(35, 0)

Index<'i', 3> i;
Index<'j', 3> j;
Index<'k', 3> k;
Index<'l', 3> l;
Index<'m', 3> m;
Index<'n', 3> n;

template struct OpStress<true>;
template struct OpStress<false>;

template <bool UPDATE>
OpStress<UPDATE>::OpStress(const std::string field_name,
                   boost::shared_ptr<CommonData> common_data_ptr,
                   BlockData &block_data)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), dAta(block_data) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
template <bool UPDATE>
MoFEMErrorCode OpStress<UPDATE>::doWork(int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();

  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.tEts.find(fe_ent) == dAta.tEts.end())
    MoFEMFunctionReturnHot(0);
  CHKERR commonDataPtr->getBlockData(dAta);

  // make separate material for each block
  // here we should read behaviour from dAta
  auto &mgis_bv = *commonDataPtr->mGisBehaviour;

  // local behaviour data
  auto beh_data = BehaviourData{mgis_bv};
  beh_data.K[0] = 0; // consistent tangent
  // beh_data.K[0] = 4; // consistent tangent
  // beh_data.K[1] = 2; // first Piola stress
  // beh_data.K[2] = 2; // dP / dF derivative
  beh_data.K[1] = 0; // cauchy
  auto b_view = make_view(beh_data);

  b_view.s0.material_properties = dAta.params.data();
  b_view.s1.material_properties = dAta.params.data();

  // get size of internal and external variables
  int size_of_vars = getArraySize(mgis_bv.isvs, mgis_bv.hypothesis);
  int size_of_evars = getArraySize(mgis_bv.esvs, mgis_bv.hypothesis);

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, size_of_vars);
  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;
 
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  commonDataPtr->mStressPtr->resize(9, nb_gauss_pts);
  auto t_stress = getFTensor2FromMat<3, 3>(*(commonDataPtr->mStressPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    auto vec_sym = get_voigt_vec_symm(t_grad);
    b_view.s0.gradients = vec_sym.data();

    VectorDouble internal_var = row(mat_int, gg); 
    b_view.s0.internal_state_variables = internal_var.data().data();

    int check = integrate(b_view, mgis_bv);

    auto &st1 = b_view.s1.thermodynamic_forces;
    Tensor2_symmetric<double, 3> forces(VOIGHT_VEC_SYMM(st1));
    auto my_stress = to_non_symm(forces);
    t_stress(i, j) = -my_stress(i, j);

    if (UPDATE) //template
      for (int dd = 0; dd != size_of_vars; ++dd)
        mat_int(gg, dd) = b_view.s1.internal_state_variables[dd];

    ++t_stress;
    ++t_grad;
  }
 
  if (UPDATE) //template
    CHKERR commonDataPtr->setInternalVar(fe_ent);

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

OpTangent::OpTangent(const std::string field_name,
                     boost::shared_ptr<CommonData> common_data_ptr,
                     BlockData &block_data)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), dAta(block_data) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

MoFEMErrorCode OpTangent::doWork(int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();

  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.tEts.find(fe_ent) == dAta.tEts.end())
    MoFEMFunctionReturnHot(0);
  CHKERR commonDataPtr->getBlockData(dAta);
  auto &mgis_bv = *commonDataPtr->mGisBehaviour;
  auto beh_data = BehaviourData{mgis_bv};
  beh_data.K[0] = 5;
  beh_data.K[1] = 0; 
  auto b_view = make_view(beh_data);

  // for (auto c : mgis_bv.isvs) {
  //   auto siz = getVariableSize(c, mgis_bv.hypothesis);
  //   const auto off =
  //       getVariableOffset(mgis_bv.isvs, c.name, mgis_bv.hypothesis);
  //   cout << c.name << " " << siz << " " << off << endl;
  // }
  //   VectorAdaptor tag_vec = VectorAdaptor(
  //           tag_size, ublas::shallow_array_adaptor<double>(tag_size, tag_data));

  int size_of_vars = getArraySize(mgis_bv.isvs, mgis_bv.hypothesis);
  int size_of_evars = getArraySize(mgis_bv.esvs, mgis_bv.hypothesis);

  CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, size_of_vars);
  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;

  MatrixDouble &S_E = *(commonDataPtr->materialTangent);
  S_E.resize(36, nb_gauss_pts, false);
  Ddg<PackPtr<double *, 1>, 3, 3> D1(DDG_MAT_PTR(S_E));
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));

   for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
     auto vec_sym = get_voigt_vec_symm(t_grad);
     b_view.s0.gradients = vec_sym.data();
     VectorDouble internal_var = row(mat_int, gg); 
     b_view.s0.internal_state_variables = internal_var.data().data();

     int check = integrate(b_view, mgis_bv);

     CHKERR get_ddg_from_voigt(beh_data.K, D1);

     ++D1;
     ++t_grad;
   }
  MoFEMFunctionReturn(0);
};

OpAssembleLhs::OpAssembleLhs(const std::string row_field_name,
                             const std::string col_field_name,
                             boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = true; // use symmetry
}

MoFEMErrorCode OpAssembleLhs::doWork(int row_side, int col_side,
                                     EntityType row_type, EntityType col_type,
                                     EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();

  if (nb_row_dofs && nb_col_dofs) {

    locK.resize(nb_row_dofs, nb_col_dofs, false);

    const size_t nb_gauss_pts = row_data.getN().size1();
    const size_t nb_row_base_funcs = row_data.getN().size2();
    auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
    auto t_w = getFTensor0IntegrationWeight();

    MatrixDouble &S_E = *(commonDataPtr->materialTangent);
    Ddg<PackPtr<double *, 1>, 3, 3> tangent(DDG_MAT_PTR(S_E));

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
          t_a(i, k) += alpha * (tangent(i, j, k, l) *
                                (t_row_diff_base(j) * t_col_diff_base(l)));
          ++t_col_diff_base;
          ++t_a;
        }

        ++t_row_diff_base;
      }
      for (; rr != nb_row_base_funcs; ++rr)
        ++t_row_diff_base;

      ++tangent;
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
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode OpPostProcElastic::doWork(int side, EntityType type,
                                         EntData &data) {
  MoFEMFunctionBegin;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end())
    MoFEMFunctionReturnHot(0);
  CHKERR commonDataPtr->getBlockData(dAta);

  auto &mgis_bv = *commonDataPtr->mGisBehaviour;
  // get size of internal and external variables
  int size_of_vars = getArraySize(mgis_bv.isvs, mgis_bv.hypothesis);
  // local behaviour data
  auto beh_data = BehaviourData{mgis_bv};
  beh_data.K[0] = 0;
  auto b_view = make_view(beh_data);

  auto get_tag = [&](const std::string name, size_t size = 9) {
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

  auto set_scalar = [&](auto t) -> MatrixDouble3by3 & {
    mat.clear();
    mat(0, 0) = t;
    return mat;
  };

  auto set_matrix_symm = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 3; ++r)
      for (size_t c = 0; c != 3; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_strain = get_tag("STRAIN");
  auto th_grad = get_tag("GRAD");
  auto th_stress = get_tag("STRESS");

  auto &t_D = commonDataPtr->tD;
  size_t nb_gauss_pts1 = data.getN().size1();
  size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  Tensor2_symmetric<double, 3> strain;
  Tensor2_symmetric<double, 3> stress;
  
  auto fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
  // CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts, size_of_vars);
  MatrixDouble &mat_int = *commonDataPtr->internalVariablePtr;

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    strain(i, j) = (t_grad(i, j) || t_grad(j, i)) / 2;
    // stress(i, j) = t_D(i, j, k, l) * strain(k, l);
    auto vec_sym = get_voigt_vec_symm(t_grad);
    // VectorDouble internal_var = row(mat_int, gg);
    // b_view.s0.internal_state_variables = internal_var.data().data();
    b_view.s0.gradients = vec_sym.data();
    int check = integrate(b_view, mgis_bv);
    auto &st1 = b_view.s1.thermodynamic_forces;
    Tensor2_symmetric<double, 3> stress(VOIGHT_VEC_SYMM(st1));
    CHKERR set_tag(th_grad, gg, set_matrix(t_grad));
    CHKERR set_tag(th_strain, gg, set_matrix_symm(strain));

    CHKERR set_tag(th_stress, gg, set_matrix(stress));

    ++t_grad;
  }

  MoFEMFunctionReturn(0);
}
} // namespace MFrontInterface