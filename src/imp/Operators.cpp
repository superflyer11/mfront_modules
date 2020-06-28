
#include <MoFEM.hpp>

using namespace MoFEM;
using namespace FTensor;

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;

#include <BasicFiniteElements.hpp>

// #ifdef WITH_MFRONT
#include <MGIS/Behaviour/Behaviour.hxx>
#include <MGIS/Behaviour/BehaviourData.hxx>
using namespace mgis;
using namespace mgis::behaviour;

#include <Operators.hpp>

Index<'i', 3> i;
Index<'j', 3> j;
Index<'k', 3> k;
Index<'l', 3> l;
Index<'m', 3> m;
Index<'n', 3> n;

OpAssembleRhs::OpAssembleRhs(const std::string field_name,
                             boost::shared_ptr<CommonData> common_data_ptr,
                             BlockData &block_data)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), dAta(block_data) {}

MoFEMErrorCode OpAssembleRhs::doWork(int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tEts.end()) 
      MoFEMFunctionReturnHot(0);
    CHKERR commonDataPtr->getBlockData(dAta);

    const size_t nb_base_functions = data.getN().size2();
    if (3 * nb_base_functions < nb_dofs)
      SETERRQ(
          PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
          "Number of DOFs is larger than number of base functions on entity");

    const size_t nb_gauss_pts = data.getN().size1();
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_diff_base = data.getFTensor1DiffN<3>();

    Tensor2_symmetric<double, 3> t_strain;
    Tensor2_symmetric<double, 3> t_stress;

    auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));

    auto &t_D = commonDataPtr->tD;

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      t_strain(i, j) = (t_grad(i, j) || t_grad(j, i)) / 2;
      t_stress(i, j) = t_D(i, j, k, l) * t_strain(k, l);

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

      ++t_grad;
      ++t_w;
    }

    CHKERR VecSetValues(getKSPf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpUpdateInternalVar::OpUpdateInternalVar(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr,
    BlockData &block_data)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), dAta(block_data) {}

MoFEMErrorCode OpUpdateInternalVar::doWork(int side, EntityType type,
                                           EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tEts.end()) 
      MoFEMFunctionReturnHot(0);
    CHKERR commonDataPtr->getBlockData(dAta);

    const size_t nb_base_functions = data.getN().size2();
    if (3 * nb_base_functions < nb_dofs)
      SETERRQ(
          PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
          "Number of DOFs is larger than number of base functions on entity");

    const size_t nb_gauss_pts = data.getN().size1();

    auto fe_ent = getFEEntityHandle();
    CHKERR commonDataPtr->getInternalVar(fe_ent, nb_gauss_pts);
    auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      // UPDATE INTERNAL VARIABLES HERE
      ++t_grad;
    }

    CHKERR commonDataPtr->setInternalVar(fe_ent, nb_gauss_pts);
  }

  MoFEMFunctionReturn(0);
}

OpAssembleLhs::OpAssembleLhs(const std::string row_field_name,
                             const std::string col_field_name,
                             boost::shared_ptr<CommonData> common_data_ptr,
                             BlockData &block_data)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), dAta(block_data) {
  sYmm = false;
}

//! [Stiffness]
MoFEMErrorCode OpAssembleLhs::doWork(int row_side, int col_side,
                                     EntityType row_type, EntityType col_type,
                                     EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();

  if (nb_row_dofs && nb_col_dofs) {

    if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tEts.end())
      MoFEMFunctionReturnHot(0);
    CHKERR commonDataPtr->getBlockData(dAta);

    locK.resize(nb_row_dofs, nb_col_dofs, false);

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_funcs = row_data.getN().size2();
    auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
    auto t_w = getFTensor0IntegrationWeight();
    auto &t_D = commonDataPtr->tD;

    locK.clear();
    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        Tensor2<PackPtr<double *, 3>, 3, 3> t_a{

            &locK(3 * rr + 0, 0), &locK(3 * rr + 0, 1), &locK(3 * rr + 0, 2),
            &locK(3 * rr + 1, 0), &locK(3 * rr + 1, 1), &locK(3 * rr + 1, 2),
            &locK(3 * rr + 2, 0), &locK(3 * rr + 2, 1), &locK(3 * rr + 2, 2)};

        auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

        for (size_t cc = 0; cc != nb_col_dofs / 3; ++cc) {
          t_a(i, k) += alpha * (t_D(i, j, k, l) *
                                (t_row_diff_base(j) * t_col_diff_base(l)));
          ++t_col_diff_base;
          ++t_a;
        }

        ++t_row_diff_base;
      }
      for (; rr != nb_row_base_funcs; ++rr)
        ++t_row_diff_base;

      ++t_w;
    }

    CHKERR MatSetValues(getKSPB(), row_data, col_data, &locK(0, 0), ADD_VALUES);
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
  size_t nb_gauss_pts = data.getN().size1();
  auto t_grad = getFTensor2FromMat<3, 3>(*(commonDataPtr->mGradPtr));
  Tensor2_symmetric<double, 3> strain;
  Tensor2_symmetric<double, 3> stress;

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    
      strain(i, j) = (t_grad(i, j) || t_grad(j, i)) / 2;
      stress(i, j) = t_D(i, j, k, l) * strain(k, l);

    CHKERR set_tag(th_grad, gg, set_matrix(t_grad));
    CHKERR set_tag(th_strain, gg, set_matrix_symm(strain));
    CHKERR set_tag(th_stress, gg, set_matrix_symm(stress));

    ++t_grad;
  }

  MoFEMFunctionReturn(0);
}
