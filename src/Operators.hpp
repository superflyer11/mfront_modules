/** \file Operators.hpp
 * @brief
 * @date 2023-01-25
 *
 * @copyright Copyright (c) 202
 *
 */

namespace MFrontInterface {

constexpr double sqr2 = boost::math::constants::root_two<double>();
constexpr double inv_sqr2 = boost::math::constants::half_root_two<double>();

using Tensor4Pack = Tensor4<PackPtr<double *, 1>, 3, 3, 3, 3>;
using DdgPack = Ddg<PackPtr<double *, 1>, 3, 3>;

using EntData = EntitiesFieldData::EntData;
using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;

enum DataTags { RHS = 0, LHS };

extern double t_dt;
extern double t_dt_prop;

struct BlockData {
  int iD;
  int oRder;

  bool isFiniteStrain;
  string behaviourPath;
  string behaviourName;

  boost::shared_ptr<Behaviour> mGisBehaviour;
  BehaviourDataView bView;
  boost::shared_ptr<BehaviourData> behDataPtr;

  int sizeIntVar;
  int sizeExtVar;
  int sizeGradVar;

  vector<double> params;

  double dIssipation;
  double storedEnergy;
  double externalVariable;

  vector<double> Kbuffer;
  array<double, 9> stress1Buffer;
  array<double, 9> grad1Buffer;
  VectorDouble intVar1Buffer;

  Range tEts;

  BlockData()
      : oRder(-1), isFiniteStrain(false), behaviourPath("src/libBehaviour.so"),
        behaviourName("IsotropicLinearHardeningPlasticity") {
    dIssipation = 0;
    storedEnergy = 0;
    externalVariable = 0;
  }

  MoFEMErrorCode setBlockBehaviourData(bool set_params_from_blocks) {
    MoFEMFunctionBeginHot;
    if (mGisBehaviour) {

      auto &mgis_bv = *mGisBehaviour;

      sizeIntVar = getArraySize(mgis_bv.isvs, mgis_bv.hypothesis);
      sizeExtVar = getArraySize(mgis_bv.esvs, mgis_bv.hypothesis);
      sizeGradVar = getArraySize(mgis_bv.gradients, mgis_bv.hypothesis);

      behDataPtr = boost::make_shared<BehaviourData>(BehaviourData{mgis_bv});
      bView = make_view(*behDataPtr);
      const int total_number_of_params = mgis_bv.mps.size();
      // const int total_number_of_params = mgis_bv.mps.size() +
      // mgis_bv.params.size() + mgis_bv.iparams.size() +
      // mgis_bv.usparams.size();

      if (set_params_from_blocks) {

        if (params.size() < total_number_of_params)
          SETERRQ2(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                   "Not enough parameters supplied for this block. We have %d "
                   "provided where %d are necessary for this block",
                   params.size(), total_number_of_params);

        for (int dd = 0; dd < total_number_of_params; ++dd) {
          bView.s0.material_properties[dd] = params[dd];
          bView.s1.material_properties[dd] = params[dd];
        }
      }

      intVar1Buffer.resize(sizeIntVar);
      intVar1Buffer.clear();

      if (isFiniteStrain) {
        Kbuffer.resize(81);
        bView.K = &*Kbuffer.begin();
        bView.K[0] = 0; // no  tangent
        bView.K[1] = 2; // PK1
        bView.K[2] = 2; // PK1
      } else {
        Kbuffer.resize(36);
        bView.K = &*Kbuffer.begin();
        bView.K[0] = 0; // no tangent
        bView.K[1] = 0; // cauchy
      }

      for (auto &mb : {&bView.s0, &bView.s1}) {
        mb->dissipated_energy = &dIssipation;
        mb->stored_energy = &storedEnergy;
        mb->external_state_variables = &externalVariable;
      }

      bView.s1.thermodynamic_forces = stress1Buffer.data();
      bView.s1.gradients = grad1Buffer.data();
      if (sizeIntVar > 0)
        bView.s1.internal_state_variables = &*intVar1Buffer.begin();
    }

    MoFEMFunctionReturnHot(0);
  }
};

template <typename T> inline size_t get_paraview_size(T &vsize) {
  return vsize > 1 ? (vsize > 3 ? 9 : 3) : 1;
};

template <typename T> inline auto get_voigt_vec_symm(T &t_grad) {
  Tensor2_symmetric<double, 3> t_strain;
  Index<'i', 3> i;
  Index<'j', 3> j;
  t_strain(i, j) = (t_grad(i, j) || t_grad(j, i)) / 2;

  array<double, 9> vec_sym{t_strain(0, 0),
                           t_strain(1, 1),
                           t_strain(2, 2),
                           sqr2 * t_strain(0, 1),
                           sqr2 * t_strain(0, 2),
                           sqr2 * t_strain(1, 2),
                           0,
                           0,
                           0};
  return vec_sym;
};

template <typename T> inline auto get_voigt_vec(T &t_grad) {
  Tensor2<double, 3, 3> F;
  Index<'i', 3> i;
  Index<'j', 3> j;
  F(i, j) = t_grad(i, j) + kronecker_delta(i, j);

  array<double, 9> vec{F(0, 0), F(1, 1), F(2, 2), F(0, 1), F(1, 0),
                       F(0, 2), F(2, 0), F(1, 2), F(2, 1)};

  return vec;
};

template <typename T>
inline auto to_non_symm(const Tensor2_symmetric<T, 3> &symm) {
  Tensor2<double, 3, 3> non_symm;
  Number<0> N0;
  Number<1> N1;
  Number<2> N2;
  non_symm(N0, N0) = symm(N0, N0);
  non_symm(N1, N1) = symm(N1, N1);
  non_symm(N2, N2) = symm(N2, N2);
  non_symm(N0, N1) = non_symm(N1, N0) = symm(N0, N1);
  non_symm(N0, N2) = non_symm(N2, N0) = symm(N0, N2);
  non_symm(N1, N2) = non_symm(N2, N1) = symm(N1, N2);
  return non_symm;
};

template <typename T1, typename T2>
inline MoFEMErrorCode get_tensor4_from_voigt(const T1 &K, T2 &D) {
  MoFEMFunctionBeginHot;
  Index<'i', 3> i;
  Index<'j', 3> j;
  Index<'k', 3> k;
  Index<'l', 3> l;

  Number<0> N0;
  Number<1> N1;
  Number<2> N2;

  if (std::is_same<T2, Tensor4Pack>::value) // for finite strains
  {

    D(N0, N0, N0, N0) = K[0];
    D(N0, N0, N1, N1) = K[1];
    D(N0, N0, N2, N2) = K[2];
    D(N0, N0, N0, N1) = K[3];
    D(N0, N0, N1, N0) = K[4];
    D(N0, N0, N0, N2) = K[5];
    D(N0, N0, N2, N0) = K[6];
    D(N0, N0, N1, N2) = K[7];
    D(N0, N0, N2, N1) = K[8];
    D(N1, N1, N0, N0) = K[9];
    D(N1, N1, N1, N1) = K[10];
    D(N1, N1, N2, N2) = K[11];
    D(N1, N1, N0, N1) = K[12];
    D(N1, N1, N1, N0) = K[13];
    D(N1, N1, N0, N2) = K[14];
    D(N1, N1, N2, N0) = K[15];
    D(N1, N1, N1, N2) = K[16];
    D(N1, N1, N2, N1) = K[17];
    D(N2, N2, N0, N0) = K[18];
    D(N2, N2, N1, N1) = K[19];
    D(N2, N2, N2, N2) = K[20];
    D(N2, N2, N0, N1) = K[21];
    D(N2, N2, N1, N0) = K[22];
    D(N2, N2, N0, N2) = K[23];
    D(N2, N2, N2, N0) = K[24];
    D(N2, N2, N1, N2) = K[25];
    D(N2, N2, N2, N1) = K[26];
    D(N0, N1, N0, N0) = K[27];
    D(N0, N1, N1, N1) = K[28];
    D(N0, N1, N2, N2) = K[29];
    D(N0, N1, N0, N1) = K[30];
    D(N0, N1, N1, N0) = K[31];
    D(N0, N1, N0, N2) = K[32];
    D(N0, N1, N2, N0) = K[33];
    D(N0, N1, N1, N2) = K[34];
    D(N0, N1, N2, N1) = K[35];
    D(N1, N0, N0, N0) = K[36];
    D(N1, N0, N1, N1) = K[37];
    D(N1, N0, N2, N2) = K[38];
    D(N1, N0, N0, N1) = K[39];
    D(N1, N0, N1, N0) = K[40];
    D(N1, N0, N0, N2) = K[41];
    D(N1, N0, N2, N0) = K[42];
    D(N1, N0, N1, N2) = K[43];
    D(N1, N0, N2, N1) = K[44];
    D(N0, N2, N0, N0) = K[45];
    D(N0, N2, N1, N1) = K[46];
    D(N0, N2, N2, N2) = K[47];
    D(N0, N2, N0, N1) = K[48];
    D(N0, N2, N1, N0) = K[49];
    D(N0, N2, N0, N2) = K[50];
    D(N0, N2, N2, N0) = K[51];
    D(N0, N2, N1, N2) = K[52];
    D(N0, N2, N2, N1) = K[53];
    D(N2, N0, N0, N0) = K[54];
    D(N2, N0, N1, N1) = K[55];
    D(N2, N0, N2, N2) = K[56];
    D(N2, N0, N0, N1) = K[57];
    D(N2, N0, N1, N0) = K[58];
    D(N2, N0, N0, N2) = K[59];
    D(N2, N0, N2, N0) = K[60];
    D(N2, N0, N1, N2) = K[61];
    D(N2, N0, N2, N1) = K[62];
    D(N1, N2, N0, N0) = K[63];
    D(N1, N2, N1, N1) = K[64];
    D(N1, N2, N2, N2) = K[65];
    D(N1, N2, N0, N1) = K[66];
    D(N1, N2, N1, N0) = K[67];
    D(N1, N2, N0, N2) = K[68];
    D(N1, N2, N2, N0) = K[69];
    D(N1, N2, N1, N2) = K[70];
    D(N1, N2, N2, N1) = K[71];
    D(N2, N1, N0, N0) = K[72];
    D(N2, N1, N1, N1) = K[73];
    D(N2, N1, N2, N2) = K[74];
    D(N2, N1, N0, N1) = K[75];
    D(N2, N1, N1, N0) = K[76];
    D(N2, N1, N0, N2) = K[77];
    D(N2, N1, N2, N0) = K[78];
    D(N2, N1, N1, N2) = K[79];
    D(N2, N1, N2, N1) = K[80];

  } else {

    D(N0, N0, N0, N0) = K[0];
    D(N0, N0, N1, N1) = K[1];
    D(N0, N0, N2, N2) = K[2];

    D(N0, N0, N0, N1) = inv_sqr2 * K[3];
    D(N0, N0, N0, N2) = inv_sqr2 * K[4];
    D(N0, N0, N1, N2) = inv_sqr2 * K[5];

    D(N1, N1, N0, N0) = K[6];
    D(N1, N1, N1, N1) = K[7];
    D(N1, N1, N2, N2) = K[8];

    D(N1, N1, N0, N1) = inv_sqr2 * K[9];
    D(N1, N1, N0, N2) = inv_sqr2 * K[10];
    D(N1, N1, N1, N2) = inv_sqr2 * K[11];

    D(N2, N2, N0, N0) = K[12];
    D(N2, N2, N1, N1) = K[13];
    D(N2, N2, N2, N2) = K[14];

    D(N2, N2, N0, N1) = inv_sqr2 * K[15];
    D(N2, N2, N0, N2) = inv_sqr2 * K[16];
    D(N2, N2, N1, N2) = inv_sqr2 * K[17];

    D(N0, N1, N0, N0) = inv_sqr2 * K[18];
    D(N0, N1, N1, N1) = inv_sqr2 * K[19];
    D(N0, N1, N2, N2) = inv_sqr2 * K[20];

    D(N0, N1, N0, N1) = 0.5 * K[21];
    D(N0, N1, N0, N2) = 0.5 * K[22];
    D(N0, N1, N1, N2) = 0.5 * K[23];

    D(N0, N2, N0, N0) = inv_sqr2 * K[24];
    D(N0, N2, N1, N1) = inv_sqr2 * K[25];
    D(N0, N2, N2, N2) = inv_sqr2 * K[26];

    D(N0, N2, N0, N1) = 0.5 * K[27];
    D(N0, N2, N0, N2) = 0.5 * K[28];
    D(N0, N2, N1, N2) = 0.5 * K[29];

    D(N1, N2, N0, N0) = inv_sqr2 * K[30];
    D(N1, N2, N1, N1) = inv_sqr2 * K[31];
    D(N1, N2, N2, N2) = inv_sqr2 * K[32];

    D(N1, N2, N0, N1) = 0.5 * K[33];
    D(N1, N2, N0, N2) = 0.5 * K[34];
    D(N1, N2, N1, N2) = 0.5 * K[35];
  }

  // D(i, j, k, l) *= -1;

  MoFEMFunctionReturnHot(0);
};

struct CommonData {

  MoFEM::Interface &mField;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;

  boost::shared_ptr<MatrixDouble> mPrevGradPtr;
  boost::shared_ptr<MatrixDouble> mPrevStressPtr;

  boost::shared_ptr<MatrixDouble> mDispPtr;
  boost::shared_ptr<MatrixDouble> materialTangentPtr;
  boost::shared_ptr<MatrixDouble> internalVariablePtr;

  std::map<int, BlockData> setOfBlocksData;
  std::map<EntityHandle, int> blocksIDmap;

  MatrixDouble lsMat;
  MatrixDouble gaussPts;
  MatrixDouble myN;

  Tag internalVariableTag;
  Tag stressTag;
  Tag gradientTag;

  CommonData(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;
    string block_name = "MFRONT";
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, block_name.size(), block_name) == 0) {
        std::vector<double> block_data;
        // FIXME: TODO: maybe this should be set only from the command line!!!
        CHKERR it->getAttributes(block_data);
        const int id = it->getMeshsetId();
        EntityHandle meshset = it->getMeshset();
        CHKERR mField.get_moab().get_entities_by_type(
            meshset, MBTET, setOfBlocksData[id].tEts, true);
        for (auto ent : setOfBlocksData[id].tEts)
          blocksIDmap[ent] = id;

        setOfBlocksData[id].iD = id;
        setOfBlocksData[id].params.resize(block_data.size());

        for (int n = 0; n != block_data.size(); n++)
          setOfBlocksData[id].params[n] = block_data[n];
      }
    }

    MoFEMFunctionReturn(0);
  }

  inline auto &getBlockDataView(BlockData &data, DataTags tag) {
    auto &mgis_bv = *data.mGisBehaviour;
    data.bView.dt = t_dt;
    if (tag == RHS) {
      data.bView.K[0] = 0;
    } else {
      data.bView.K[0] = 5;
    }

    return data.bView;
  };

  MoFEMErrorCode getInternalVar(const EntityHandle fe_ent,
                                const int nb_gauss_pts, const int var_size,
                                const int grad_size) {
    MoFEMFunctionBegin;

    auto mget_tag_data = [&](Tag &m_tag, boost::shared_ptr<MatrixDouble> &m_mat,
                             const int &m_size, bool is_def_grad = false) {
      MoFEMFunctionBeginHot;

      double *tag_data;
      int tag_size;
      rval = mField.get_moab().tag_get_by_ptr(
          m_tag, &fe_ent, 1, (const void **)&tag_data, &tag_size);

      if (rval != MB_SUCCESS || tag_size != m_size * nb_gauss_pts) {
        m_mat->resize(nb_gauss_pts, m_size, false);
        m_mat->clear();
        // initialize deformation gradient properly
        if (is_def_grad && m_size == 9)
          for (int gg = 0; gg != nb_gauss_pts; ++gg) {
            (*m_mat)(gg, 0) = 1;
            (*m_mat)(gg, 1) = 1;
            (*m_mat)(gg, 2) = 1;
          }
        void const *tag_data2[] = {&*m_mat->data().begin()};
        const int tag_size2 = m_mat->data().size();
        CHKERR mField.get_moab().tag_set_by_ptr(m_tag, &fe_ent, 1, tag_data2,
                                                &tag_size2);

      } else {
        MatrixAdaptor tag_vec = MatrixAdaptor(
            nb_gauss_pts, m_size,
            ublas::shallow_array_adaptor<double>(tag_size, tag_data));

        *m_mat = tag_vec;
      }

      MoFEMFunctionReturnHot(0);
    };

    CHKERR mget_tag_data(internalVariableTag, internalVariablePtr, var_size);
    CHKERR mget_tag_data(stressTag, mPrevStressPtr, grad_size);
    CHKERR mget_tag_data(gradientTag, mPrevGradPtr, grad_size, true);

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setInternalVar(const EntityHandle fe_ent) {
    MoFEMFunctionBegin;

    auto mset_tag_data = [&](Tag &m_tag,
                             boost::shared_ptr<MatrixDouble> &m_mat) {
      MoFEMFunctionBeginHot;
      void const *tag_data[] = {&*m_mat->data().begin()};
      const int tag_size = m_mat->data().size();
      CHKERR mField.get_moab().tag_set_by_ptr(m_tag, &fe_ent, 1, tag_data,
                                              &tag_size);
      MoFEMFunctionReturnHot(0);
    };

    CHKERR mset_tag_data(internalVariableTag, internalVariablePtr);
    CHKERR mset_tag_data(stressTag, mPrevStressPtr);
    CHKERR mset_tag_data(gradientTag, mPrevGradPtr);

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode createTags() {
    MoFEMFunctionBegin;
    double def_val = 0.0;
    const int default_length = 0;
    CHKERR mField.get_moab().tag_get_handle(
        "_INTERNAL_VAR", default_length, MB_TYPE_DOUBLE, internalVariableTag,
        MB_TAG_CREAT | MB_TAG_VARLEN | MB_TAG_SPARSE, PETSC_NULL);
    CHKERR mField.get_moab().tag_get_handle(
        "_STRESS_TAG", default_length, MB_TYPE_DOUBLE, stressTag,
        MB_TAG_CREAT | MB_TAG_VARLEN | MB_TAG_SPARSE, PETSC_NULL);
    CHKERR mField.get_moab().tag_get_handle(
        "_GRAD_TAG", default_length, MB_TYPE_DOUBLE, gradientTag,
        MB_TAG_CREAT | MB_TAG_VARLEN | MB_TAG_SPARSE, PETSC_NULL);

    MoFEMFunctionReturn(0);
  }
};
extern boost::shared_ptr<CommonData> commonDataPtr;

// MoFEMErrorCode saveOutputMesh(int step, bool print_gauss);

template <bool IS_LARGE_STRAIN>
inline MoFEMErrorCode mgis_integration(
    size_t gg, FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 3, 3> &t_grad,
    CommonData &common_data, BlockData &block_data, BehaviourDataView &b_view) {
  MoFEMFunctionBegin;
  int check_integration;
  MatrixDouble &mat_int = *common_data.internalVariablePtr;
  MatrixDouble &mat_grad0 = *common_data.mPrevGradPtr;
  MatrixDouble &mat_stress0 = *common_data.mPrevStressPtr;

  int &size_of_vars = block_data.sizeIntVar;
  int &size_of_grad = block_data.sizeGradVar;
  auto &mgis_bv = *block_data.mGisBehaviour;

  auto grad0_vec =
      getVectorAdaptor(&mat_grad0.data()[gg * size_of_grad], size_of_grad);
  if (IS_LARGE_STRAIN)
    block_data.grad1Buffer = get_voigt_vec(t_grad);
  else
    block_data.grad1Buffer = get_voigt_vec_symm(t_grad);

  b_view.s0.gradients = &*grad0_vec.begin();

  auto stress0_vec =
      getVectorAdaptor(&mat_stress0.data()[gg * size_of_grad], size_of_grad);

  b_view.s0.thermodynamic_forces = &*stress0_vec.begin();

  if (size_of_vars) {
    auto internal_var =
        getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
    b_view.s0.internal_state_variables = &*internal_var.begin();
    check_integration = integrate(b_view, mgis_bv);
  } else
    check_integration = integrate(b_view, mgis_bv);
  // FIXME: this should be handled somehow
  // if (check_integration < 0)
  //   SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
  //           "Something went wrong with MGIS integration.");

  MoFEMFunctionReturn(0);
}

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcVolumeOnRefinedMesh> post_proc_fe,
          boost::shared_ptr<DomainEle> update_history,
          moab::Interface &moab_mesh, bool print_gauss)
      : dM(dm), postProcFe(post_proc_fe), updateHist(update_history),
        internalVarMesh(moab_mesh), printGauss(print_gauss){};

  MoFEMErrorCode preProcess() {

    CHKERR TSGetTimeStep(ts, &t_dt);
    return 0;
  }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    // CHKERR saveOutputMesh(ts_step);

    CHKERR TSSetTimeStep(ts, t_dt_prop);

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProcFe;
  boost::shared_ptr<DomainEle> updateHist;
  moab::Interface &internalVarMesh;
  bool printGauss;
};

template <bool UPDATE, bool IS_LARGE_STRAIN>
struct OpStressTmp : public DomainEleOp {
  OpStressTmp(const std::string field_name,
              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <typename T> struct OpTangent : public DomainEleOp {
  OpTangent(const std::string field_name,
            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPostProcElastic : public DomainEleOp {
  OpPostProcElastic(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPostProcInternalVariables : public DomainEleOp {
  OpPostProcInternalVariables(const std::string field_name,
                              moab::Interface &post_proc_mesh,
                              std::vector<EntityHandle> &map_gauss_pts,
                              boost::shared_ptr<CommonData> common_data_ptr,
                              int global_rule);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  // MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
  //                       EntityType col_type, EntData &row_data,
  //                       EntData &col_data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
  int globalRule;
};

template <typename T> T get_tangent_tensor(MatrixDouble &mat);

struct OpSaveGaussPts : public DomainEleOp {
  OpSaveGaussPts(const std::string field_name, moab::Interface &moab_mesh,
                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  moab::Interface &internalVarMesh;
};

struct FePrePostProcess : public FEMethod {

  boost::ptr_vector<MethodForForceScaling> methodsOp;

  // FePrePostProcess(){}

  MoFEMErrorCode preProcess() {
    //
    MoFEMFunctionBegin;
    switch (ts_ctx) {
    case CTX_TSSETIJACOBIAN: {
      snes_ctx = CTX_SNESSETJACOBIAN;
      snes_B = ts_B;
      break;
    }
    case CTX_TSSETIFUNCTION: {
      snes_ctx = CTX_SNESSETFUNCTION;
      snes_f = ts_F;
      break;
    }
    default:
      break;
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode postProcess() { return 0; }
};

typedef struct OpTangent<Tensor4Pack> OpTangentFiniteStrains;
typedef struct OpTangent<DdgPack> OpTangentSmallStrains;

typedef struct OpStressTmp<true, true> OpUpdateVariablesFiniteStrains;
typedef struct OpStressTmp<true, false> OpUpdateVariablesSmallStrains;
typedef struct OpStressTmp<false, true> OpStressFiniteStrains;
typedef struct OpStressTmp<false, false> OpStressSmallStrains;

} // namespace MFrontInterface