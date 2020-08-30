
namespace MFrontInterface {

constexpr double sqr2 = boost::math::constants::root_two<double>();
constexpr double inv_sqr2 = boost::math::constants::half_root_two<double>();

// using Tensor4Pack = Tensor4< double*, 3, 3, 3, 3>;
using Tensor4Pack = Tensor4<PackPtr<double *, 1>, 3, 3, 3, 3>;
using DdgPack = Ddg<PackPtr<double *, 1>, 3, 3>;

#define TENSOR4_K_PTR2(K)                                                      \
  &K[0], &K[1], &K[2], &K[3], &K[4], &K[5], &K[6], &K[7], &K[8], &K[9],        \
      &K[10], &K[11], &K[12], &K[13], &K[14], &K[15], &K[16], &K[17], &K[18],  \
      &K[19], &K[20], &K[21], &K[22], &K[23], &K[24], &K[25], &K[26], &K[27],  \
      &K[28], &K[29], &K[30], &K[31], &K[32], &K[33], &K[34], &K[35], &K[36],  \
      &K[37], &K[38], &K[39], &K[40], &K[41], &K[42], &K[43], &K[44], &K[45],  \
      &K[46], &K[47], &K[48], &K[49], &K[50], &K[51], &K[52], &K[53], &K[54],  \
      &K[55], &K[56], &K[57], &K[58], &K[59], &K[60], &K[61], &K[62], &K[63],  \
      &K[64], &K[65], &K[66], &K[67], &K[68], &K[69], &K[70], &K[71], &K[72],  \
      &K[73], &K[74], &K[75], &K[76], &K[77], &K[78], &K[79], &K[80]

enum DataTags { RHS = 0, LHS };

struct BodyForceData {
  Range tEts;
  VectorDouble accValues;
  double dEnsity;
  VectorDouble accValuesScaled;
  BodyForceData(VectorDouble acc_values, double density, Range ents)
      : accValues(acc_values), dEnsity(density), tEts(ents) {
    accValuesScaled = accValues;
  }
  BodyForceData() = delete;
};
extern vector<shared_ptr<BodyForceData>> body_force_vec;
extern double t_dt;

struct BlockData {
  int iD;
  int oRder;

  bool isFiniteStrain;
  string behaviourPath;
  string behaviourName;

  boost::shared_ptr<Behaviour> mGisBehaviour;
  boost::shared_ptr<BehaviourData> behDataPtr;

  int sizeIntVar;
  int sizeExtVar;

  vector<double> params;

  Range tEts;

  BlockData()
      : oRder(-1), isFiniteStrain(false), behaviourPath("src/libBehaviour.so"),
        behaviourName("IsotropicLinearHardeningPlasticity") {}

  MoFEMErrorCode setBlockBehaviourData(bool set_params_from_blocks) {
    MoFEMFunctionBeginHot;
    if (mGisBehaviour) {

      auto &mgis_bv = *mGisBehaviour;

      sizeIntVar = getArraySize(mgis_bv.isvs, mgis_bv.hypothesis);
      sizeExtVar = getArraySize(mgis_bv.esvs, mgis_bv.hypothesis);

      behDataPtr = boost::make_shared<BehaviourData>(BehaviourData{mgis_bv});

      const int total_number_of_params = mgis_bv.mps.size();
      // const int total_number_of_params = mgis_bv.mps.size() +
      // mgis_bv.params.size() + mgis_bv.iparams.size() +
      // mgis_bv.usparams.size();

      if (set_params_from_blocks) {

        // SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
        //         "Not implemented (FIXME please)");
        if (params.size() < total_number_of_params)
          SETERRQ2(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                   "Not enough parameters supplied for this block. We have %d "
                   "provided where %d are necessary for this block",
                   params.size(), total_number_of_params);
        // auto it = params.begin();
        // for (auto &p : beh_data.s0.material_properties)
        //   p = *it++;
        for (int dd = 0; dd < params.size(); + dd) {
          behDataPtr->s0.material_properties[dd] = params[dd];
          behDataPtr->s1.material_properties[dd] = params[dd];
        }
      }

      if (isFiniteStrain) {
        // rhs
        behDataPtr->K[0] = 0; // no  tangent
        behDataPtr->K[1] = 2; // PK1
        // lhs
      } else {
        // rhs
        behDataPtr->K[0] = 0; // no tangent
        behDataPtr->K[1] = 0; // cauchy
        // lhs
      }

      //  auto it = data.params.begin();
      //  for (auto &p : beh_data.s0.material_properties)
      //    p = *it++;

      // for (auto &p : mgis_bv.params)
      //   p = *it++;
      // for (auto &p : mgis_bv.iparams)
      //   p = *it++;
      // for (auto &p : mgis_bv.usparams)
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
  // array<double, 9> vec{F(0, 0), F(0, 1), F(0, 2), F(1, 0), F(1, 1),
  //                      F(1, 2), F(2, 0), F(2, 1), F(2, 2)};

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

  D(i, j, k, l) *= -1;

  MoFEMFunctionReturnHot(0);
};

struct CommonData {

  MoFEM::Interface &mField;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;
  boost::shared_ptr<MatrixDouble> mDispPtr;
  boost::shared_ptr<MatrixDouble> materialTangentPtr;
  boost::shared_ptr<MatrixDouble> internalVariablePtr;
  std::map<int, BlockData> setOfBlocksData;
  Tag internalVariableTag;

  CommonData(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;
    string block_name = "MAT";
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      // FIXME: set up a proper name
      if (it->getName().compare(0, 3, block_name) == 0) {
        std::vector<double> block_data;
        CHKERR it->getAttributes(block_data);
        int id = it->getMeshsetId();
        EntityHandle meshset = it->getMeshset();
        CHKERR mField.get_moab().get_entities_by_type(
            meshset, MBTET, setOfBlocksData[id].tEts, true);

        setOfBlocksData[id].iD = id;
        setOfBlocksData[id].params.resize(block_data.size());

        for (int n = 0; n != block_data.size(); n++)
          setOfBlocksData[id].params[n] = block_data[n];
      }
    }

    MoFEMFunctionReturn(0);
  }

  inline auto getBlockDataView(BlockData &data, DataTags tag) {
    auto &mgis_bv = *data.mGisBehaviour;
    if (tag == RHS) {
      data.behDataPtr->K[0] = 0;
      return make_view(*data.behDataPtr);

    } else {

      data.behDataPtr->K[0] = 5;
      return make_view(*data.behDataPtr);
    }
  };

  MoFEMErrorCode getInternalVar(const EntityHandle fe_ent,
                                const int nb_gauss_pts, const int var_size) {
    MoFEMFunctionBegin;
    double *tag_data;
    int tag_size;
    rval = mField.get_moab().tag_get_by_ptr(
        internalVariableTag, &fe_ent, 1, (const void **)&tag_data, &tag_size);

    if (rval != MB_SUCCESS || tag_size != var_size * nb_gauss_pts) {
      internalVariablePtr->resize(nb_gauss_pts, var_size, false);
      internalVariablePtr->clear();
      void const *tag_data[] = {&*internalVariablePtr->data().begin()};
      const int tag_size = internalVariablePtr->data().size();
      CHKERR mField.get_moab().tag_set_by_ptr(internalVariableTag, &fe_ent, 1,
                                              tag_data, &tag_size);

    } else {
      MatrixAdaptor tag_vec = MatrixAdaptor(
          nb_gauss_pts, var_size,
          ublas::shallow_array_adaptor<double>(tag_size, tag_data));

      *internalVariablePtr = tag_vec;
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setInternalVar(const EntityHandle fe_ent) {
    MoFEMFunctionBegin;
    void const *tag_data[] = {&*internalVariablePtr->data().begin()};
    const int tag_size = internalVariablePtr->data().size();
    CHKERR mField.get_moab().tag_set_by_ptr(internalVariableTag, &fe_ent, 1,
                                            tag_data, &tag_size);

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode createTags() {
    MoFEMFunctionBegin;
    double def_val = 0.0;
    const int default_length = 0;
    CHKERR mField.get_moab().tag_get_handle(
        "_INTERNAL_VAR", default_length, MB_TYPE_DOUBLE, internalVariableTag,
        MB_TAG_CREAT | MB_TAG_VARLEN | MB_TAG_SPARSE, PETSC_NULL);

    MoFEMFunctionReturn(0);
  }
};

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcVolumeOnRefinedMesh> post_proc_fe,
          boost::shared_ptr<DomainEle> update_history,
          moab::Interface &moab_mesh, bool print_gauss)
      : dM(dm), postProcFe(post_proc_fe), updateHist(update_history),
        internalVarMesh(moab_mesh), printGauss(print_gauss){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    auto make_vtks = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");

      if (printGauss) {
        string file_name =
            "out_gauss_" + boost::lexical_cast<std::string>(ts_step) + ".h5m";

        CHKERR internalVarMesh.write_file(file_name.c_str(), "MOAB",
                                          "PARALLEL=WRITE_PART");
        CHKERR internalVarMesh.delete_mesh();
      }

      MoFEMFunctionReturn(0);
    };

    CHKERR DMoFEMLoopFiniteElements(dM, "dFE", updateHist);

    CHKERR make_vtks();

    // switch (atom_test_nb) {
    // case 1: {
    //   if (ts_step == 12)
    //     if (fabs(min_disp + 9.3536) > 1e-4)
    //       SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
    //               "atom test diverged!");
    //   break;
    // }
    // default:
    //   break;
    // }

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
              boost::shared_ptr<CommonData> common_data_ptr,
              BlockData &block_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  BlockData &dAta;
};

struct OpAssembleRhs : public DomainEleOp {
  OpAssembleRhs(const std::string field_name,
                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <typename T> struct OpTangent : public DomainEleOp {
  OpTangent(const std::string field_name,
            boost::shared_ptr<CommonData> common_data_ptr,
            BlockData &block_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  BlockData &dAta;
};

template <typename T> struct OpAssembleLhs : public DomainEleOp {
  OpAssembleLhs(const std::string row_field_name,
                const std::string col_field_name,
                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  MatrixDouble locK;
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPostProcElastic : public DomainEleOp {
  OpPostProcElastic(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<CommonData> common_data_ptr,
                    BlockData &block_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
  BlockData &dAta;
};

template <typename T> T get_tangent_tensor(MatrixDouble &mat);

struct OpSaveGaussPts : public DomainEleOp {
  OpSaveGaussPts(const std::string field_name, moab::Interface &moab_mesh,
                 boost::shared_ptr<CommonData> common_data_ptr,
                 BlockData &block_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  moab::Interface &internalVarMesh;
  BlockData &dAta;
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

    CHKERR TSGetTimeStep(ts, &t_dt);
    // scale body force data
    for (auto &bdata : body_force_vec) {
      bdata->accValuesScaled = bdata->accValues;
      CHKERR MethodForForceScaling::applyScale(this, methodsOp,
                                               bdata->accValuesScaled);
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode postProcess() { return 0; }
};

struct OpBodyForceRhs : public DomainEleOp {
  OpBodyForceRhs(const std::string field_name,
             boost::shared_ptr<CommonData> common_data_ptr,
             BodyForceData &body_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  BodyForceData &bodyData;
  boost::shared_ptr<CommonData> commonDataPtr;
};


typedef struct OpAssembleLhs<Tensor4Pack> OpAssembleLhsFiniteStrains;
typedef struct OpAssembleLhs<DdgPack> OpAssembleLhsSmallStrains;

typedef struct OpTangent<Tensor4Pack> OpTangentFiniteStrains;
typedef struct OpTangent<DdgPack> OpTangentSmallStrains;

typedef struct OpStressTmp<true, true> OpUpdateVariablesFiniteStrains;
typedef struct OpStressTmp<true, false> OpUpdateVariablesSmallStrains;
typedef struct OpStressTmp<false, true> OpStressFiniteStrains;
typedef struct OpStressTmp<false, false> OpStressSmallStrains;

} // namespace MFrontInterface