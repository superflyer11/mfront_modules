/** \file MFrontOperators.hpp
 * @brief
 * @date 2023-01-25
 *
 * @copyright Copyright (c) 202
 *
 */

#include <MFrontMoFEMInterface.hpp>

namespace MFrontInterface {

constexpr double sqr2 = boost::math::constants::root_two<double>();
constexpr double inv_sqr2 = boost::math::constants::half_root_two<double>();

template <int DIM>
using Tensor4Pack = Tensor4<PackPtr<double *, 1>, DIM, DIM, DIM, DIM>;

template <int DIM>
using Tensor2Pack = Tensor2<PackPtr<double *, 1>, DIM, DIM>;

template <int DIM>
using DdgPack = Ddg<PackPtr<double *, 1>, DIM, DIM>;

// using EntData = EntitiesFieldData::EntData;
// using DomainEle = VolumeElementForcesAndSourcesCore;
// using DomainEleOp = DomainEle::UserDataOperator;

enum DataTags { RHS = 0, LHS };


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
  int sizeStressVar;

  vector<double> params;

  double dIssipation;
  double storedEnergy;
  double externalVariable;

  Range tEts;

  BlockData()
      : oRder(-1), isFiniteStrain(false), behaviourPath("src/libBehaviour.so"),
        behaviourName("IsotropicLinearHardeningPlasticity") {
    dIssipation = 0;
    storedEnergy = 0;
    externalVariable = 0;
  }

  inline MoFEMErrorCode setTag(DataTags tag) {
    MoFEMFunctionBeginHot;
    if (tag == RHS) {
      behDataPtr->K[0] = 0;
    } else {
      behDataPtr->K[0] = 5;
    }
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode setBlockBehaviourData(bool set_params_from_blocks) {
    MoFEMFunctionBeginHot;
    if (mGisBehaviour) {

      auto &mgis_bv = *mGisBehaviour;

      sizeIntVar = getArraySize(mgis_bv.isvs, mgis_bv.hypothesis);
      sizeExtVar = getArraySize(mgis_bv.esvs, mgis_bv.hypothesis);
      sizeGradVar = getArraySize(mgis_bv.gradients, mgis_bv.hypothesis);
      sizeStressVar =
          getArraySize(mgis_bv.thermodynamic_forces, mgis_bv.hypothesis);

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
          setMaterialProperty(behDataPtr->s0, dd, params[dd]);
          setMaterialProperty(behDataPtr->s1, dd, params[dd]);
        }
      }

      if (isFiniteStrain) {
        behDataPtr->K[0] = 0; // no tangent
        behDataPtr->K[1] = 2; // PK1
        behDataPtr->K[2] = 2; // PK1
      } else {
        behDataPtr->K[0] = 0; // no tangent
        behDataPtr->K[1] = 0; // cauchy
      }

      for (auto &mb : {&behDataPtr->s0, &behDataPtr->s1}) {
        mb->dissipated_energy = dIssipation;
        mb->stored_energy = storedEnergy;
        setExternalStateVariable(*mb, 0, externalVariable);
      }
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

template <typename T> inline auto get_voigt_vec_symm_plane_strain(T &t_grad) {
  Tensor2_symmetric<double, 2> t_strain;
  Index<'i', 2> i;
  Index<'j', 2> j;
  t_strain(i, j) = (t_grad(i, j) || t_grad(j, i)) / 2;

  array<double, 4> vec_sym{t_strain(0, 0), t_strain(1, 1), 0,
                           sqr2 * t_strain(0, 1)};
  return vec_sym;
};

template <typename T> inline auto get_voigt_vec(T &t_grad) {
  Tensor2<double, 3, 3> F;
  Index<'i', 3> i;
  Index<'j', 3> j;
  F(i, j) = t_grad(i, j) + kronecker_delta(i, j);

  // bool flag = std::is_same<T, Tensor4Pack>::value;
  // cout << flag << endl;

  // double det;
  // CHKERR determinantTensor3by3(F, det);
  // if (det < 0)
  //   MOFEM_LOG("WORLD", Sev::error) << "NEGATIVE DET!!!" << det;

  array<double, 9> vec{F(0, 0), F(1, 1), F(2, 2), F(0, 1), F(1, 0),
                       F(0, 2), F(2, 0), F(1, 2), F(2, 1)};

  return vec;
};

template <typename T> inline auto get_voigt_vec_plane_strain(T &t_grad) {
  Tensor2<double, 2, 2> F;
  Index<'i', 2> i;
  Index<'j', 2> j;
  F(i, j) = t_grad(i, j) + kronecker_delta(i, j);

  // bool flag = std::is_same<T, Tensor4Pack>::value;
  // cout << flag << endl;

  // double det;
  // CHKERR determinantTensor3by3(F, det);
  // if (det < 0)
  //   MOFEM_LOG("WORLD", Sev::error) << "NEGATIVE DET!!!" << det;

  array<double, 5> vec{F(0, 0), F(1, 1), 1, F(0, 1), F(1, 0)};

  return vec;
};

// template deduction
// template <typename T, int DIM> struct GetVoightVecImpl;

// template <typename T, int DIM> struct GetVoightVecImpl<FTensor::Tensor2<T, 3, 3>> {
//   static inline auto get(FTensor::Tensor2<T, 3, 3> &t_grad) {
//     Tensor2<double, 3, 3> F;
//     Index<'i', 3> i;
//     Index<'j', 3> j;
//     F(i, j) = t_grad(i, j) + kronecker_delta(i, j);

//     // double det;
//     // CHKERR determinantTensor3by3(F, det);
//     // if (det < 0)
//     //   MOFEM_LOG("WORLD", Sev::error) << "NEGATIVE DET!!!" << det;

//     array<double, 9> vec{F(0, 0), F(1, 1), F(2, 2), F(0, 1), F(1, 0),
//                          F(0, 2), F(2, 0), F(1, 2), F(2, 1)};

//     return vec;
//   }
// };

// template <typename T, int DIM> inline auto get_voigt_vec(T &t_grad) {
//   return GetVoightVecImpl<T, DIM>::get(t_grad);
// };

// // template <typename V>
// // struct GetVoightVecImpl<FTensor::Tensor2<V,2, 2>> {
// //   static inline auto get(FTensor::Tensor2<V,2, 2> &t_grad) {
// //   Tensor2<double, 3, 3> F;
// //   Index<'i', 3> i;
// //   Index<'j', 3> j;
// //   F(i, j) = t_grad(i, j) + kronecker_delta(i, j);

// //   // double det;
// //   // CHKERR determinantTensor3by3(F, det);
// //   // if (det < 0)
// //   //   MOFEM_LOG("WORLD", Sev::error) << "NEGATIVE DET!!!" << det;

// //   array<double, 9> vec{F(0, 0), F(1, 1), F(2, 2), F(0, 1), F(1, 0),
// //                        F(0, 2), F(2, 0), F(1, 2), F(2, 1)};

// //   return vec;
// // }
// // };
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

template <typename T>
inline auto to_non_symm_plane_strain(const Tensor2_symmetric<T, 2> &symm) {
  Tensor2<double, 2, 2> non_symm;
  Number<0> N0;
  Number<1> N1;
  non_symm(N0, N0) = symm(N0, N0);
  non_symm(N1, N1) = symm(N1, N1);
  non_symm(N0, N1) = non_symm(N1, N0) = symm(N0, N1);
  return non_symm;
};

template <typename T1, typename T2>
inline MoFEMErrorCode get_tensor4_from_voigt(const T1 &K, T2 &D) {
  MoFEMFunctionBeginHot;
  // Index<'i', 3> i;
  // Index<'j', 3> j;
  // Index<'k', 3> k;
  // Index<'l', 3> l;

  Number<0> N0;
  Number<1> N1;
  Number<2> N2;

  if (std::is_same<T2, Tensor4Pack<3>>::value) { // 3D finite strain
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
  } 

  if (std::is_same<T2, Tensor4Pack<2>>::value) { // plane strain finite strain
    D(N0, N0, N0, N0) = K[0];
    D(N0, N0, N1, N1) = K[1];
    // D(N0, N0, N2, N2) = K[2];
    D(N0, N0, N0, N1) = K[3];
    D(N0, N0, N1, N0) = K[4];
    D(N1, N1, N0, N0) = K[5];
    D(N1, N1, N1, N1) = K[6];
    // D(N1, N1, N2, N2) = K[7];
    D(N1, N1, N0, N1) = K[8];
    D(N1, N1, N1, N0) = K[9];
    // D(N2, N2, N0, N0) = K[10];
    // D(N2, N2, N1, N1) = K[11];
    // D(N2, N2, N2, N2) = K[12];
    // D(N2, N2, N0, N1) = K[13];
    // D(N2, N2, N1, N0) = K[14];
    D(N0, N1, N0, N0) = K[15];
    D(N0, N1, N1, N1) = K[16];
    // D(N0, N1, N2, N2) = K[17];
    D(N0, N1, N0, N1) = K[18];
    D(N0, N1, N1, N0) = K[19];
    D(N1, N0, N0, N0) = K[20];
    D(N1, N0, N1, N1) = K[21];
    // D(N1, N0, N2, N2) = K[22];
    D(N1, N0, N0, N1) = K[23];
    D(N1, N0, N1, N0) = K[24];
  } 

  if (std::is_same<T2, DdgPack<3>>::value) { // 3D small strain

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

  if (std::is_same<T2, DdgPack<2>>::value) { // plane strain, small strain

    D(N0, N0, N0, N0) = K[0];
    D(N0, N0, N1, N1) = K[1];
    // D(N0, N0, N2, N2) = K[2];

    D(N0, N0, N0, N1) = inv_sqr2 * K[3];

    D(N1, N1, N0, N0) = K[4];
    D(N1, N1, N1, N1) = K[5];
    // D(N1, N1, N2, N2) = K[6];

    D(N1, N1, N0, N1) = inv_sqr2 * K[7];

    // D(N2, N2, N0, N0) = K[8];
    // D(N2, N2, N1, N1) = K[9];
    // D(N2, N2, N2, N2) = K[10];

    // D(N2, N2, N0, N1) = inv_sqr2 * K[11];

    D(N0, N1, N0, N0) = inv_sqr2 * K[12];
    D(N0, N1, N1, N1) = inv_sqr2 * K[13];
    // D(N0, N1, N2, N2) = inv_sqr2 * K[14];

    D(N0, N1, N0, N1) = 0.5 * K[15];

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

  MoFEMErrorCode setBlocks(int dim) {
    MoFEMFunctionBegin;
    string block_name = "MFRONT";
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, block_name.size(), block_name) == 0) {
        std::vector<double> block_data;
        // FIXME: TODO: maybe this should be set only from the command line!!!
        CHKERR it->getAttributes(block_data);
        const int id = it->getMeshsetId();
        EntityHandle meshset = it->getMeshset();
        CHKERR mField.get_moab().get_entities_by_dimension(
            meshset, dim, setOfBlocksData[id].tEts, true);
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
  //FIXME: pass stress_size
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

  MoFEMErrorCode clearTags() {
    MoFEMFunctionBegin;
    double zero = 0;
    for (auto &[id, data] : setOfBlocksData) {
      CHKERR mField.get_moab().tag_clear_data(internalVariableTag, data.tEts,
                                              &zero);
      CHKERR mField.get_moab().tag_clear_data(stressTag, data.tEts, &zero);
      CHKERR mField.get_moab().tag_clear_data(gradientTag, data.tEts, &zero);
    }
    MoFEMFunctionReturn(0);
  }
};
extern boost::shared_ptr<CommonData> commonDataPtr;

// MoFEMErrorCode saveOutputMesh(int step, bool print_gauss);

template <bool IS_LARGE_STRAIN, int DIM>
inline MoFEMErrorCode mgis_integration(size_t gg, Tensor2Pack<DIM> &t_grad,
                                       CommonData &common_data,
                                       BlockData &block_data) {
  MoFEMFunctionBegin;
  int check_integration;
  MatrixDouble &mat_int = *common_data.internalVariablePtr;
  MatrixDouble &mat_grad0 = *common_data.mPrevGradPtr;
  MatrixDouble &mat_stress0 = *common_data.mPrevStressPtr;

  int &size_of_vars = block_data.sizeIntVar;
  int &size_of_grad = block_data.sizeGradVar;
  int &size_of_stress = block_data.sizeStressVar;
  
  auto &mgis_bv = *block_data.mGisBehaviour;

  if (IS_LARGE_STRAIN) {
    if (DIM == 3)
      setGradient(block_data.behDataPtr->s1, 0, size_of_grad,
                  &*get_voigt_vec(t_grad).data());
    if (DIM == 2)
      setGradient(block_data.behDataPtr->s1, 0, size_of_grad,
                  &*get_voigt_vec_plane_strain(t_grad).data());
  } else {
    if (DIM == 3)
      setGradient(block_data.behDataPtr->s1, 0, size_of_grad,
                  &*get_voigt_vec_symm(t_grad).data());
    if (DIM == 2)
      setGradient(block_data.behDataPtr->s1, 0, size_of_grad,
                  &*get_voigt_vec_symm_plane_strain(t_grad).data());
  }

  auto grad0_vec =
      getVectorAdaptor(&mat_grad0.data()[gg * size_of_grad], size_of_grad);
  setGradient(block_data.behDataPtr->s0, 0, size_of_grad, &*grad0_vec.begin());

  auto stress0_vec = getVectorAdaptor(&mat_stress0.data()[gg * size_of_stress],
                                      size_of_stress);
  setThermodynamicForce(block_data.behDataPtr->s0, 0, size_of_stress,
                        &*stress0_vec.begin());

  if (size_of_vars) {
    auto internal_var =
        getVectorAdaptor(&mat_int.data()[gg * size_of_vars], size_of_vars);
    setInternalStateVariable(block_data.behDataPtr->s0, 0, size_of_vars,
                             &*internal_var.begin());
  }

  check_integration = integrate(block_data.bView, mgis_bv);
  switch (check_integration) {
  case -1:
    SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
            "MFront integration failed");
    break;
  case 0:
    MOFEM_LOG("WORLD", Sev::inform)
        << "Mfront integration succeeded but results are unreliable";
    break;
  case 1:
  default:
    break;
  }

  MoFEMFunctionReturn(0);
}

// struct Monitor : public FEMethod {

//   Monitor(SmartPetscObj<DM> &dm,
//           boost::shared_ptr<PostProcVolumeOnRefinedMesh> post_proc_fe,
//           boost::shared_ptr<DomainEle> update_history,
//           moab::Interface &moab_mesh, bool print_gauss)
//       : dM(dm), postProcFe(post_proc_fe), updateHist(update_history),
//         internalVarMesh(moab_mesh), printGauss(print_gauss){};

//   MoFEMErrorCode preProcess() {

//     CHKERR TSGetTimeStep(ts, &t_dt);
//     return 0;
//   }
//   MoFEMErrorCode operator()() { return 0; }

//   MoFEMErrorCode postProcess() {
//     MoFEMFunctionBegin;

//     // CHKERR saveOutputMesh(ts_step);

//     CHKERR TSSetTimeStep(ts, t_dt_prop);

//     MoFEMFunctionReturn(0);
//   }

// private:
//   SmartPetscObj<DM> dM;
//   boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProcFe;
//   boost::shared_ptr<DomainEle> updateHist;
//   moab::Interface &internalVarMesh;
//   bool printGauss;
// };

template <typename T> T get_tangent_tensor(MatrixDouble &mat);

template <bool UPDATE, bool IS_LARGE_STRAIN>
struct OpStressTmp : public MFrontMoFEMInterface<PLANESTRAIN>::DomainEleOp {
  static constexpr int DIM = MFrontEleType<PLANESTRAIN>::SPACE_DIM;

  OpStressTmp(const std::string field_name,
              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <typename T> struct OpTangent : public MFrontMoFEMInterface<PLANESTRAIN>::DomainEleOp {
  static constexpr int DIM = MFrontEleType<PLANESTRAIN>::SPACE_DIM;
  
  OpTangent(const std::string field_name,
            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

// struct OpPostProcElastic : public MFrontMoFEMInterface::DomainEleOp {
//   OpPostProcElastic(const std::string field_name,
//                     moab::Interface &post_proc_mesh,
//                     std::vector<EntityHandle> &map_gauss_pts,
//                     boost::shared_ptr<CommonData> common_data_ptr);
//   MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

// private:
//   moab::Interface &postProcMesh;
//   std::vector<EntityHandle> &mapGaussPts;
//   boost::shared_ptr<CommonData> commonDataPtr;
// };

// struct OpPostProcInternalVariables : public MFrontMoFEMInterface::DomainEleOp {
//   OpPostProcInternalVariables(const std::string field_name,
//                               moab::Interface &post_proc_mesh,
//                               std::vector<EntityHandle> &map_gauss_pts,
//                               boost::shared_ptr<CommonData> common_data_ptr,
//                               int global_rule);
//   MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
//   // MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
//   //                       EntityType col_type, EntData &row_data,
//   //                       EntData &col_data);

// private:
//   moab::Interface &postProcMesh;
//   std::vector<EntityHandle> &mapGaussPts;
//   boost::shared_ptr<CommonData> commonDataPtr;
//   int globalRule;
// };

// struct OpSaveGaussPts : public MFrontMoFEMInterface::DomainEleOp {
//   OpSaveGaussPts(const std::string field_name, moab::Interface &moab_mesh,
//                  boost::shared_ptr<CommonData> common_data_ptr);
//   MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

// private:
//   boost::shared_ptr<CommonData> commonDataPtr;
//   moab::Interface &internalVarMesh;
// };

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

template <int DIM>
using OpTangentFiniteStrains = struct OpTangent<Tensor4Pack<DIM>>;

template <int DIM>
using OpTangentSmallStrains = struct OpTangent<DdgPack<DIM>>;

// typedef struct OpTangent<Tensor4Pack> OpTangentFiniteStrains;
// typedef struct OpTangent<DdgPack> OpTangentSmallStrains;


using OpUpdateVariablesFiniteStrains = struct OpStressTmp<true, true>;


using OpUpdateVariablesSmallStrains = struct OpStressTmp<true, false>;


using OpStressFiniteStrains = struct OpStressTmp<false, true>;


using OpStressSmallStrains = struct OpStressTmp<false, false>;

// typedef struct OpStressTmp<true, true, 3> OpUpdateVariablesFiniteStrains;
// typedef struct OpStressTmp<true, false, 3> OpUpdateVariablesSmallStrains;
// typedef struct OpStressTmp<false, true, 3> OpStressFiniteStrains;
// typedef struct OpStressTmp<false, false, 3> OpStressSmallStrains;

} // namespace MFrontInterface