#include <arm_neon.h>
#include <torch/all.h>

#include <cmath>

namespace vec_op {

// FIXME: FP16 is not fully supported in Torch-CPU
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef CPU_OP_GUARD
#define CPU_KERNEL_GUARD_IN(NAME)
#define CPU_KERNEL_GUARD_OUT(NAME)
#else
#define CPU_KERNEL_GUARD_IN(NAME)                                              \
  std::cout << #NAME << " invoked." << std::endl;
#define CPU_KERNEL_GUARD_OUT(NAME) std::cout << #NAME << " exit." << std::endl;
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F &&f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
}; // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F &&f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T> struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

struct FP32Vec8;
struct FP32Vec16;

struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  float16x8_t reg;

  explicit FP16Vec8(__fp16 v) : reg(vmovq_n_f16(v)) {} //all 8 values set to v

  explicit FP16Vec8(const void *ptr) : reg(vld1q_f16((const __fp16 *)ptr)) {} 

  explicit FP16Vec8(float16x8_t data) : reg(data) {}

  explicit FP16Vec8(float32x4x2_t v) : reg(vcombine_f16(vcvt_f16_f32(v.val[0]), vcvt_f16_f32(v.val[1]))) {}

  FP16Vec8 operator*(const FP16Vec8 &b) const {
    return FP16Vec8(vmulq_f16(reg, b.reg));
  }

  FP16Vec8 operator+(const FP16Vec8 &b) const {
    return FP16Vec8(vaddq_f16(reg, b.reg));
  }

  FP16Vec8 operator-(const FP16Vec8 &b) const {
    return FP16Vec8(vsubq_f16(reg, b.reg));
  }

  FP16Vec8 operator/(const FP16Vec8 &b) const {
    return FP16Vec8(vdivq_f16(reg, b.reg));
  }

  void save(void *ptr) const { vst1q_f16((__fp16 *)ptr, reg); }
};

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  bfloat16x8_t reg;

  explicit BF16Vec8(const void *ptr)
      : reg(*reinterpret_cast<const bfloat16x8_t *>(ptr)) {}

  explicit BF16Vec8(bfloat16x8_t data) : reg(data) {}

  explicit BF16Vec8(const FP32Vec8 &);

  explicit BF16Vec8(float32x4x2_t v) : reg(vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[0]), v.val[1])) {}  

  void save(void *ptr) const { *reinterpret_cast<bfloat16x8_t *>(ptr) = reg; }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  bfloat16x8x2_t reg;

  explicit BF16Vec16(const void *ptr)
      : reg(*reinterpret_cast<const bfloat16x8x2_t *>(ptr)) {}

  explicit BF16Vec16(bfloat16x8x2_t data) : reg(data) {}

  explicit BF16Vec16(const FP32Vec16 &);

  explicit BF16Vec16(float32x4x4_t v) : reg({
    vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[0]), v.val[1]),
    vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[2]), v.val[3])
  }){};

  void save(void *ptr) const { *reinterpret_cast<bfloat16x8x2_t *>(ptr) = reg; }
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  bfloat16x8x4_t reg;

  explicit BF16Vec32(const void *ptr)
      : reg(*reinterpret_cast<const bfloat16x8x4_t *>(ptr)) {}

  explicit BF16Vec32(bfloat16x8x4_t data) : reg(data) {}

  explicit BF16Vec32(const BF16Vec8 &vec8_data) : reg({
    vec8_data.reg,
    vec8_data.reg,
    vec8_data.reg,
    vec8_data.reg
  }) {}

  void save(void *ptr) const { *reinterpret_cast<bfloat16x8x4_t *>(ptr) = reg; }
};

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;

  union AliasReg {
    float32x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4_t reg;

  // Constructor that initializes all elements with a single float value
  explicit FP32Vec4(float v) : reg(vdupq_n_f32(v)) {}

  // Default constructor initializes all elements to zero
  explicit FP32Vec4() : reg(vdupq_n_f32(0.0f)) {}

  // Constructor to load from memory
  explicit FP32Vec4(const float *ptr) : reg(vld1q_f32(ptr)) {}

  // Constructor to initialize directly from float32x4_t data
  explicit FP32Vec4(float32x4_t data) : reg(data) {}

  // Copy constructor
  explicit FP32Vec4(const FP32Vec4 &data) : reg(data.reg) {}
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    float32x4x2_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4x2_t reg;

  explicit FP32Vec8(float v) : reg({vmovq_n_f32(v), vmovq_n_f32(v)}) {}

  explicit FP32Vec8() : reg({vmovq_n_f32(0.0), vmovq_n_f32(0.0)}) {}

  explicit FP32Vec8(const float *ptr) : reg({vld1q_f32(ptr), vld1q_f32(ptr + 4)}) {}

  explicit FP32Vec8(float32x4x2_t data) : reg(data) {}

  explicit FP32Vec8(const FP32Vec8 &data) : reg(data.reg) {}

  explicit FP32Vec8(float16x8_t v) : reg({vcvt_f32_f16(vget_low_f16(v)), vcvt_f32_f16(vget_high_f16(v))}) {}

  explicit FP32Vec8(bfloat16x8_t v) : reg({vcvtq_low_f32_bf16(v), vcvtq_high_f32_bf16(v)}) {}

  explicit FP32Vec8(const BF16Vec8 &v) : reg({vcvtq_low_f32_bf16(v.reg), vcvtq_high_f32_bf16(v.reg)}) {}

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float ans = 0;
    unroll_loop<int, VEC_ELEM_NUM>([&ans, &ar](int i) { ans += ar.values[i]; });

    return ans;
  }

  FP32Vec8 exp() const {
    AliasReg ar;
    ar.reg = reg;

    // Extract float32x2_t elements for each float32x4_t
    float32x2_t exp_vec0 = {expf(ar.values[0]), expf(ar.values[1])};
    float32x2_t exp_vec1 = {expf(ar.values[2]), expf(ar.values[3])};
    float32x2_t exp_vec2 = {expf(ar.values[4]), expf(ar.values[5])};
    float32x2_t exp_vec3 = {expf(ar.values[6]), expf(ar.values[7])};

    // Combine the results into float32x4_t vectors
    float32x4_t result0 = vcombine_f32(exp_vec0, exp_vec1);
    float32x4_t result1 = vcombine_f32(exp_vec2, exp_vec3);

    // Initialize the float32x4x2_t object
    float32x4x2_t result;
    result.val[0] = result0;
    result.val[1] = result1;

    return FP32Vec8(result);
  }

  FP32Vec8 tanh() const {
    AliasReg ar;
    ar.reg = reg;

    // Extract float32x2_t elements for each float32x4_t
    float32x2_t tanh_vec0 = {tanhf(ar.values[0]), tanhf(ar.values[1])};
    float32x2_t tanh_vec1 = {tanhf(ar.values[2]), tanhf(ar.values[3])};
    float32x2_t tanh_vec2 = {tanhf(ar.values[4]), tanhf(ar.values[5])};
    float32x2_t tanh_vec3 = {tanhf(ar.values[6]), tanhf(ar.values[7])};

    // Combine the results into float32x4_t vectors
    float32x4_t result0 = vcombine_f32(tanh_vec0, tanh_vec1);
    float32x4_t result1 = vcombine_f32(tanh_vec2, tanh_vec3);

    // Initialize the float32x4x2_t object
    float32x4x2_t result;
    result.val[0] = result0;
    result.val[1] = result1;

    return FP32Vec8(result);
  }

  FP32Vec8 er() const {
    AliasReg ar;
    ar.reg = reg;

    // Extract float32x2_t elements for each float32x4_t
    float32x2_t er_vec0 = {erf(ar.values[0]), erf(ar.values[1])};
    float32x2_t er_vec1 = {erf(ar.values[2]), erf(ar.values[3])};
    float32x2_t er_vec2 = {erf(ar.values[4]), erf(ar.values[5])};
    float32x2_t er_vec3 = {erf(ar.values[6]), erf(ar.values[7])};

    // Combine the results into float32x4_t vectors
    float32x4_t result0 = vcombine_f32(er_vec0, er_vec1);
    float32x4_t result1 = vcombine_f32(er_vec2, er_vec3);

    // Initialize the float32x4x2_t object
    float32x4x2_t result;
    result.val[0] = result0;
    result.val[1] = result1;

    return FP32Vec8(result);
  }


  FP32Vec8 operator*(const FP32Vec8 &b) const {
    return FP32Vec8(float32x4x2_t({vmulq_f32(reg.val[0], b.reg.val[0]), vmulq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator+(const FP32Vec8 &b) const {
    return FP32Vec8(float32x4x2_t({vaddq_f32(reg.val[0], b.reg.val[0]), vaddq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator-(const FP32Vec8 &b) const {
    return FP32Vec8(float32x4x2_t({vsubq_f32(reg.val[0], b.reg.val[0]), vsubq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator/(const FP32Vec8 &b) const {
    return FP32Vec8(float32x4x2_t({vdivq_f32(reg.val[0], b.reg.val[0]), vdivq_f32(reg.val[1], b.reg.val[1])}));
  }

  void save(float *ptr) const {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr + 4, reg.val[1]);
  }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    float32x4x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4x4_t reg;

  explicit FP32Vec16(float v) : reg({vmovq_n_f32(v), vmovq_n_f32(v), vmovq_n_f32(v), vmovq_n_f32(v)}) {}

  explicit FP32Vec16() : reg({vmovq_n_f32(0.0), vmovq_n_f32(0.0), vmovq_n_f32(0.0), vmovq_n_f32(0.0)}) {}

  explicit FP32Vec16(const float *ptr) : reg({vld1q_f32(ptr), vld1q_f32(ptr + 4), vld1q_f32(ptr + 8), vld1q_f32(ptr + 12)}) {}

  explicit FP32Vec16(float32x4x4_t data) : reg(data) {}

  explicit FP32Vec16(const FP32Vec16 &data) : reg(data.reg) {}

  explicit FP32Vec16(bfloat16x8x2_t v) : reg({
    vcvtq_low_f32_bf16(v.val[0]),
    vcvtq_high_f32_bf16(v.val[0]),
    vcvtq_low_f32_bf16(v.val[1]),
    vcvtq_high_f32_bf16(v.val[1])
  }) {}

  explicit FP32Vec16(const FP32Vec4 &data) {
    reg.val[0] = data.reg;
    reg.val[1] = data.reg;
    reg.val[2] = data.reg;
    reg.val[3] = data.reg;
  }

  explicit FP32Vec16(const FP32Vec8 &data) {
    reg.val[0] = data.reg.val[0]; // First half (low 128 bits)
    reg.val[1] = data.reg.val[1]; // Second half (high 128 bits)
    reg.val[2] = data.reg.val[0]; // Repeating the first half
    reg.val[3] = data.reg.val[1]; // Repeating the second half
  }


  explicit FP32Vec16(const BF16Vec16 &v) : reg({
    vcvtq_low_f32_bf16(v.reg.val[0]),
    vcvtq_high_f32_bf16(v.reg.val[0]),
    vcvtq_low_f32_bf16(v.reg.val[1]),
    vcvtq_high_f32_bf16(v.reg.val[1])
  }) {}

  explicit FP32Vec16(const BF16Vec8 &v) : FP32Vec16(FP32Vec8(v)) {}
  
  FP32Vec16 operator+(const FP32Vec16 &b) const {
    return FP32Vec16(float32x4x4_t({
        vaddq_f32(reg.val[0], b.reg.val[0]),
        vaddq_f32(reg.val[1], b.reg.val[1]),
        vaddq_f32(reg.val[2], b.reg.val[2]),
        vaddq_f32(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator*(const FP32Vec16 &b) const {
    return FP32Vec16(float32x4x4_t({
        vmulq_f32(reg.val[0], b.reg.val[0]),
        vmulq_f32(reg.val[1], b.reg.val[1]),
        vmulq_f32(reg.val[2], b.reg.val[2]),
        vmulq_f32(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator-(const FP32Vec16 &b) const {
    return FP32Vec16(float32x4x4_t({
        vsubq_f32(reg.val[0], b.reg.val[0]),
        vsubq_f32(reg.val[1], b.reg.val[1]),
        vsubq_f32(reg.val[2], b.reg.val[2]),
        vsubq_f32(reg.val[3], b.reg.val[3])
    }));
  }

  FP32Vec16 operator/(const FP32Vec16 &b) const {
    return FP32Vec16(float32x4x4_t({
        vdivq_f32(reg.val[0], b.reg.val[0]),
        vdivq_f32(reg.val[1], b.reg.val[1]),
        vdivq_f32(reg.val[2], b.reg.val[2]),
        vdivq_f32(reg.val[3], b.reg.val[3])
    }));
  }

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float ans = 0;
    unroll_loop<int, VEC_ELEM_NUM>([&ans, &ar](int i) { ans += ar.values[i]; });

    return ans;
  }

  template <int group_size> float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);

    AliasReg ar;
    ar.reg = reg;
    float ans = 0;
    const int start = idx * group_size;
    unroll_loop<int, group_size>(
        [&ans, &start, ar](int i) { ans += ar.values[start + i]; });

    return ans;
  }

  void save(float *ptr) const {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr + 4, reg.val[1]);
    vst1q_f32(ptr + 8, reg.val[2]);
    vst1q_f32(ptr + 12, reg.val[3]);
  }
};

template <typename T> struct VecType { using vec_type = void; };

template <typename T> using vec_t = typename VecType<T>::vec_type;

template <> struct VecType<float> { using vec_type = FP32Vec8; };

template <> struct VecType<c10::Half> { using vec_type = FP16Vec8; };

template <> struct VecType<c10::BFloat16> { using vec_type = BF16Vec8; };

template <typename T> void storeFP32(float v, T *ptr) { *ptr = v; }

template <> inline void storeFP32<c10::Half>(float v, c10::Half *ptr) {
  *reinterpret_cast<__fp16 *>(ptr) = v;
}

inline void fma(FP32Vec16 &acc, FP32Vec16 &a, FP32Vec16 &b) {
  acc.reg.val[0] = vfmaq_f32(acc.reg.val[0], a.reg.val[0], b.reg.val[0]);
  acc.reg.val[1] = vfmaq_f32(acc.reg.val[1], a.reg.val[1], b.reg.val[1]);
  acc.reg.val[2] = vfmaq_f32(acc.reg.val[2], a.reg.val[2], b.reg.val[2]);
  acc.reg.val[3] = vfmaq_f32(acc.reg.val[3], a.reg.val[3], b.reg.val[3]);
}

inline void fma(FP32Vec16 &acc, BF16Vec32 &a, BF16Vec32 &b) {
  // Convert BF16 to FP32 for each half of the BF16 vectors
  float32x4_t a0_low = vcvt_f32_bf16(vget_low_bf16(a.reg.val[0]));
  float32x4_t a0_high = vcvt_f32_bf16(vget_high_bf16(a.reg.val[0]));
  float32x4_t a1_low = vcvt_f32_bf16(vget_low_bf16(a.reg.val[1]));
  float32x4_t a1_high = vcvt_f32_bf16(vget_high_bf16(a.reg.val[1]));

  float32x4_t b0_low = vcvt_f32_bf16(vget_low_bf16(b.reg.val[0]));
  float32x4_t b0_high = vcvt_f32_bf16(vget_high_bf16(b.reg.val[0]));
  float32x4_t b1_low = vcvt_f32_bf16(vget_low_bf16(b.reg.val[1]));
  float32x4_t b1_high = vcvt_f32_bf16(vget_high_bf16(b.reg.val[1]));

  // Perform FMA on FP32 vectors
  acc.reg.val[0] = vfmaq_f32(acc.reg.val[0], a0_low, b0_low);
  acc.reg.val[1] = vfmaq_f32(acc.reg.val[1], a0_high, b0_high);
  acc.reg.val[2] = vfmaq_f32(acc.reg.val[2], a1_low, b1_low);
  acc.reg.val[3] = vfmaq_f32(acc.reg.val[3], a1_high, b1_high);
}

inline BF16Vec8::BF16Vec8(const FP32Vec8 &v) : reg(vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.reg.val[0]), v.reg.val[1])) {};

inline BF16Vec16::BF16Vec16(const FP32Vec16 &v) : reg({
    vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.reg.val[0]), v.reg.val[1]),
    vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.reg.val[2]), v.reg.val[3])
  }){};

inline void prefetch(const void *addr) {
    __builtin_prefetch(addr, 0, 1);
}

template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16 *ptr) { // storeFP32ToT
  *reinterpret_cast<__bf16 *>(ptr) = vcvth_bf16_f32(v);
}

};
