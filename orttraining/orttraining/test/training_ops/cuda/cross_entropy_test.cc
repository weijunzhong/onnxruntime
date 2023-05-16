// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "test/compare_ortvalue.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_utils.h"
#include "test/providers/compare_provider_test_utils.h"

using namespace std;

namespace onnxruntime {
namespace test {

#if USE_CUDA
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#elif USE_ROCM
constexpr const char* kGpuExecutionProvider = kRocmExecutionProvider;
#endif

static void TestSoftmaxCrossEntropy(const std::vector<int64_t>& X_dims,
                                    const std::vector<int64_t>& label_dims,
                                    const std::vector<int64_t>& Y_dims,
                                    const std::vector<int64_t>& log_prob_dims,
                                    const std::string& reduction) {
  CompareOpTester test("SoftmaxCrossEntropy", 1, kMSDomain);
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{2333};
  std::vector<float> X_data = random.Uniform<float>(X_dims, -200.0f, 200.0f);
  std::vector<float> label_data = random.OneHot<float>(label_dims, label_dims.back());

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("label", label_dims, label_data);

  std::vector<float> Y_data = FillZeros<float>(Y_dims);
  std::vector<float> log_prob_data = FillZeros<float>(log_prob_dims);

  test.AddOutput<float>("output", Y_dims, Y_data);
  test.AddOutput<float>("log_prob", log_prob_dims, log_prob_data);

  test.CompareWithCPU(kGpuExecutionProvider);
}

static void TestSoftmaxCrossEntropyGrad(const std::vector<int64_t>& dY_dims,
                                        const std::vector<int64_t>& log_prob_dims,
                                        const std::vector<int64_t>& label_dims,
                                        const std::vector<int64_t>& dX_dims,
                                        const std::string& reduction) {
  CompareOpTester test("SoftmaxCrossEntropyGrad", 1, kMSDomain);
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{2333};
  std::vector<float> dY_data = random.Uniform<float>(dY_dims, -10.0f, 10.0f);
  std::vector<float> log_prob_data = random.Uniform<float>(log_prob_dims, -10.0f, 10.0f);
  std::vector<float> label_data = random.Uniform<float>(label_dims, 0.0f, 1.0f);

  test.AddInput<float>("dY", dY_dims, dY_data);
  test.AddInput<float>("log_prob", log_prob_dims, log_prob_data);
  test.AddInput<float>("label", label_dims, label_data);

  std::vector<float> dX_data = FillZeros<float>(dX_dims);

  test.AddOutput<float>("dX", dX_dims, dX_data);

  test.CompareWithCPU(kGpuExecutionProvider);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropy_TinySizeTensor) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> label_dims{8, 2};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "mean");
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "sum");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropy_SmallSizeTensor) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> label_dims{8, 20, 10};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "mean");
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "sum");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropy_MediumSizeTensor) {
  std::vector<int64_t> X_dims{7, 1024};
  std::vector<int64_t> label_dims{7, 1024};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{7, 1024};
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "mean");
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "sum");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropy_LargeSizeTensor) {
  std::vector<int64_t> X_dims{2, 512, 30528};
  std::vector<int64_t> label_dims{2, 512, 30528};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "mean");
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "sum");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyGrad_TinySizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> label_dims{8, 2};
  std::vector<int64_t> dX_dims{8, 2};
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "sum");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyGrad_SmallSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  std::vector<int64_t> label_dims{8, 20, 10};
  std::vector<int64_t> dX_dims{8, 20, 10};
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "sum");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyGrad_LargeSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  std::vector<int64_t> label_dims{2, 512, 30528};
  std::vector<int64_t> dX_dims{2, 512, 30528};
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "sum");
}

TEST(CrossEntropyTest, SparseSoftmaxCrossEntropy_Basic) {
  OpTester test("SparseSoftmaxCrossEntropy", 9);
  test.AddAttribute("reduction", "mean");

  std::vector<float> X_data{-0.9468f, 1.3250f, 1.0438f, 0.4106f, -0.2150f,
                            -0.3399f, -0.4396f, 1.1835f, 1.2089f, -1.0617f,
                            -0.5239f, -0.2767f, 0.9910f, -1.5688f, -0.2863f};
  std::vector<int64_t> index_data = {3, 4, 1};
  std::vector<float> Y_data = {2.2956f};
  std::vector<float> log_prob_data = {-3.1773f, -0.9054f, -1.1867f, -1.8199f, -2.4454f,
                                      -2.4583f, -2.5580f, -0.9349f, -0.9094f, -3.1800f,
                                      -2.1341f, -1.8869f, -0.6192f, -3.1789f, -1.8965f};

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOutput<float>("output", {}, Y_data);
  test.AddOutput<float>("log_prob", {3, 5}, log_prob_data);

  test.Run();
}

static void TestSparseSoftmaxCrossEntropy(const std::vector<int64_t>* X_dims,
                                          const std::vector<int64_t>* index_dims,
                                          const std::vector<int64_t>* weight_dims,
                                          const std::vector<int64_t>* Y_dims,
                                          const std::vector<int64_t>* log_prob_dims,
                                          const std::string& reduction) {
  CompareOpTester test("SparseSoftmaxCrossEntropy");
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{2333};
  std::vector<float> X_data = random.Uniform<float>(*X_dims, -200.0f, 200.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(*index_dims, 0, X_dims->back());

  test.AddInput<float>("X", *X_dims, X_data);
  test.AddInput<int64_t>("index", *index_dims, index_data);

  if (weight_dims) {
    std::vector<float> weight_data = random.Uniform<float>(*weight_dims, 0.0f, 1.0f);
    test.AddInput<float>("weight", *weight_dims, weight_data);
  }

  std::vector<float> Y_data = FillZeros<float>(*Y_dims);
  std::vector<float> log_prob_data = FillZeros<float>(*log_prob_dims);

  test.AddOutput<float>("output", *Y_dims, Y_data);
  test.AddOutput<float>("log_prob", *log_prob_dims, log_prob_data);

  test.CompareWithCPU(kGpuExecutionProvider);
}

TEST(CrossEntropyTest, SparseSoftmaxCrossEntropy_TinySizeTensor) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{8};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
}

TEST(CrossEntropyTest, SparseSoftmaxCrossEntropy_SmallSizeTensor) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 20};
  std::vector<int64_t> weight_dims{8, 20};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
}

TEST(CrossEntropyTest, SparseSoftmaxCrossEntropy_MediumSizeTensor) {
  std::vector<int64_t> X_dims{8, 1024};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{8};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 1024};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
}

TEST(CrossEntropyTest, SparseSoftmaxCrossEntropy_LargeSizeTensor) {
  std::vector<int64_t> X_dims{4, 512, 30528};
  std::vector<int64_t> index_dims{4, 512};
  std::vector<int64_t> weight_dims{4, 512};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{4, 512, 30528};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
}

static void TestSparseSoftmaxCrossEntropyGrad(const std::vector<int64_t>& dY_dims,
                                              const std::vector<int64_t>& log_prob_dims,
                                              const std::vector<int64_t>& index_dims,
                                              const std::vector<int64_t>& dX_dims,
                                              const std::string& reduction) {
  CompareOpTester test("SparseSoftmaxCrossEntropyGrad");
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{2333};
  std::vector<float> dY_data = random.Uniform<float>(dY_dims, -10.0f, 10.0f);
  std::vector<float> log_prob_data = random.Uniform<float>(log_prob_dims, -10.0f, 10.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(index_dims, 0, dX_dims.back());

  test.AddInput<float>("dY", dY_dims, dY_data);
  test.AddInput<float>("log_prob", log_prob_dims, log_prob_data);
  test.AddInput<int64_t>("index", index_dims, index_data);

  std::vector<float> dX_data = FillZeros<float>(dX_dims);

  test.AddOutput<float>("dX", dX_dims, dX_data);

  test.CompareWithCPU(kGpuExecutionProvider);
}

TEST(CrossEntropyTest, SparseSoftmaxCrossEntropyGrad_TinySizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> dX_dims{8, 2};
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
}

TEST(CrossEntropyTest, SparseSoftmaxCrossEntropyGrad_SmallSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 20};
  std::vector<int64_t> dX_dims{8, 20, 10};
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
}

TEST(CrossEntropyTest, SparseSoftmaxCrossEntropyGrad_LargeSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  std::vector<int64_t> index_dims{2, 512};
  std::vector<int64_t> dX_dims{2, 512, 30528};
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
}

static void PrepareSCELossTestData(const std::vector<int64_t>* X_dims,
                                   const std::vector<int64_t>* index_dims, const std::vector<int64_t>* weight_dims,
                                   const std::int64_t ignore_index,
                                   std::vector<float>& X_data, std::vector<int64_t>& index_data,
                                   std::vector<float>& weight_data) {
  // create rand inputs
  RandomValueGenerator random{2333};
  X_data = random.Uniform<float>(*X_dims, -200.0f, 200.0f);
  index_data = random.Uniform<int64_t>(*index_dims, 0, (*X_dims)[1]);
  // Add one data point that has ignore_index.
  if (index_data.size() > 0) {
    index_data[0] = ignore_index;
  }

  if (weight_dims) {
    weight_data = random.Uniform<float>(*weight_dims, 0.0f, 1.0f);
  }
}

template <typename T, typename TOut>
static std::vector<OrtValue> RunSCELossWithEP(const char* op,
                                              int opset_version,
                                              const char* domain,
                                              std::function<std::unique_ptr<IExecutionProvider>()>
                                                  ep_creator,
                                              const std::string& reduction,
                                              const std::int64_t ignore_index,
                                              const double error_tolerance,
                                              const std::vector<int64_t>* X_dims,
                                              const std::vector<int64_t>* index_dims,
                                              const std::vector<int64_t>* weight_dims,
                                              const std::vector<int64_t>* Y_dims,
                                              const std::vector<int64_t>* log_prob_dims,
                                              std::vector<float>& X_data,
                                              std::vector<int64_t>& index_data,
                                              std::vector<float>& weight_data) {
  /**
   * OpTester's atol/rtol check is too strict for our testing cases. Imagine expected value is 4.7683704451628728e-07,
   * real value is 0, even we set rtol=1e-1, atol = 1e-4. The check still fail.
   * The difference between expected[i] and output[i] is 4.7683704451628728e-07,
   * which exceeds *(params.relative_error_) * std::abs(expected[i])
   * (params.relative_error_) * std::abs(expected[i]) evaluates to 4.7683705872714199e-08.
   *
   * CompareOrtValue called by test.CompareWithCPU looses the check by compare diff with atol + rtol * expected[i].
   * So here we disable OpTester's check by default, and do the check externally.
   */
  bool need_verify_outputs = false;
  std::vector<std::vector<float>> expected_values;
  // Still need feed the output data even we don't want to verify it.
  expected_values.push_back(FillZeros<float>(*Y_dims));
  if (log_prob_dims) {
    expected_values.push_back(FillZeros<float>(*log_prob_dims));
  }

  OpTester test(op, opset_version, domain, need_verify_outputs /*verify_output*/);
  const bool is_internal_op = (std::string(op).compare("SoftmaxCrossEntropyLossInternal") == 0);

  if (is_internal_op && !std::is_same<T, TOut>::value) {
    test.AddAttribute("output_type", static_cast<int64_t>(utils::ToTensorProtoElementType<TOut>()));
  }

  test.AddAttribute("reduction", reduction);
  if (!is_internal_op) {
    test.AddAttribute("ignore_index", ignore_index);
  }

  if (std::is_same<T, MLFloat16>::value) {
    std::vector<MLFloat16> X_data_half(X_data.size());
    ConvertFloatToMLFloat16(X_data.data(), X_data_half.data(), static_cast<int>(X_data.size()));
    test.AddInput<MLFloat16>("X", *X_dims, X_data_half);
  } else {
    test.AddInput<float>("X", *X_dims, X_data);
  }

  test.AddInput<int64_t>("index", *index_dims, index_data);

  if (weight_dims) {
    if (std::is_same<T, MLFloat16>::value) {
      std::vector<MLFloat16> weight_data_half(weight_data.size());
      ConvertFloatToMLFloat16(weight_data.data(), weight_data_half.data(), static_cast<int>(weight_data.size()));
      test.AddInput<MLFloat16>("weight", *weight_dims, weight_data_half);
    } else {
      test.AddInput<float>("weight", *weight_dims, weight_data);
    }
  }

  if (is_internal_op && ignore_index != -1) {
    test.AddInput<int64_t>("ignore_index", {}, &ignore_index, 1);
  }

  if (std::is_same<TOut, MLFloat16>::value) {
    std::vector<MLFloat16> output_half(expected_values[0].size());
    ConvertFloatToMLFloat16(expected_values[0].data(), output_half.data(), static_cast<int>(expected_values[0].size()));
    test.AddOutput<MLFloat16>("output", *Y_dims, output_half);

    if (log_prob_dims) {
      std::vector<MLFloat16> log_prob_half(expected_values[1].size());
      ConvertFloatToMLFloat16(expected_values[1].data(), log_prob_half.data(), static_cast<int>(expected_values[1].size()));
      test.AddOutput<MLFloat16>("log_prob", *log_prob_dims, log_prob_half);
    }

  } else {
    test.AddOutput<float>("output", *Y_dims, expected_values[0]);
    if (log_prob_dims) {
      test.AddOutput<float>("log_prob", *log_prob_dims, expected_values[1]);
    }
  }

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.emplace_back(ep_creator());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  return test.GetFetches();
}

template <typename T, typename TOut>
static void TestSCELoss(const char* op, int opset_version,
                        const char* domain, const std::vector<int64_t>* X_dims,
                        const std::vector<int64_t>* index_dims, const std::vector<int64_t>* weight_dims,
                        const std::vector<int64_t>* Y_dims, const std::vector<int64_t>* log_prob_dims,
                        const std::string& reduction, const std::int64_t ignore_index,
                        const double error_tolerance) {
  ASSERT_TRUE((std::is_same<T, MLFloat16>::value || std::is_same<T, float>::value));
  ASSERT_TRUE((std::is_same<TOut, MLFloat16>::value || std::is_same<TOut, float>::value));

  std::vector<float> X_data, weight_data;
  std::vector<int64_t> index_data;
  PrepareSCELossTestData(X_dims, index_dims, weight_dims, ignore_index, X_data, index_data, weight_data);

  // Run on CPU using float input and output
  // (because the CPU implementation doesn't support variant input output types.)
  // Be noted, no result comparison is done here.
  std::vector<OrtValue> cpu_fetches = RunSCELossWithEP<float, float>(
      op, opset_version, domain,
      []() -> std::unique_ptr<IExecutionProvider> { return DefaultCpuExecutionProvider(); },
      reduction, ignore_index, error_tolerance,
      X_dims, index_dims, weight_dims,
      Y_dims, log_prob_dims,
      X_data, index_data, weight_data);

  // Run on CUDA.
  // Be noted, no result comparison is done here because OpTester's check is too strick for our test cases.
  // Check more details in comment of RunSCELossWithEP.
  std::vector<OrtValue> target_fetches = RunSCELossWithEP<T, TOut>(
      op, opset_version, domain,
      []() -> std::unique_ptr<IExecutionProvider> {
#ifdef USE_CUDA
        return DefaultCudaExecutionProvider();
#elif USE_ROCM
        return DefaultRocmExecutionProvider();
#endif
      },
      reduction, ignore_index, error_tolerance,
      X_dims, index_dims, weight_dims,
      Y_dims, log_prob_dims,
      X_data, index_data, weight_data);

  // Compare
  ASSERT_EQ(cpu_fetches.size(), target_fetches.size());
  for (size_t i = 0; i < cpu_fetches.size(); i++) {
    if (std::is_same<TOut, MLFloat16>::value) {
      auto y_data_size = cpu_fetches[i].Get<Tensor>().Shape().Size();
      std::vector<float> cpu_temp_buffer;
      cpu_temp_buffer.resize(y_data_size);
      const float* y_buffer = cpu_fetches[i].Get<Tensor>().Data<float>();
      std::copy(y_buffer, y_buffer + y_data_size, cpu_temp_buffer.begin());

      std::vector<MLFloat16> ret_half(cpu_temp_buffer.size());
      ConvertFloatToMLFloat16(cpu_temp_buffer.data(), ret_half.data(), static_cast<int>(cpu_temp_buffer.size()));

      OrtValue target;
      test::CreateInputOrtValueOnCPU<MLFloat16>(cpu_fetches[i].Get<Tensor>().Shape().GetDims(), ret_half, &target);
      auto ret = CompareOrtValue(target_fetches[i], target, error_tolerance /*per_sample_tolerance*/,
                                 error_tolerance /*relative_per_sample_tolerance*/, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;

    } else {
      auto ret = CompareOrtValue(target_fetches[i], cpu_fetches[i], error_tolerance /*per_sample_tolerance*/,
                                 error_tolerance /*relative_per_sample_tolerance*/, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

template <typename T, typename TOut>
static void TestSoftmaxCrossEntropyLoss(const std::vector<int64_t>* X_dims, const std::vector<int64_t>* index_dims,
                                        const std::vector<int64_t>* weight_dims, const std::vector<int64_t>* Y_dims,
                                        const std::vector<int64_t>* log_prob_dims, const std::string& reduction,
                                        const std::int64_t ignore_index = -1,
                                        const double error_tolerance = 1e-4) {
  // Only test SoftmaxCrossEntropyLoss if input and output type matches.
  if (std::is_same<T, TOut>::value) {
    TestSCELoss<T, TOut>("SoftmaxCrossEntropyLoss", 12, onnxruntime::kOnnxDomain,
                         X_dims, index_dims, weight_dims, Y_dims, log_prob_dims,
                         reduction, ignore_index, error_tolerance);
  }

  // Can we add a empty optional input before a non-empty input?
  if (weight_dims || ignore_index == -1) {
    TestSCELoss<T, TOut>("SoftmaxCrossEntropyLossInternal", 1, onnxruntime::kMSDomain,
                         X_dims, index_dims, weight_dims, Y_dims, log_prob_dims, reduction,
                         ignore_index, error_tolerance);
  }
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_TinySizeTensor) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{2};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  std::vector<int64_t> log_prob_dims{8, 2};
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none");

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean", 0);
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean", 0);
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum", 0);
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum", 0);
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none", 0);
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none", 0);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_TinySizeTensor_half) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{2};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  std::vector<int64_t> log_prob_dims{8, 2};
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                    "mean", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                    "mean", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                    "sum", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                    "sum", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims,
                                                    "none", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims,
                                                    "none", -1, 5e-2);

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                    "mean", 0, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                    "mean", 0, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                    "sum", 0, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                    "sum", 0, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims,
                                                    "none", 0, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims,
                                                    "none", 0, 5e-2);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_SmallSizeTensor) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 10};
  std::vector<int64_t> weight_dims{20};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8, 10};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_SmallSizeTensor_half) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 10};
  std::vector<int64_t> weight_dims{20};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8, 10};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                    "mean", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                    "mean", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                    "sum", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                    "sum", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims,
                                                    "none", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims,
                                                    "none", -1, 5e-2);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_MediumSizeTensor) {
  std::vector<int64_t> X_dims{8, 1024};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{1024};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  std::vector<int64_t> log_prob_dims{8, 1024};
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_MediumSizeTensor_half) {
  std::vector<int64_t> X_dims{8, 1024};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{1024};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  std::vector<int64_t> log_prob_dims{8, 1024};
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                    "mean", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                    "mean", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                    "sum", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                    "sum", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims,
                                                    "none", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, MLFloat16>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims,
                                                    "none", -1, 5e-2);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_MediumSizeTensor_half_input_float_output) {
  std::vector<int64_t> X_dims{8, 1024};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{1024};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  std::vector<int64_t> log_prob_dims{8, 1024};
  TestSoftmaxCrossEntropyLoss<MLFloat16, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                "mean", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                "mean", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims,
                                                "sum", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims,
                                                "sum", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, float>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims,
                                                "none", -1, 5e-2);
  TestSoftmaxCrossEntropyLoss<MLFloat16, float>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims,
                                                "none", -1, 5e-2);
}

// TODO fix flaky test
// failing random seed: 2873512643
TEST(CrossEntropyTest, DISABLED_SoftmaxCrossEntropyLoss_LargeSizeTensor) {
  std::vector<int64_t> X_dims{4, 512, 30528};
  std::vector<int64_t> index_dims{4, 30528};
  std::vector<int64_t> weight_dims{512};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{4, 30528};
  std::vector<int64_t> log_prob_dims{4, 512, 30528};
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none");
  TestSoftmaxCrossEntropyLoss<float, float>(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none");
}

static void TestSoftmaxCrossEntropyLossGrad(const std::vector<int64_t>& dY_dims,
                                            const std::vector<int64_t>& log_prob_dims,
                                            const std::vector<int64_t>& index_dims,
                                            const std::vector<int64_t>& dX_dims,
                                            const std::string& reduction,
                                            const std::int64_t ignore_index = -1,
                                            const bool test_fp16 = false,
                                            const double error_tolerance = 1e-4) {
  CompareOpTester test("SoftmaxCrossEntropyLossGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", reduction);
  test.AddAttribute("ignore_index", ignore_index);

  // create rand inputs
  RandomValueGenerator random{2333};
  std::vector<float> dY_data = random.Uniform<float>(dY_dims, -10.0f, 10.0f);
  std::vector<float> log_prob_data = random.Uniform<float>(log_prob_dims, -10.0f, 10.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(index_dims, 0, dX_dims[1]);
  // Add one data point that has ignore_index.
  if (index_data.size() > 0) {
    index_data[0] = ignore_index;
  }
  if (test_fp16) {
    std::vector<MLFloat16> dY_data_half(dY_data.size());
    ConvertFloatToMLFloat16(dY_data.data(), dY_data_half.data(), static_cast<int>(dY_data.size()));
    test.AddInput<MLFloat16>("dY", dY_dims, dY_data_half);

    std::vector<MLFloat16> log_prob_data_half(log_prob_data.size());
    ConvertFloatToMLFloat16(log_prob_data.data(), log_prob_data_half.data(), static_cast<int>(log_prob_data.size()));
    test.AddInput<MLFloat16>("log_prob", log_prob_dims, log_prob_data_half);

    test.AddInput<int64_t>("index", index_dims, index_data);

    std::vector<MLFloat16> dX_data = FillZeros<MLFloat16>(dX_dims);

    test.AddOutput<MLFloat16>("dX", dX_dims, dX_data);
    test.CompareWithCPU(kGpuExecutionProvider, error_tolerance, error_tolerance);
  } else {
    test.AddInput<float>("dY", dY_dims, dY_data);
    test.AddInput<float>("log_prob", log_prob_dims, log_prob_data);
    test.AddInput<int64_t>("index", index_dims, index_data);

    std::vector<float> dX_data = FillZeros<float>(dX_dims);

    test.AddOutput<float>("dX", dX_dims, dX_data);
    test.CompareWithCPU(kGpuExecutionProvider);
  }
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_TinySizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> dX_dims{8, 2};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
  TestSoftmaxCrossEntropyLossGrad({8}, log_prob_dims, index_dims, dX_dims, "none");

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", 0);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", 0);
  TestSoftmaxCrossEntropyLossGrad({8}, log_prob_dims, index_dims, dX_dims, "none", 0);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_SmallSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 10};
  std::vector<int64_t> dX_dims{8, 20, 10};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
  TestSoftmaxCrossEntropyLossGrad({8, 10}, log_prob_dims, index_dims, dX_dims, "none");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_LargeSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  std::vector<int64_t> index_dims{2, 30528};
  std::vector<int64_t> dX_dims{2, 512, 30528};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
  TestSoftmaxCrossEntropyLossGrad({2, 30528}, log_prob_dims, index_dims, dX_dims, "none");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_TinySizeTensor_half) {  //  [E:onnxruntime:Default, compare_provider_test_utils.cc:105 CompareWithCPU] Initialize failed with status: Could not find an implementation for Equal(19) node with name ''
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> dX_dims{8, 2};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad({8}, log_prob_dims, index_dims, dX_dims, "none", -1, true, 5e-2);

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", 0, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", 0, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad({8}, log_prob_dims, index_dims, dX_dims, "none", 0, true, 5e-2);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_SmallSizeTensor_half) {  //  [E:onnxruntime:Default, compare_provider_test_utils.cc:105 CompareWithCPU] Initialize failed with status: Could not find an implementation for Equal(19) node with name ''
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 10};
  std::vector<int64_t> dX_dims{8, 20, 10};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad({8, 10}, log_prob_dims, index_dims, dX_dims, "none", -1, true, 5e-2);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_LargeSizeTensor_half) {  //  [E:onnxruntime:Default, compare_provider_test_utils.cc:105 CompareWithCPU] Initialize failed with status: Could not find an implementation for Equal(19) node with name ''
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  std::vector<int64_t> index_dims{2, 30528};
  std::vector<int64_t> dX_dims{2, 512, 30528};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad({2, 30528}, log_prob_dims, index_dims, dX_dims, "none", -1, true, 5e-2);
}

static void PrepareSCELossInternalGradTestData(
    const std::vector<int64_t>& dY_dims, const std::vector<int64_t>& log_prob_dims,
    const std::vector<int64_t>& index_dims, const std::vector<int64_t>& weight_dims,
    const std::vector<int64_t>& dX_dims, const std::int64_t ignore_index, bool has_bias,
    std::vector<float>& dY_data, std::vector<float>& log_prob_data,
    std::vector<int64_t>& index_data, std::vector<float>& weight_data,
    std::vector<float>& bias_data) {
  // Create rand inputs
  RandomValueGenerator random{2333};
  dY_data = random.Uniform<float>(dY_dims, -10.0f, 10.0f);
  log_prob_data = random.Uniform<float>(log_prob_dims, -10.0f, 10.0f);
  index_data = random.Uniform<int64_t>(index_dims, 0, dX_dims[1]);
  // Add one data point that has ignore_index.
  if ((ignore_index != -1) && (index_data.size() > 0)) {
    index_data[0] = ignore_index;
  }
  weight_data = random.Uniform<float>(weight_dims, 0.0f, 1.0f);

  if (has_bias) {
    bias_data = random.Uniform<float>(dX_dims, 0.0f, 1.0f);
  }
}

template <typename T, typename TOut>
static std::vector<OrtValue> RunSCELossInternalGradWithEP(
    std::function<std::unique_ptr<IExecutionProvider>()>
        ep_creator,
    const std::string& reduction,
    const std::int64_t ignore_index,
    const double error_tolerance,
    const bool has_bias,
    const std::vector<int64_t>& dY_dims,
    const std::vector<int64_t>& log_prob_dims,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& weight_dims,
    const std::vector<int64_t>& dX_dims,
    std::vector<float>& dY_data,
    std::vector<float>& log_prob_data,
    std::vector<int64_t>& index_data,
    std::vector<float>& weight_data,
    std::vector<float>& bias_data) {
  /**
   * OpTester's atol/rtol check is too strict for our testing cases. Imagine expected value is 4.7683704451628728e-07,
   * real value is 0, even we set rtol=1e-1, atol = 1e-4. The check still fail with the following check:
   * The difference between expected[i] and output[i] is 4.7683704451628728e-07,
   * which exceeds *(params.relative_error_) * std::abs(expected[i])
   * (params.relative_error_) * std::abs(expected[i]) evaluates to 4.7683705872714199e-08.
   *
   * CompareOrtValue called by test.CompareWithCPU looses the check by compare diff with atol + rtol * expected[i].
   * So here we disable OpTester's check by default, and do the check externally.
   */
  bool need_verify_outputs = false;
  std::vector<float> expected_value = FillZeros<float>(dX_dims);

  ORT_ENFORCE((std::is_same<T, MLFloat16>::value || std::is_same<T, float>::value));
  ORT_ENFORCE((std::is_same<TOut, MLFloat16>::value || std::is_same<TOut, float>::value));

  std::unique_ptr<OpTester> test_unique_ptr = std::make_unique<OpTester>(
      "SoftmaxCrossEntropyLossInternalGrad", 1, onnxruntime::kMSDomain, need_verify_outputs /*verify_output*/);

  OpTester* test = test_unique_ptr.get();

  test->AddAttribute("reduction", reduction);
  if (!std::is_same<T, TOut>::value) {
    test->AddAttribute("output_type", static_cast<int64_t>(utils::ToTensorProtoElementType<TOut>()));
  }

  if (std::is_same<T, MLFloat16>::value) {
    std::vector<MLFloat16> dY_data_half(dY_data.size());
    ConvertFloatToMLFloat16(dY_data.data(), dY_data_half.data(), static_cast<int>(dY_data.size()));
    test->AddInput<MLFloat16>("dY", dY_dims, dY_data_half);

    std::vector<MLFloat16> log_prob_data_half(log_prob_data.size());
    ConvertFloatToMLFloat16(log_prob_data.data(), log_prob_data_half.data(), static_cast<int>(log_prob_data.size()));
    test->AddInput<MLFloat16>("log_prob", log_prob_dims, log_prob_data_half);

    test->AddInput<int64_t>("index", index_dims, index_data);

    std::vector<MLFloat16> weight_data_half(weight_data.size());
    ConvertFloatToMLFloat16(weight_data.data(), weight_data_half.data(), static_cast<int>(weight_data.size()));
    test->AddInput<MLFloat16>("weight", weight_dims, weight_data_half);

    if (ignore_index != -1 || has_bias) {
      test->AddInput<int64_t>("ignore_index", {}, &ignore_index, 1);
    }
  } else {
    test->AddInput<float>("dY", dY_dims, dY_data);
    test->AddInput<float>("log_prob", log_prob_dims, log_prob_data);
    test->AddInput<int64_t>("index", index_dims, index_data);
    test->AddInput<float>("weight", weight_dims, weight_data);
    if (ignore_index != -1 || has_bias) {
      test->AddInput<int64_t>("ignore_index", {}, &ignore_index, 1);
    }
  }

  if (std::is_same<TOut, MLFloat16>::value) {
    // Be noted, bias should be aligned with output's data type.
    if (has_bias) {
      std::vector<MLFloat16> bias_data_half(bias_data.size());
      ConvertFloatToMLFloat16(bias_data.data(), bias_data_half.data(), static_cast<int>(bias_data.size()));
      test->AddInput<MLFloat16>("bias", dX_dims, bias_data_half);
    }

    std::vector<MLFloat16> expected_data_half(expected_value.size());
    ConvertFloatToMLFloat16(expected_value.data(), expected_data_half.data(), static_cast<int>(expected_value.size()));
    test->AddOutput<MLFloat16>("dX", dX_dims, expected_data_half, false /*sort_output*/,
                               error_tolerance /*rel_error*/, error_tolerance /*abs_error*/);

  } else {
    if (has_bias) {
      test->AddInput<float>("bias", dX_dims, bias_data);
    }

    test->AddOutput<float>("dX", dX_dims, expected_value, false /*sort_output*/,
                           error_tolerance /*rel_error*/, error_tolerance /*abs_error*/);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.emplace_back(ep_creator());
  test->Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  return test->GetFetches();
}

template <typename T, typename TOut>
static void TestSoftmaxCrossEntropyLossInternalGrad(const std::vector<int64_t>& dY_dims,
                                                    const std::vector<int64_t>& log_prob_dims,
                                                    const std::vector<int64_t>& index_dims,
                                                    const std::vector<int64_t>& weight_dims,
                                                    const std::vector<int64_t>& dX_dims,
                                                    const std::string& reduction,
                                                    const std::int64_t ignore_index = -1,
                                                    const double error_tolerance = 1e-4,
                                                    const bool has_bias = false) {
  std::vector<float> dY_data, log_prob_data, weight_data, bias_data;
  std::vector<int64_t> index_data;
  PrepareSCELossInternalGradTestData(dY_dims, log_prob_dims, index_dims, weight_dims, dX_dims, ignore_index, has_bias,
                                     dY_data, log_prob_data, index_data, weight_data, bias_data);

  // Run on CPU using float input and output
  // (because the CPU implementation doesn't support variant input output types.)
  // Be noted, no result comparison is done here.
  std::vector<OrtValue> cpu_fetches =
      RunSCELossInternalGradWithEP<float, float>(
          []() -> std::unique_ptr<IExecutionProvider> { return DefaultCpuExecutionProvider(); },
          reduction, ignore_index, error_tolerance, has_bias,
          dY_dims, log_prob_dims, index_dims, weight_dims, dX_dims,
          dY_data, log_prob_data, index_data, weight_data, bias_data);

  // Run on CUDA and compare results with cpu results.
  std::vector<OrtValue> target_fetches =
      RunSCELossInternalGradWithEP<T, TOut>(
          []() -> std::unique_ptr<IExecutionProvider> {
#ifdef USE_CUDA
            return DefaultCudaExecutionProvider();
#elif USE_ROCM
            return DefaultRocmExecutionProvider();
#endif
          },
          reduction, ignore_index, error_tolerance, has_bias,
          dY_dims, log_prob_dims, index_dims, weight_dims, dX_dims,
          dY_data, log_prob_data, index_data, weight_data, bias_data);

  // Compare
  ASSERT_EQ(cpu_fetches.size(), target_fetches.size());
  for (size_t i = 0; i < cpu_fetches.size(); i++) {
    if (std::is_same<TOut, MLFloat16>::value) {
      auto y_data_size = cpu_fetches[i].Get<Tensor>().Shape().Size();
      std::vector<float> cpu_temp_buffer;
      cpu_temp_buffer.resize(y_data_size);
      const float* y_buffer = cpu_fetches[i].Get<Tensor>().Data<float>();
      std::copy(y_buffer, y_buffer + y_data_size, cpu_temp_buffer.begin());

      std::vector<MLFloat16> ret_half(cpu_temp_buffer.size());
      ConvertFloatToMLFloat16(cpu_temp_buffer.data(), ret_half.data(), static_cast<int>(cpu_temp_buffer.size()));

      OrtValue target;
      test::CreateInputOrtValueOnCPU<MLFloat16>(cpu_fetches[i].Get<Tensor>().Shape().GetDims(), ret_half, &target);
      auto ret = CompareOrtValue(target_fetches[i], target, error_tolerance /*per_sample_tolerance*/,
                                 error_tolerance /*relative_per_sample_tolerance*/, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;

    } else {
      auto ret = CompareOrtValue(target_fetches[i], cpu_fetches[i], error_tolerance /*per_sample_tolerance*/,
                                 error_tolerance /*relative_per_sample_tolerance*/, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossInternalGrad_TinySizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{2};
  std::vector<int64_t> dX_dims{8, 2};
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                        dX_dims, "mean");
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                        dX_dims, "sum");
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "none");

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>(dY_dims, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "mean", 0);
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>(dY_dims, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "sum", 0);
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "none", 0);

  // Bias.
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>(dY_dims, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "mean", -1, 1e-4, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>(dY_dims, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "sum", -1, 1e-4, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "none", -1, 1e-4, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>(dY_dims, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "mean", 0, 1e-4, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>(dY_dims, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "sum", 0, 1e-4, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, float>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                        "none", 0, 1e-4, true);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossInternalGrad_TinySizeTensor_half) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{2};
  std::vector<int64_t> dX_dims{8, 2};
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                                dX_dims, "mean", -1, 5e-2);
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                                dX_dims, "sum", -1, 5e-2);
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                                "none", -1, 5e-2);

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                                dX_dims, "mean", 0, 5e-2);
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                                dX_dims, "sum", 0, 5e-2);
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                                "none", 0, 5e-2);

  // Bias.
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                                dX_dims, "mean", -1, 5e-2, true);
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                                dX_dims, "sum", -1,
                                                                5e-2, true);
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                                "none", -1, 5e-2, true);
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                                dX_dims, "mean", 0, 5e-2, true);
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                                dX_dims, "sum", 0, 5e-2, true);
  TestSoftmaxCrossEntropyLossInternalGrad<MLFloat16, MLFloat16>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                                "none", 0, 5e-2, true);
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossInternalGrad_TinySizeTensorFloatInputHalfOutput) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{2};
  std::vector<int64_t> dX_dims{8, 2};
  // Set run_cpu_baseline_seperately = True because CPU kernel did not support multiple type support
  // for input and output.
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                            dX_dims, "mean", -1, 5e-2, false /*has_bias*/);
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                            dX_dims, "sum", -1, 5e-2, false /*has_bias*/);
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                            "none", -1, 5e-2, false /*has_bias*/);

  //  Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                            dX_dims, "mean", 0, 5e-2, false /*has_bias*/);
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                            dX_dims, "sum", 0, 5e-2, false /*has_bias*/);
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                            "none", 0, 5e-2, false /*has_bias*/);

  // Bias.
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                            dX_dims, "mean", -1, 5e-2, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                            dX_dims, "sum", -1, 1e0, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                            "none", -1, 5e-2, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                            dX_dims, "mean", 0, 5e-2, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>(dY_dims, log_prob_dims, index_dims, weight_dims,
                                                            dX_dims, "sum", 0, 5e-2, true);
  TestSoftmaxCrossEntropyLossInternalGrad<float, MLFloat16>({8}, log_prob_dims, index_dims, weight_dims, dX_dims,
                                                            "none", 0, 5e-2, true);
}

}  // namespace test
}  // namespace onnxruntime
