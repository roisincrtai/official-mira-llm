#include <torch/extension.h>
#include <algorithm>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void mira_affine_launch(uint32_t, void *, void *, void *, void *, void *,
                                    uint64_t, uint32_t, uint32_t, uint32_t,
                                    uint32_t, uint32_t);

void affine(const at::Tensor &matrix, const at::Tensor &carry,
            at::Tensor matrixOut, at::Tensor carryOut,
            int64_t distance) {
    TORCH_CHECK(matrix.dim() == 5 && carry.dim() == 5, "affine tensors must have rank 5");
    auto B = matrix.size(0), T = matrix.size(1), C = matrix.size(2), K = matrix.size(3);
    auto M = carry.size(4);
    TORCH_CHECK(B > 0 && T > 0 && C > 0 && K > 0 && K <= 8 && matrix.size(4) == 8,
                "affine matrix shape must be [B,T,C,K,8], 1 <= K <= 8");
    TORCH_CHECK(M >= 8 && M <= 64 && M % 8 == 0 && distance > 0 && distance < T,
                "affine memory or distance is unsupported");
    TORCH_CHECK(carry.size(0) == B && carry.size(1) == T && carry.size(2) == C &&
                carry.size(3) == K, "affine carry shape must be [B,T,C,K,M]");
    TORCH_CHECK(matrixOut.sizes() == matrix.sizes() && carryOut.sizes() == carry.sizes(),
                "affine output shapes differ from inputs");
    for (const auto &value : {matrix, carry, matrixOut, carryOut}) {
        TORCH_CHECK(value.device().type() == c10::DeviceType::PrivateUse1 &&
                    value.device() == matrix.device() && value.scalar_type() == at::kFloat &&
                    value.is_contiguous(), "affine tensors must be contiguous FP32 on one NPU");
    }
    TORCH_CHECK(matrix.data_ptr() != matrixOut.data_ptr() &&
                carry.data_ptr() != carryOut.data_ptr(), "affine outputs must not alias inputs");
    c10_npu::NPUGuard guard(matrix.device());
    uint64_t planes = B * T * C;
    uint32_t blocks = std::min<uint64_t>(32, planes);
    auto handler = [matrix, carry, matrixOut, carryOut, planes, T, C, K, M, distance, blocks]() -> int {
        auto stream = c10_npu::getCurrentNPUStream(matrix.device().index()).stream();
        mira_affine_launch(blocks, stream, matrix.data_ptr(), carry.data_ptr(),
                           matrixOut.data_ptr(), carryOut.data_ptr(), planes, T, C, K, M, distance);
        return 0;
    };
    at_npu::native::OpCommand command;
    command.Name("MiraAffine").Input(matrix).Input(carry).Output(matrixOut).Output(carryOut)
        .SetCustomHandler(handler).Run();
}

PYBIND11_MODULE(mira_ascend, module) {
    module.def("affine", &affine);
    module.def("soc_version", []() { return std::string(MIRA_SOC_VERSION); });
}
