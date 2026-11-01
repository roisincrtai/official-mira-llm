#include <torch/extension.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

id<MTLLibrary> library = nil;
std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

std::string describe(NSError *error) {
    return error == nil ? std::string("no reason given")
                        : std::string([[error localizedDescription] UTF8String]);
}

id<MTLDevice> device() {
    return at::mps::MPSDevice::getInstance()->device();
}

void load(const std::string &path) {
    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    id<MTLLibrary> loaded = [device() newLibraryWithURL:url error:&error];
    TORCH_CHECK(loaded != nil, "mira_metal: cannot load ", path, ": ", describe(error));
    library = loaded;
    pipelines.clear();
}

id<MTLComputePipelineState> pipeline(const std::string &name) {
    auto found = pipelines.find(name);
    if (found != pipelines.end()) {
        return found->second;
    }
    TORCH_CHECK(library != nil, "mira_metal: no metallib is loaded");
    id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    TORCH_CHECK(function != nil, "mira_metal: ", name, " is not in the metallib");
    NSError *error = nil;
    id<MTLComputePipelineState> state =
        [device() newComputePipelineStateWithFunction:function error:&error];
    [function release];
    TORCH_CHECK(state != nil, "mira_metal: ", name, ": ", describe(error));
    pipelines.emplace(name, state);
    return state;
}

int64_t max_threads(const std::string &name) {
    return static_cast<int64_t>([pipeline(name) maxTotalThreadsPerThreadgroup]);
}

bool has(const std::string &name) {
    TORCH_CHECK(library != nil, "mira_metal: no metallib is loaded");
    id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (function == nil) {
        return false;
    }
    [function release];
    return true;
}

void launch(const std::string &name, const std::vector<at::Tensor> &buffers,
            const std::vector<int64_t> &params, int64_t threads, int64_t group) {
    id<MTLComputePipelineState> state = pipeline(name);
    TORCH_CHECK(group > 0 && static_cast<NSUInteger>(group) <= [state maxTotalThreadsPerThreadgroup],
                "mira_metal: ", name, " cannot run ", group, " threads per threadgroup");
    for (const at::Tensor &tensor : buffers) {
        TORCH_CHECK(tensor.device().is_mps(), "mira_metal: every buffer must be an MPS tensor");
    }
    const std::vector<int32_t> values(params.begin(), params.end());
    const std::vector<at::Tensor> bound(buffers);
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:state];
        for (size_t index = 0; index < bound.size(); ++index) {
            const at::Tensor &tensor = bound[index];
            [encoder setBuffer:__builtin_bit_cast(id<MTLBuffer>, tensor.storage().data())
                        offset:tensor.storage_offset() * tensor.element_size()
                       atIndex:index];
        }
        [encoder setBytes:values.data()
                   length:values.size() * sizeof(int32_t)
                  atIndex:bound.size()];
        [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(threads), 1, 1)
           threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(group), 1, 1)];
    });
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("load", &load);
    module.def("has", &has);
    module.def("max_threads", &max_threads);
    module.def("launch", &launch);
}
