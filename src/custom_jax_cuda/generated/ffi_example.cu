// This file was automatically generated
// You can modify it, but I recommend automatically regenerating this code whenever you adapt 
// one of the kernels. The FFI Bindings are very tedious in jax and they involve a lot of 
// boilerplate code that is easy to mess up.

#include <map>
#include <tuple>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

// A wrapper to encapsulate an FFI call
template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}
#include "../common/math.cuh"
#include "../ffi_example.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SimpleArange                              */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SimpleArangeFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer add,
    ffi::Result<ffi::AnyBuffer> output,
    int size,
    int p,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(output->element_count(), block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* add_val = reinterpret_cast<int*>(add.untyped_data());
    int* output_val = reinterpret_cast<int*>(output->untyped_data());

    void* args[] = {
        &add_val,
        &output_val,
        &size
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<int>;
    using TFunctionType = decltype(SimpleArange<1>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{1}] = SimpleArange<1>;
    instance_map[{2}] = SimpleArange<2>;
    instance_map[{3}] = SimpleArange<3>;
    instance_map[{4}] = SimpleArange<4>;
    instance_map[{5}] = SimpleArange<5>;

    auto it = instance_map.find({p});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (p)"\
            " in SimpleArangeFFIHost -- Only supporting:\n"\
            "(1), (2), (3), (4), (5)"
        );
    }

    TFunctionType* instance = it->second;
    
    cudaLaunchKernel((const void*)instance, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SimpleArangeFFI, SimpleArangeFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // add
        .Ret<ffi::AnyBuffer>() // output
        .Attr<int>("size")
        .Attr<int>("p")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SetToConstantCall                         */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SetToConstantCallFFIHost(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> output,
    int value,
    size_t block_size,
    int tpar
) {
    int size = output->element_count();

    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<int>;
    using TFunctionType = decltype(SetToConstantCall<16>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{16}] = SetToConstantCall<16>;
    instance_map[{32}] = SetToConstantCall<32>;
    instance_map[{64}] = SetToConstantCall<64>;

    auto it = instance_map.find({tpar});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (tpar)"\
            " in SetToConstantCallFFIHost -- Only supporting:\n"\
            "(16), (32), (64)"
        );
    }

    TFunctionType* instance = it->second;

    // Now call our function
    instance(
        stream,
        reinterpret_cast<int*>(output->untyped_data()),
        value,
        size,
        block_size
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SetToConstantCallFFI, SetToConstantCallFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ret<ffi::AnyBuffer>() // output
        .Attr<int>("value")
        .Attr<size_t>("block_size")
        .Attr<int>("tpar"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: NestedTemplate                            */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error NestedTemplateFFIHost(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> output,
    int p1,
    int p2,
    bool flag,
    size_t block_size
) {
    int size = output->element_count();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(output->element_count(), block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* output_val = reinterpret_cast<int*>(output->untyped_data());

    void* args[] = {
        &output_val,
        &size
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<int, int, bool>;
    using TFunctionType = decltype(NestedTemplate<0, 22, true>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{0, 22, true}] = NestedTemplate<0, 22, true>;
    instance_map[{0, 22, false}] = NestedTemplate<0, 22, false>;
    instance_map[{0, 33, true}] = NestedTemplate<0, 33, true>;
    instance_map[{0, 33, false}] = NestedTemplate<0, 33, false>;
    instance_map[{1, 22, true}] = NestedTemplate<1, 22, true>;
    instance_map[{1, 22, false}] = NestedTemplate<1, 22, false>;
    instance_map[{1, 33, true}] = NestedTemplate<1, 33, true>;
    instance_map[{1, 33, false}] = NestedTemplate<1, 33, false>;

    auto it = instance_map.find({p1, p2, flag});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (p1, p2, flag)"\
            " in NestedTemplateFFIHost -- Only supporting:\n"\
            "(0, 22, true), (0, 22, false), (0, 33, true), (0, 33, false), (1, 22, true), (1, 22, false), (1, 33, true), (1, 33, false)"
        );
    }

    TFunctionType* instance = it->second;
    
    cudaLaunchKernel((const void*)instance, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NestedTemplateFFI, NestedTemplateFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ret<ffi::AnyBuffer>() // output
        .Attr<int>("p1")
        .Attr<int>("p2")
        .Attr<bool>("flag")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_example, m) {
    m.def("SimpleArange", []() { return EncapsulateFfiCall(&SimpleArangeFFI); });
    m.def("SetToConstantCall", []() { return EncapsulateFfiCall(&SetToConstantCallFFI); });
    m.def("NestedTemplate", []() { return EncapsulateFfiCall(&NestedTemplateFFI); });
}