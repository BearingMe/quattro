#include "tensor_ops.h"
#include "core/numerics.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace reference {
namespace {
float Mul(float a, float b) { volatile float r = a * b; return r; }
float Add(float a, float b) { volatile float r = a + b; return r; }
uint32_t Load(const std::vector<uint8_t>& s, const core::TensorView& v,
              uint32_t x, uint32_t y, uint32_t c) {
    uint32_t bits = 0;
    const uint32_t size = core::ElementBytes(v.type);
    memcpy(&bits, s.data() + v.offset + size_t(y)*v.rowStride + size_t(x)*v.pixelStride + c*size, size);
    return bits;
}
int S8(uint32_t bits) { return bits >= 128 ? int(bits) - 256 : int(bits); }
uint32_t Quant(float value, bool relu) {
    const float bounded = std::clamp(value, relu ? 0.0f : -128.0f, 127.0f);
    const float lo = std::floor(bounded), frac = bounded - lo;
    const int lower = int(lo);
    const int result = lower + (frac > 0.5f || (frac == 0.5f && lower % 2 != 0) ? 1 : 0);
    return uint32_t(result) & 255;
}
}

std::vector<uint8_t> CpuTensorOp(const core::TensorOpDesc& d, const std::vector<uint8_t>& s) {
    using core::TensorOp;
    const uint32_t size = core::ElementBytes(d.outputType);
    const size_t bytes = size_t(d.width)*d.height*d.channels*size;
    std::vector<uint8_t> out((bytes+3)&~size_t(3),0);
    for (uint32_t y=0; y<d.height; ++y)
        for (uint32_t x=0; x<d.width; ++x)
            for (uint32_t c=0; c<d.channels; ++c) {
                uint32_t result = 0;
                if (d.op == TensorOp::Pad) {
                    if (x >= d.padX && y >= d.padY && x-d.padX < d.a.width && y-d.padY < d.a.height)
                        result = Load(s,d.a,x-d.padX,y-d.padY,c);
                } else if (d.op == TensorOp::Concat) {
                    result = c < d.a.channels ? Load(s,d.a,x,y,c) : Load(s,d.b,x,y,c-d.a.channels);
                } else if (d.op == TensorOp::Spatial2x) {
                    result = Load(s,d.a,x/2,y/2,(2*(y&1)+(x&1))*d.channels+c);
                } else {
                    const uint32_t bits = Load(s,d.a,x,y,c);
                    if (d.op == TensorOp::Copy) result = bits;
                    else if (d.op == TensorOp::Relu) result = uint32_t(std::max(0,S8(bits)));
                    else {
                        core::ChannelTransform t;
                        memcpy(&t,s.data()+d.transformOffset+size_t(c)*16,16);
                        if (d.op == TensorOp::Quantize) {
                            float input;
                            if (d.a.type == core::TensorType::F16) input=core::F16ToF32(uint16_t(bits));
                            else memcpy(&input,&bits,4);
                            result = Quant(Mul(Add(input,core::F16ToF32(uint16_t(t.biasF16))),t.reciprocalOutputScale),false);
                        } else {
                            int32_t acc;
                            memcpy(&acc,&bits,4);
                            float value = Mul(float(acc),t.inputScale);
                            const float bias = core::F16ToF32(uint16_t(t.biasF16));
                            if (d.op == TensorOp::HalfOutput) {
                                value = core::F16ToF32(core::F32ToF16(value));
                                result = core::F32ToF16(Add(value,bias));
                            } else {
                                value = Add(value,bias);
                                if (d.op == TensorOp::AffineResidual)
                                    value = Add(value,Mul(float(S8(Load(s,d.b,x,y,c))),t.residualScale));
                                result = Quant(Mul(value,t.reciprocalOutputScale),d.relu);
                            }
                        }
                    }
                }
                memcpy(out.data()+((size_t(y)*d.width+x)*d.channels+c)*size,&result,size);
            }
    return out;
}
} // namespace reference
