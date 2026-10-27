// One thread owns a full output dword: packed INT8/FP16 tails cannot race.
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

uint H(uint i) { return src.Load(i * 4); }
uint Bytes(uint type) { return type == 0 ? 1 : (type == 1 ? 2 : 4); }
uint Read(uint view, uint x, uint y, uint c) {
    uint size = Bytes(H(view+6));
    uint address = H(view+5) + y * H(view+4) + x * H(view+3) + c * size;
    uint word = src.Load(address & ~3u);
    return (word >> ((address & 3) * 8)) & (size == 4 ? 0xffffffff : (size == 2 ? 65535 : 255));
}
int Signed8(uint bits) { return int(bits << 24) >> 24; }
uint HalfRne(float value) {
    // Explicit IEEE RNE packing. The initial gfx1101 f32tof16 probes disagreed
    // at overflow and rounding boundaries; do not depend on that conversion.
    uint bits = asuint(value), magnitude = bits & 0x7fffffff;
    uint sign = (bits >> 16) & 0x8000, exponent = magnitude >> 23;
    if (magnitude >= 0x477ff000) return sign | 0x7c00;
    if (exponent >= 113) {
        uint rounded = (magnitude + 0xfff + ((magnitude >> 13) & 1)) >> 13;
        return sign | (rounded - 0x1c000);
    }
    if (exponent < 102) return sign;
    uint shift = 126 - exponent;
    uint mantissa = (magnitude & 0x7fffff) | 0x800000;
    uint result = mantissa >> shift;
    uint remainder = mantissa & ((1u << shift) - 1);
    uint midpoint = 1u << (shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (result & 1))) ++result;
    return sign | result;
}
uint Element(uint i) {
    uint width = H(15), channels = H(17);
    uint c = i % channels, x = (i / channels) % width, y = i / (channels * width);
    uint op = H(0);
    if (op == 6) // concat, preserves each source's own channel-group scale
        return c < H(3) ? Read(1,x,y,c) : Read(8,x,y,c-H(3));
    if (op == 7) {
        uint px = H(20), py = H(21);
        if (x < px || y < py || x-px >= H(1) || y-py >= H(2)) return 0;
        return Read(1,x-px,y-py,c);
    }
    if (op == 8) // four independent k2s2 transposed-conv phases, not interpolation
        return Read(1,x/2,y/2,((y%2)*2+x%2)*channels+c);
    uint a = Read(1,x,y,c);
    if (op == 0) return uint(max(Signed8(a),0));
    if (op == 5) return a;
    uint t = H(19) + c * 16;
    float inputScale = asfloat(src.Load(t));
    float residualScale = asfloat(src.Load(t+4));
    float reciprocal = asfloat(src.Load(t+8));
    float bias = f16tof32(src.Load(t+12));
    // Explicitly separate FP32 steps. AMD source formulas can contract; this
    // reproducible contract does not claim bit-equivalence to all fused PSOs.
    precise float value = op == 1 ? (H(7) == 1 ? f16tof32(a) : asfloat(a)) : float(asint(a));
    if (op == 1) {
        value = value + bias;
        value = value * reciprocal;
    }
    else {
        value = value * inputScale;
        if (op == 4) {
            // Two boundaries: half(acc*scale), then half(+half bias).
            value = f16tof32(HalfRne(value));
            value = value + bias;
            return HalfRne(value);
        }
        value = value + bias;
        if (op == 3) {
            precise float residual = float(Signed8(Read(8,x,y,c))) * residualScale;
            value = value + residual;
        }
        value = value * reciprocal;
    }
    // Clamp in floating point before integer conversion, including overflow
    // to infinity from finite operands/scales. No undefined float->int cast.
    return uint(int(round(clamp(value, H(22) != 0 ? 0.0f : -128.0f, 127.0f)))) & 255;
}

[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint size = Bytes(H(18)), count = H(15)*H(16)*H(17);
    uint wordIndex = tid.x + tid.y * H(23);
    uint perWord = 4/size, start = wordIndex * perWord;
    if (start >= count) return;
    uint result = 0;
    for (uint k = 0; k < perWord; ++k)
        if (start+k < count) result |= Element(start+k) << (k*size*8);
    dst.Store(wordIndex*4,result);
}
