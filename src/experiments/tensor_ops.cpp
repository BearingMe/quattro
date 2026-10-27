#include "experiments.h"
#include "core/numerics.h"
#include "gpu/tensor_ops.h"
#include "reference/tensor_ops.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace experiments {
namespace {
using core::TensorType;
using core::TensorOp;
using core::TensorOpDesc;
using core::TensorView;
using core::ChannelTransform;

struct Fixture {
    TensorOpDesc d;
    std::vector<uint8_t> data;

    TensorView Add(uint32_t w, uint32_t h, uint32_t c, TensorType type,
                   uint32_t extraChannels = 0) {
        const uint32_t size = core::ElementBytes(type);
        const uint32_t offset = uint32_t((data.size()+3)&~size_t(3));
        TensorView v{w,h,c,(c+extraChannels)*size,(w*(c+extraChannels)+3)*size,offset,type};
        data.resize(offset+size_t(h)*v.rowStride,0xcd);
        for (uint32_t y=0; y<h; ++y)
            for (uint32_t x=0; x<w; ++x)
                for (uint32_t z=0; z<c; ++z) {
                    const int n = int((y*w*c+x*c+z)*37 % 256)-128;
                    const int32_t i32 = n*127;
                    const uint16_t half = core::F32ToF16(float(n)*0.25f);
                    const float f32 = float(n)*0.25f;
                    const uint8_t i8 = uint8_t(n);
                    const void* p = type == TensorType::I8 ? static_cast<const void*>(&i8) :
                        (type == TensorType::F16 ? static_cast<const void*>(&half) :
                        (type == TensorType::F32 ? static_cast<const void*>(&f32) : static_cast<const void*>(&i32)));
                    memcpy(data.data()+offset+size_t(y)*v.rowStride+x*v.pixelStride+z*size,p,size);
                }
        return v;
    }
    void Set(const TensorView& v, uint32_t index, uint32_t bits) {
        const uint32_t x = (index/v.channels)%v.width, y = index/(v.channels*v.width), c=index%v.channels;
        memcpy(data.data()+v.offset+size_t(y)*v.rowStride+x*v.pixelStride+c*core::ElementBytes(v.type),
               &bits,core::ElementBytes(v.type));
    }
    void Transforms(const std::vector<ChannelTransform>& t) {
        d.transformOffset = uint32_t((data.size()+3)&~size_t(3));
        data.resize(d.transformOffset+t.size()*16);
        memcpy(data.data()+d.transformOffset,t.data(),t.size()*16);
    }
};

Fixture Unary(TensorOp op, TensorType in, TensorType out, uint32_t w, uint32_t h, uint32_t c) {
    Fixture f;
    f.d.op=op; f.d.a=f.Add(w,h,c,in,3); f.d.width=w; f.d.height=h; f.d.channels=c; f.d.outputType=out;
    return f;
}

bool Dump(const std::filesystem::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream out(p,std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));
    return bool(out);
}
}

int RunTensorOps() {
    gpu::DeviceContext ctx;
    std::string error;
    if (!gpu::CreateComputeDevice(ctx,error)) {
        std::fprintf(stderr,"error: %s\n",error.c_str()); return 1;
    }
    std::printf("M10 adapter: %ls vendor=%04x device=%04x; debug + GPU validation enabled\n",
        ctx.adapter.name.c_str(),ctx.adapter.vendorId,ctx.adapter.deviceId);
    wchar_t exe[32768];
    const DWORD pathLength = GetModuleFileNameW(nullptr,exe,32768);
    if (!pathLength || pathLength >= 32768) return 1;
    const auto reports = std::filesystem::path(exe).parent_path().parent_path()/"reports"/"m10";
    std::error_code ec;
    std::filesystem::create_directories(reports,ec);
    if (ec) { std::fprintf(stderr,"error: report directory: %s\n",ec.message().c_str()); return 1; }
    bool ok=true;
    size_t passed=0, rejected=0;
    const auto check = [&](const char* name, Fixture& f, const std::vector<uint8_t>* anchor=nullptr) {
        error.clear();
        if (!gpu::ValidateTensorOp(f.d,f.data,error)) {
            std::fprintf(stderr,"FAIL %s validation: %s\n",name,error.c_str()); ok=false; return;
        }
        const auto cpu = reference::CpuTensorOp(f.d,f.data);
        std::vector<uint8_t> warm,last;
        if (!gpu::RunTensorOp(ctx,f.d,f.data,warm,last,error)) {
            std::fprintf(stderr,"FAIL %s dispatch: %s\n",name,error.c_str()); ok=false; return;
        }
        const bool equal = warm==cpu && last==cpu && (!anchor || cpu==*anchor);
        std::printf("%s %-34s %ux%ux%u type=%u bytes=%zu (warm + 3 repeats)\n",
            equal?"PASS":"FAIL",name,f.d.width,f.d.height,f.d.channels,uint32_t(f.d.outputType),cpu.size());
        if (!equal) {
            for (size_t i=0;i<cpu.size();++i) if (warm[i]!=cpu[i] || last[i]!=cpu[i]) {
                std::fprintf(stderr,"  first mismatch byte %zu cpu=%02x warm=%02x last=%02x\n",i,cpu[i],warm[i],last[i]); break;
            }
            ok=false;
        } else ++passed;
        const bool saved = Dump(reports/(std::string(name)+".input.bin"),f.data) &&
            Dump(reports/(std::string(name)+".cpu.bin"),cpu) &&
            Dump(reports/(std::string(name)+".gpu.bin"),last);
        std::ofstream meta(reports/(std::string(name)+".txt"));
        const auto view = [&](const char* label,const TensorView& v) {
            meta << label << " WHC=" << v.width << ',' << v.height << ',' << v.channels
                 << " pixelStride=" << v.pixelStride << " rowStride=" << v.rowStride
                 << " offset=" << v.offset << " type=" << uint32_t(v.type) << '\n';
        };
        view("a",f.d.a); view("b",f.d.b);
        meta << "op=" << uint32_t(f.d.op) << " outputWHC=" << f.d.width << ',' << f.d.height << ',' << f.d.channels
             << " outputType=" << uint32_t(f.d.outputType) << " transformOffset=" << f.d.transformOffset
             << " padXY=" << f.d.padX << ',' << f.d.padY << " relu=" << f.d.relu << '\n';
        if (!saved || !meta) { std::fprintf(stderr,"FAIL report write\n"); ok=false; }
    };
    const auto reject = [&](const char* name, const Fixture& f) {
        error.clear();
        const bool bad = !gpu::ValidateTensorOp(f.d,f.data,error);
        std::printf("%s rejection: %s\n",bad?"PASS":"FAIL",name);
        if (bad) ++rejected; else ok=false;
    };

    // Full signed-byte domain, with a non-dword logical tail in another case.
    auto relu=Unary(TensorOp::Relu,TensorType::I8,TensorType::I8,1,1,256);
    for (uint32_t i=0;i<256;++i) relu.Set(relu.d.a,i,i);
    std::vector<uint8_t> reluAnchor(256);
    for (uint32_t i=0;i<128;++i) reluAnchor[i]=uint8_t(i);
    check("relu-all-encodings",relu,&reluAnchor);

    for (uint32_t c : {1u,7u,16u,32u,64u}) {
        const auto suffix=std::to_string(c);
        auto copy=Unary(TensorOp::Copy,TensorType::I8,TensorType::I8,5,3,c);
        check(("copy-strided-c"+suffix).c_str(),copy);
        auto activation=Unary(TensorOp::Relu,TensorType::I8,TensorType::I8,5,3,c);
        check(("relu-c"+suffix).c_str(),activation);
        auto affine=Unary(TensorOp::Affine,TensorType::I32,TensorType::I8,5,3,c);
        affine.d.b=affine.Add(5,3,c,TensorType::I8,5);
        std::vector<ChannelTransform> transforms(c);
        for (uint32_t z=0;z<c;++z) transforms[z]={0.00390625f,z<c/2?0.125f:0.5f,
            z<c/2?4.0f:8.0f,core::F32ToF16(float(int(z%5)-2)*0.25f)};
        affine.Transforms(transforms);
        check(("affine-c"+suffix).c_str(),affine);
        affine.d.relu=true;
        check(("affine-relu-c"+suffix).c_str(),affine);
        affine.d.relu=false; affine.d.op=TensorOp::AffineResidual;
        check(("scaled-residual-c"+suffix).c_str(),affine);
        auto pad=copy;
        pad.d.op=TensorOp::Pad; pad.d.padX=1; pad.d.padY=2; pad.d.width=7; pad.d.height=7;
        check(("zero-pad-c"+suffix).c_str(),pad);
        auto concat=copy;
        concat.d.op=TensorOp::Concat; concat.d.b=concat.Add(5,3,c+2,TensorType::I8,1);
        concat.d.channels=c*2+2;
        check(("concat-c"+suffix).c_str(),concat);
        // A split is a view: retain strides, advance by channel bytes.
        auto split=concat;
        split.d.op=TensorOp::Copy; split.d.channels=c; split.d.a=concat.d.b;
        split.d.a.offset+=1; split.d.a.channels=c;
        check(("split-view-c"+suffix).c_str(),split);
    }
    for (TensorType type : {TensorType::I8,TensorType::F16,TensorType::I32}) {
        auto scatter=Unary(TensorOp::Spatial2x,type,type,3,2,28);
        scatter.d.width=6; scatter.d.height=4; scatter.d.channels=7;
        check(("spatial2x-type"+std::to_string(uint32_t(type))).c_str(),scatter);
        auto concat=Unary(TensorOp::Concat,type,type,1,2,3);
        concat.d.b=concat.Add(1,2,2,type,2); concat.d.channels=5;
        check(("concat-type"+std::to_string(uint32_t(type))).c_str(),concat);
    }
    // Literal upstream phase order: (0,0),(1,0),(0,1),(1,1).
    // Two adjacent low-res pixels ensure rows are not accidentally interleaved.
    auto phaseAnchor=Unary(TensorOp::Spatial2x,TensorType::I8,TensorType::I8,2,1,4);
    for (uint32_t i=0;i<8;++i) phaseAnchor.Set(phaseAnchor.d.a,i,i+1);
    phaseAnchor.d.width=4; phaseAnchor.d.height=2; phaseAnchor.d.channels=1;
    std::vector<uint8_t> phaseExpected={1,2,5,6,3,4,7,8};
    check("spatial2x-literal-phase-order",phaseAnchor,&phaseExpected);
    auto catAnchor=Unary(TensorOp::Concat,TensorType::I8,TensorType::I8,2,1,2);
    catAnchor.d.b=catAnchor.Add(2,1,1,TensorType::I8,2); catAnchor.d.channels=3;
    for (uint32_t i=0;i<4;++i) catAnchor.Set(catAnchor.d.a,i,i+1);
    catAnchor.Set(catAnchor.d.b,0,9); catAnchor.Set(catAnchor.d.b,1,10);
    std::vector<uint8_t> catExpected={1,2,9,3,4,10,0,0};
    check("concat-literal-channel-order",catAnchor,&catExpected);
    catAnchor.d.op=TensorOp::Copy; catAnchor.d.a.offset+=1; catAnchor.d.a.channels=1; catAnchor.d.channels=1;
    std::vector<uint8_t> splitExpected={2,4,0,0};
    check("split-literal-strided-view",catAnchor,&splitExpected);
    // FP16 input quantization: ties, saturation and signed endpoints.
    auto quant=Unary(TensorOp::Quantize,TensorType::F16,TensorType::I8,1,1,11);
    const float qvalues[]={-65504,-128.5f,-127.5f,-2.5f,-1.5f,-0.5f,0.5f,1.5f,2.5f,126.5f,65504};
    for (uint32_t i=0;i<11;++i) quant.Set(quant.d.a,i,core::F32ToF16(qvalues[i]));
    quant.Transforms(std::vector<ChannelTransform>(11));
    std::vector<uint8_t> quantAnchor={128,128,128,254,254,0,0,2,2,126,127,0};
    check("quantize-half-boundaries",quant,&quantAnchor);
    auto quantAll=Unary(TensorOp::Quantize,TensorType::F16,TensorType::I8,256,256,1);
    for (uint32_t i=0;i<65536;++i)
        quantAll.Set(quantAll.d.a,i,(i&0x7c00)==0x7c00?0:i);
    quantAll.Transforms({{1,1,1.0f/0.014884727f,0}});
    check("quantize-all-finite-half",quantAll);
    auto quantFloat=Unary(TensorOp::Quantize,TensorType::F32,TensorType::I8,7,3,16);
    quantFloat.Transforms(std::vector<ChannelTransform>(16,{1,1,1.0f/0.014884727f,core::F32ToF16(0.125f)}));
    check("quantize-fp32-with-half-bias",quantFloat);
    auto invalid=quant; invalid.Set(invalid.d.a,0,0x7c00); reject("infinite FP16",invalid);
    invalid=quant; invalid.Set(invalid.d.a,0,0x7e00); reject("NaN FP16",invalid);

    auto boundary=Unary(TensorOp::Affine,TensorType::I32,TensorType::I8,1,1,1);
    boundary.Set(boundary.d.a,0,4095);
    const float scale=1.0001220703125f;
    boundary.Transforms({{scale,1,1,core::F32ToF16(-4094)}});
    std::vector<uint8_t> two={2,0,0,0};
    check("requant-no-contraction",boundary,&two);
    auto denorm=Unary(TensorOp::Affine,TensorType::I32,TensorType::I8,1,1,1);
    denorm.Set(denorm.d.a,0,1073741824);
    denorm.Transforms({{std::ldexp(1.0f,-140),1,std::ldexp(1.0f,111),0}});
    check("subnormal-scale-preservation",denorm,&two);
    denorm.d.op=TensorOp::AffineResidual;
    denorm.d.b=denorm.Add(1,1,1,TensorType::I8);
    denorm.Set(denorm.d.a,0,1); denorm.Set(denorm.d.b,0,255);
    denorm.Transforms({{std::ldexp(1.5f,-126),std::ldexp(1.0f,-126),std::ldexp(1.0f,127),0}});
    std::vector<uint8_t> one={1,0,0,0};
    check("subnormal-intermediate-preservation",denorm,&one);
    // This anchor MUST distinguish contraction, not merely happen to agree.
    const float fused=std::fma(4095.0f,scale,-4094.0f);
    if (!(fused<1.5f)) { std::fprintf(stderr,"FAIL contraction fixture\n"); ok=false; }

    auto half=Unary(TensorOp::HalfOutput,TensorType::I32,TensorType::F16,1,1,7);
    const int32_t accs[]={4097,-4097,1,0,65504,65520,3};
    std::vector<ChannelTransform> halfTransforms(7);
    for (uint32_t i=0;i<7;++i) half.Set(half.d.a,i,uint32_t(accs[i]));
    halfTransforms[0].biasF16=core::F32ToF16(-4096);
    halfTransforms[1].biasF16=core::F32ToF16(4096);
    halfTransforms[2].inputScale=0.0000000298023223876953125f; // half min-subnormal/2
    half.Transforms(halfTransforms);
    std::vector<uint8_t> halfAnchor(16);
    const uint16_t halfBits[]={0,0,0,0,0x7bff,0x7c00,0x4200};
    memcpy(halfAnchor.data(),halfBits,14);
    check("half-double-round-and-overflow",half,&halfAnchor);
    auto halfSweep=Unary(TensorOp::HalfOutput,TensorType::I32,TensorType::F16,31,17,32);
    std::vector<ChannelTransform> sweepTransforms(32);
    for (uint32_t c=0;c<32;++c) {
        sweepTransforms[c].inputScale=std::ldexp(1.0f,int(c)-26);
        sweepTransforms[c].biasF16=core::F32ToF16(float(int(c%7)-3)*0.0001f);
    }
    for (uint32_t i=0;i<31*17*32;++i) {
        const int32_t a=int32_t((i*1664525u+1013904223u)%32769)-16384;
        halfSweep.Set(halfSweep.d.a,i,uint32_t(a));
    }
    halfSweep.Transforms(sweepTransforms);
    check("half-exponent-and-subnormal-sweep",halfSweep);
    auto fullOutput=Unary(TensorOp::Spatial2x,TensorType::F16,TensorType::F16,960,540,32);
    fullOutput.d.width=1920; fullOutput.d.height=1080; fullOutput.d.channels=8;
    check("spatial2x-full-1080p-half-output",fullOutput);
    // Full INT32 conversion endpoints, negative saturation and per-channel scales.
    auto extremes=Unary(TensorOp::Affine,TensorType::I32,TensorType::I8,1,1,4);
    extremes.Set(extremes.d.a,0,0x80000000); extremes.Set(extremes.d.a,1,0x7fffffff);
    extremes.Set(extremes.d.a,2,0xffffffff); extremes.Set(extremes.d.a,3,1);
    extremes.Transforms(std::vector<ChannelTransform>(4));
    std::vector<uint8_t> extremeAnchor={128,127,255,1};
    check("int32-endpoints",extremes,&extremeAnchor);

    invalid=quant; invalid.d.a.rowStride=1; reject("short row stride",invalid);
    invalid=quant; invalid.d.a.offset=uint32_t(invalid.data.size()); reject("out-of-range view",invalid);
    invalid=quant; invalid.d.a.offset=1; reject("unaligned half view",invalid);
    invalid=quant; invalid.d.channels=0; reject("empty output",invalid);
    invalid=quant; invalid.d.width=0xffffffff; reject("overflowing output shape",invalid);
    invalid=quant; invalid.d.outputType=TensorType::I32; reject("wrong output type",invalid);
    invalid=quant; invalid.d.transformOffset=uint32_t(invalid.data.size()); reject("short transform storage",invalid);
    invalid=quant; invalid.d.transformOffset=1; reject("unaligned transforms",invalid);
    invalid=quant; invalid.Transforms(std::vector<ChannelTransform>(11,{1,1,0,0})); reject("zero reciprocal scale",invalid);
    invalid=quant; invalid.Transforms(std::vector<ChannelTransform>(11,{1,1,std::numeric_limits<float>::infinity(),0})); reject("infinite scale",invalid);
    invalid=quant; invalid.Transforms(std::vector<ChannelTransform>(11,{1,1,std::numeric_limits<float>::max(),0})); reject("unsafe FP32 range",invalid);
    invalid=boundary; invalid.d.op=TensorOp::AffineResidual; invalid.d.b=invalid.d.a; reject("wrong residual type",invalid);
    invalid=relu; invalid.d.op=TensorOp(99); reject("unknown operation",invalid);
    invalid=relu; invalid.d.op=TensorOp::Spatial2x; reject("wrong phase shape",invalid);
    invalid=relu; invalid.d.op=TensorOp::Concat; invalid.d.b=invalid.d.a; reject("wrong concat shape",invalid);
    invalid=relu; invalid.d.op=TensorOp::Pad; invalid.d.padX=0xffffffff; reject("overflowing padding",invalid);

    const auto guard = [&](const char* name,uint64_t terms,uint64_t bias,bool expected) {
        error.clear(); const bool actual=core::ValidateInt8Accumulation(terms,bias,error);
        const bool pass=actual==expected;
        std::printf("%s accumulation guard: %s\n",pass?"PASS":"FAIL",name);
        if (!pass) ok=false; else ++rejected;
    };
    guard("largest full-range safe K",131071,0,true);
    guard("-128*-128 crosses INT32",131072,0,false);
    guard("bias consumes remaining headroom",131071,16384,false);
    guard("uint64 terms overflow",std::numeric_limits<uint64_t>::max(),0,false);
    guard("zero terms",0,0,false);
    guard("bias exceeds INT32",1,2147483648ull,false);
    if (!gpu::InfoQueueClean(ctx,error)) { std::fprintf(stderr,"FAIL diagnostics: %s\n",error.c_str()); ok=false; }
    std::printf("M10 %s: %zu hardware operator cases, %zu rejection/boundary checks; snapshots build/reports/m10/\n",
        ok?"PASS":"FAIL",passed,rejected);
    return ok?0:1;
}
} // namespace experiments
