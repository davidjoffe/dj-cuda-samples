#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString* kMetalSrc = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device const float* in  [[buffer(0)]],
                    device float*       out [[buffer(1)]],
                    uint gid [[thread_position_in_grid]])
{
    out[gid] = in[gid] + 1.0f;
}
)";

int main()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal device not available.");
            return 1;
        }

        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kMetalSrc options:nil error:&err];
        if (!lib) {
            NSLog(@"Failed to compile Metal source: %@", err);
            return 1;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"add_one"];
        if (!fn) {
            NSLog(@"Failed to find kernel function add_one.");
            return 1;
        }

        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            NSLog(@"Failed to create compute pipeline: %@", err);
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            NSLog(@"Failed to create command queue.");
            return 1;
        }

        const NSUInteger N = 16;
        float in[N];
        for (NSUInteger i = 0; i < N; ++i) in[i] = (float)i;

        id<MTLBuffer> inBuf  = [device newBufferWithBytes:in length:sizeof(in) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [device newBufferWithLength:sizeof(in) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:inBuf offset:0 atIndex:0];
        [enc setBuffer:outBuf offset:0 atIndex:1];

        // Dispatch N threads total (1D)
        MTLSize grid  = MTLSizeMake(N, 1, 1);
        NSUInteger tgs = MIN(pso.maxTotalThreadsPerThreadgroup, N);
        MTLSize group = MTLSizeMake(tgs, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        float* out = (float*)outBuf.contents;

        // Print results (our “GPU hello world” proof)
        NSLog(@"Metal compute done. First few:");
        for (NSUInteger i = 0; i < N; ++i) {
            NSLog(@"%2lu: %.1f -> %.1f", (unsigned long)i, in[i], out[i]);
        }
    }

    return 0;
}
