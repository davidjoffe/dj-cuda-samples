@rem dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
@rem (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
@rem
@rem Docker run script
@rem 

@rem docker run --gpus all --runtime=nvidia dj-cuda-sample1:local
docker run --gpus all --runtime=nvidia davidjoffe/dj-cuda-samples:local
docker run --gpus all --runtime=nvidia davidjoffe/dj-molecular-sim:local
