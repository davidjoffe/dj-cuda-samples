@rem dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
@rem (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE

@rem Docker build helper script (local)

@rem docker build -t dj-cuda-sample1:local -f .\samples\bouncing_balls\docker\Dockerfile .

docker buildx build --compress --progress=plain -t davidjoffe/dj-cuda-samples:local -f .\samples\bouncing_balls\docker\Dockerfile .
docker buildx build --compress --progress=plain -t davidjoffe/dj-molecular-sim:local -f .\samples\molecular_sim\docker\Dockerfile .

@rem docker buildx build --compress --progress=plain -t davidjoffe/dj-cuda-samples:latest -f .\samples\bouncing_balls\docker\Dockerfile .
