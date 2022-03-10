# nerf
PyTorch Implementation of Neural Radiance Fields.
## Author:
Chris Fritz fritz17236 AT hotmail DOT com

## Files
- constants.py : Contains list of network, simulation, and data configuration parameters. 
- definitions.py: Contains class definitions including the nerf network MLP, and PyTorch datasets; contains function definitions including data i/o, and ray-tracing / volumetric rendering helpers.
- network_model: script to load and train dataset according to configuration parameters specified in constants.py
- query_network: script to a) generate a synthetic video panning around the visual scene and b) run a numerical experiment plotting PSNR vs distance to nearest training image.
- utils.py: additional helper functions (not specific to nerf) 
- requirements.txt: list of project dependencies (includes both classic nerf and this fork) - of particular note: CUDA is highly recommended. 

