An OptiX 7 implementation of [SPCBPT: Subspace-based Probabilistic Connections for Bidirectional Path Tracing]([SPCBPT (ssufujia.github.io)](https://ssufujia.github.io/SPCBPT/)). 

### requirement (Environment on my computer)：

* OptiX 7.5.0
* Cuda 11.7
* Visual Studio 2019  
* Cmake 3.24.2

### How to Build:  

* Start up cmake-gui from the Start Menu.
* Select the "src" directory and the source code
* Create a build directory that isn't the same as the source directory. 
* Press "Configure" button and select the version of Visual Studio 2019.
* Select "x64" as the platform
* Press "OK".
* Set OptiX_INSTALL_DIR to wherever you installed OptiX, e.g., C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0 
* Press "Configure" button again.
* Press "generate" and then "open Project"
* Right click the optixPathTracer project and set it as Startup project and run the renderer program (in Release).

### Difference from the paper-version code:

Due to various reasons, some details of this implementation are slightly different from the paper-version code. 

* This implementation disables the t = 1 strategy, i.e., the strategy of light sub-path connecting to the eye sub-path directly, because it is usually of low efficiency. 
* The parts of cross-iteration reuse of light sub-path, environment map, and transparent material are not yet completed, I plan to implement them in the future update.  
* Direction is ignored in the classification. Position and normal are more important in most cases.
* Subspace Sampling Matrix is trained from an initial matrix built from the full contribution integral of the paths in the corresponding subspace pair to speed up the training.   
* Paths for training are traced by a simple unidirectional path tracer with NEE implementation.
* The over-bright fireflies are slightly more than my paper-version code, I would try to figure out and solve this problem in the future.