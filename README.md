

\# SNN-PYNQ-Z2



\## Spiking Neural Network Accelerator on FPGA



An FPGA-based Spiking Neural Network (SNN) implementation targeting the PYNQ-Z2 platform, featuring CIFAR-10 classification, INT8 weight quantization, and FPGA hardware integration.



\## 1. Project Overview



This project explores the implementation of a Spiking Neural Network accelerator on FPGA.



The development workflow includes:



\- SNN model training using Python and PyTorch

\- CIFAR-10 dataset experimentation

\- Model weight export and quantization

\- Hardware parameter generation

\- RTL implementation using Verilog

\- FPGA integration using Xilinx Vivado

\- PYNQ-based hardware deployment



\## 2. Repository Structure



\- `docs/` - Project documentation and reports

\- `fpga/bitstreams/` - FPGA deployment artifacts

\- `fpga/completion\_pack/` - FPGA completion package

\- `fpga/vivado/v6\_cifar10/` - Vivado V6 CIFAR-10 project

\- `software/models/` - Trained model checkpoints

\- `software/training/` - Training notebooks and scripts

\- `software/validation/` - Validation artifacts



\## 3. Hardware Platform



\- FPGA Board: PYNQ-Z2

\- FPGA Development Tool: Xilinx Vivado

\- Hardware Description Language: Verilog

\- Weight Quantization: INT8

\- Dataset: CIFAR-10



\## 4. Software Stack



\- Python

\- PyTorch

\- NumPy

\- Jupyter Notebook



\## 5. FPGA Implementation



The V6 CIFAR-10 implementation includes:



\- SNN hardware architecture

\- Convolution processing engines

\- LIF neuron processing

\- Batch normalization parameters

\- INT8 quantized weights

\- Hardware memory initialization files

\- Vivado block design and IP integration



\## 6. Deployment



The repository includes:



\- FPGA bitstream (`.bit`)

\- Hardware handoff file (`.hwh`)

\- PYNQ board-side Python scripts



Refer to `fpga/completion\_pack/README.md` for deployment information.



\## 7. Validation



Simulation testbenches and Python validation artifacts are maintained in the repository.



\## 8. Project Status



Development and hardware integration artifacts are organized for further testing, validation, and deployment on the PYNQ-Z2 platform.



