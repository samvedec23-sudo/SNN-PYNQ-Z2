#!/bin/bash
# Needs: python3 + numpy, Icarus Verilog (sudo apt install iverilog)  -- or run the same files in Vivado xsim.
set -e
cd "$(dirname "$0")"
python3 gen_vectors.py
iverilog -g2005 -o tb_lif.vvp ../rtl/lif_bn_unit.v tb_lif_bn_unit.v && vvp tb_lif.vvp
iverilog -g2005 -o tb_ee.vvp  ../rtl/early_exit_ctrl.v tb_early_exit_ctrl.v && vvp tb_ee.vvp
iverilog -g2005 -o tb_axil.vvp ../rtl/early_exit_ctrl.v ../rtl/ee_axil.v tb_ee_axil.v && vvp tb_axil.vvp
