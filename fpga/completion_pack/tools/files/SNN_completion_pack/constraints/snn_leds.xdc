## -----------------------------------------------------------------------------
## snn_leds.xdc  -  PYNQ-Z2 (xc7z020clg400-1) LED pin constraints for the ee_axil LED outputs.
## Pin numbers cross-checked against several independent PYNQ-Z2 pin lists (LD0-LD3 and RGB LED4/LED5).
## ALSO compare them with the official PYNQ-Z2 master XDC from TUL / pynq.io before you build.
##
## The external ports in the Block Design MUST be named exactly:  led[3:0]  rgb0_g  rgb1_g  rgb1_r
## -----------------------------------------------------------------------------

## 4 user LEDs  LD0..LD3
set_property -dict { PACKAGE_PIN R14 IOSTANDARD LVCMOS33 } [get_ports { led[0] }]
set_property -dict { PACKAGE_PIN P14 IOSTANDARD LVCMOS33 } [get_ports { led[1] }]
set_property -dict { PACKAGE_PIN N16 IOSTANDARD LVCMOS33 } [get_ports { led[2] }]
set_property -dict { PACKAGE_PIN M14 IOSTANDARD LVCMOS33 } [get_ports { led[3] }]

## RGB LED 4 (LD4): green channel = "first output spike arrived"
set_property -dict { PACKAGE_PIN G17 IOSTANDARD LVCMOS33 } [get_ports { rgb0_g }]

## RGB LED 5 (LD5): green = early exit, red = ran the full number of timesteps
set_property -dict { PACKAGE_PIN L14 IOSTANDARD LVCMOS33 } [get_ports { rgb1_g }]
set_property -dict { PACKAGE_PIN M15 IOSTANDARD LVCMOS33 } [get_ports { rgb1_r }]
