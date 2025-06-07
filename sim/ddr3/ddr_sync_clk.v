`timescale 1ps/100fs 


module ddr3_top(
  input [27:0]  app_addr,
  input [2:0]   app_cmd,
  input         app_en,
  input [31:0]  app_wdf_data,
  input         app_wdf_end,
  input         app_wdf_wren,
  output [31:0] app_rd_data,
  output        app_rd_data_end,
  output        app_rd_data_valid,
  output        app_rdy,
  output        app_wdf_rdy,
  input [3:0]   app_wdf_mask,
  input         sys_clk_i,
  input         clk_ref_i,
  input         sys_rst,
  output        init_calib_complete,
  output        ui_clk,
  output        ui_clk_sync_rst
);
    

endmodule