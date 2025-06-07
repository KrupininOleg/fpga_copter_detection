`timescale 1ps/100fs 

module top;
    localparam N_BITS_ONE_POINT = 8 * 3;
    localparam CAMERA_WIDTH_BUS_N_POINTS = 4;
    localparam CAMERA_WIDTH_BUS = N_BITS_ONE_POINT * CAMERA_WIDTH_BUS_N_POINTS;
    localparam STRAT_ADDRESS = 32'd0;

    reg sys_rst = 0;
    initial
      #1000 sys_rst = 1;
   
      ///////////////////////////////////////////////////////////////////////////////////////////
     //Тактовый генератор///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////    
  
    wire sys_clk;
    wire ref_clk;

    clk_generator u_clk_generator(
        .sys_clk(sys_clk),
        .ref_clk(ref_clk)
    );
     
      ///////////////////////////////////////////////////////////////////////////////////////////
     //Память///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    reg [27:0]  app_addr_reg;
    reg [2:0]   app_cmd_reg;
    reg         app_en_reg;
    reg [31:0]  app_wdf_data_reg;
    reg [3:0]   app_wdf_mask_reg;
    reg         app_wdf_wren_reg;
    reg         app_wdf_end_reg;
    wire [31:0] app_rd_data;
    wire        app_rd_data_end;
    wire        app_rd_data_valid;
    wire        app_rdy;
    wire        app_wdf_rdy;
    wire        init_calib_complete;
    wire        ui_clk;
    wire        ui_clk_sync_rst;

    ddr3_top u_ddr3_top (
        .app_addr              (app_addr_reg),
        .app_cmd               (app_cmd_reg),
        .app_en                (app_en_reg),
        .app_wdf_data          (app_wdf_data_reg),
        .app_wdf_mask          (app_wdf_mask_reg),
        .app_wdf_wren          (app_wdf_wren_reg),
        .app_wdf_end           (app_wdf_end_reg),
        .app_rd_data           (app_rd_data),
        .app_rd_data_end       (app_rd_data_end),
        .app_rd_data_valid     (app_rd_data_valid),
        .app_rdy               (app_rdy),
        .app_wdf_rdy           (app_wdf_rdy),
        .sys_clk_i             (sys_clk),
        .clk_ref_i             (ref_clk),
        .sys_rst               (sys_rst),
        .init_calib_complete   (init_calib_complete),
        .ui_clk                (ui_clk),
        .ui_clk_sync_rst       (ui_clk_sync_rst)
    );

    
    always @ (posedge ui_clk) begin
        if (ui_clk_sync_rst) begin
            app_en_reg <= 0;
            app_wdf_wren_reg <= 0;
        end 
    end
   
      ///////////////////////////////////////////////////////////////////////////////////////////
     //Камера///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////

    reg                           camera_recieve_ready_reg;
    wire                          camera_recieve_ready;
    wire                          camera_in_progress;
    wire [CAMERA_WIDTH_BUS - 1:0] camera_data;
    wire                          camera_data_valid;
    wire                          camera_frame_end;

    assign camera_recieve_ready = camera_recieve_ready_reg;

    camera #(
        .BUS_WIDTH (CAMERA_WIDTH_BUS)
    ) 
    u_camera (
        .clk           (sys_clk),
        .recieve_ready (camera_recieve_ready),
        .in_progress   (camera_in_progress),
        .data          (camera_data),
        .data_valid    (camera_data_valid),
        .frame_end     (camera_frame_end)
    );

      ///////////////////////////////////////////////////////////////////////////////////////////
     //Предобработка данных с камеры////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    localparam PREPROCESSING_READY                   = 4'd0;
    localparam PREPROCESSING_WAIT_CAMERA_IN_PROGRESS = 4'd1;
    localparam PREPROCESSING_WAIT_CAMERA_DATA        = 4'd2;
    localparam PREPROCESSING_WAIT                    = 4'd3;
    localparam PREPROCESSING_WRITE_RESULT            = 4'd4;
    localparam PREPROCESSING_WAIT_WRITING            = 4'd5;

    reg [3:0]  preprocessing_state = PREPROCESSING_READY;
    reg [31:0] cur_mem_addr = STRAT_ADDRESS;

    reg                                        preprocessor_input_en_reg;
    reg  [CAMERA_WIDTH_BUS - 1:0]              preprocessor_point_input_reg;
    wire [N_BITS_ONE_POINT - 1:0]              preprocessor_point_input [CAMERA_WIDTH_BUS_N_POINTS - 1:0];
    wire                                       preprocessor_input_en;
    wire [CAMERA_WIDTH_BUS_N_POINTS * 8 - 1:0] preprocessor_point_output;
    wire [CAMERA_WIDTH_BUS_N_POINTS - 1:0]     preprocessor_output_valid;

    assign preprocessor_input_en = preprocessor_input_en_reg;

    genvar i;
    integer j;

    generate 
        for (i = CAMERA_WIDTH_BUS_N_POINTS; i > 0; i = i - 1) begin:  gen_preprocessor_input
            assign preprocessor_point_input[i - 1] = preprocessor_point_input_reg[i * N_BITS_ONE_POINT - 1:(i - 1) * N_BITS_ONE_POINT];
        end
    endgenerate

    generate 
        for (i = CAMERA_WIDTH_BUS_N_POINTS - 1; i >= 0; i = i - 1) begin: gen_preprocessor_units
            frame_preprocessor u_frame_preprocessor(
                .clk                   (sys_clk),
                .point_input           (preprocessor_point_input[i]),
                .input_en              (preprocessor_input_en),
                .point_output          (preprocessor_point_output[(i + 1) * 8 - 1:i * 8]),
                .output_valid          (preprocessor_output_valid[i])
            );
        end
    endgenerate

    always @(sys_clk) begin
        case (preprocessing_state)
            PREPROCESSING_READY: begin
                if (init_calib_complete) begin // условие готовности следующих шагов
                    camera_recieve_ready_reg <= 1;
                    preprocessing_state <= PREPROCESSING_WAIT_CAMERA_IN_PROGRESS;
                end
            end
            PREPROCESSING_WAIT_CAMERA_IN_PROGRESS: begin
                if (camera_in_progress) begin
                    camera_recieve_ready_reg <= 0;
                    preprocessing_state <= PREPROCESSING_WAIT_CAMERA_DATA;
                end
            end
            PREPROCESSING_WAIT_CAMERA_DATA: begin
                if (camera_data_valid) begin
                    // TODO: Если данные с камер пришли, перепишим их в регистр,
                    //  и дадим камере передать следующие значения.
                    //  Для этого нужно разделить получение данных и препроцессинг
                    preprocessor_point_input_reg <= camera_data;
                    preprocessor_input_en_reg <= 1;
                    preprocessing_state <= PREPROCESSING_WAIT;
                end
            end
            PREPROCESSING_WAIT: begin
                // Все единицы
                if (&preprocessor_output_valid & preprocessor_input_en_reg) begin
                    preprocessor_input_en_reg <= 0;
                    preprocessing_state <= PREPROCESSING_WRITE_RESULT;
                    $display("%d %d %d %d", preprocessor_point_output[31:24], preprocessor_point_output[23:16], preprocessor_point_output[15:8], preprocessor_point_output[7:0]);
                    $display("preproc result %h", preprocessor_point_output);
                end
            end
            default;
        endcase
    end

    always @ (posedge ui_clk) begin
        case (preprocessing_state)
            PREPROCESSING_WRITE_RESULT: begin
                if (app_rdy & app_wdf_rdy) begin
                    app_cmd_reg = 3'b000; // Команда записать
                    app_addr_reg = cur_mem_addr; // Адрес
                    app_en_reg = 1; // Подтверждение адреса и команды
                    app_wdf_data_reg = preprocessor_point_output; // Данные
                    app_wdf_mask_reg = 0; // Маска данных
                    app_wdf_wren_reg = 1; // Подтверждение данных
                    app_wdf_end_reg = 1;
                    preprocessing_state = PREPROCESSING_WAIT_WRITING;
                end
            end
            PREPROCESSING_WAIT_WRITING: begin
                if (app_rdy & app_en_reg) begin
                    app_en_reg <= 0;
                end
                if (app_wdf_rdy & app_wdf_wren_reg) begin
                    app_wdf_wren_reg <= 0;
                end
                if (~app_en_reg & ~app_wdf_wren_reg) begin
                    $display("write %h to %h", app_wdf_data_reg, cur_mem_addr);
                    cur_mem_addr = cur_mem_addr + 8 * CAMERA_WIDTH_BUS_N_POINTS;
                    preprocessing_state = PREPROCESSING_READY;
                end
            end
            default;
        endcase
    end
endmodule