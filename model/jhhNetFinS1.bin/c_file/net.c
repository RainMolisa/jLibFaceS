#include "cnn_conv2d.h"
#include "cnn_pool2d.h"
#include "cnn_tensor_function.h"
/* @brief cnn_run_a_lovely_net() - We transfrom the prototxt file to C code as below, you can copy the code to complete the whole calculation of your network
/*        feature occupies 640000 bytes
 * @param feature_addr_offset: the address where input and output of all layers are put
 * @param weight_addr_offset: the address where weights and biases all layers are put
 * @param layer_end: if you want to stop computation after N layers finished, please set layer_end to N, otherwise, please set layer_end to 0
 */
void cnn_run_a_lovely_net(uint32_t feature_addr_offset, uint32_t weight_addr_offset, uint16_t layer_end) {
    #define RETURN_JUDGE { if(layer_index++ == layer_end) return; }
    uint16_t layer_index = 1;
    struct cnn_conv2D_config_bean conv2d_bean;
    struct cnn_pool2D_config_bean pool2d_bean;
    // layer index: 0, the 1st layer is of index 0
    // layer name: Conv_0
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 1 to Round 1
    // input tensor names are input.1
    // output tensor names are 147
    // input tensor occupy 25600 bytes individually
    // output tensor occupy 76800 bytes individually
    // weight occupies 32 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 1; // how many channels of input
    conv2d_bean.in_height = 160; // height of input
    conv2d_bean.in_width = 160; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 3; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 0; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 0; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x0 + feature_addr_offset, 0x6400 + feature_addr_offset, 0x0 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 1
    // layer name: Conv_1
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 2 to Round 2
    // input tensor names are 147
    // output tensor names are 150
    // input tensor occupy 76800 bytes individually
    // output tensor occupy 122880 bytes individually
    // weight occupies 480 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 3; // how many channels of input
    conv2d_bean.in_height = 160; // height of input
    conv2d_bean.in_width = 160; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 16; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 2;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x6400 + feature_addr_offset, 0x19000 + feature_addr_offset, 0x100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 2
    // layer name: Conv_3
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 3 to Round 3
    // input tensor names are 150
    // output tensor names are 151
    // input tensor occupy 122880 bytes individually
    // output tensor occupy 122880 bytes individually
    // weight occupies 288 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 16; // how many channels of input
    conv2d_bean.in_height = 80; // height of input
    conv2d_bean.in_width = 80; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 16; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x19000 + feature_addr_offset, 0x37000 + feature_addr_offset, 0x300 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 3
    // layer name: Conv_4
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 4 to Round 4
    // input tensor names are 151
    // output tensor names are 154
    // input tensor occupy 122880 bytes individually
    // output tensor occupy 122880 bytes individually
    // weight occupies 512 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 16; // how many channels of input
    conv2d_bean.in_height = 80; // height of input
    conv2d_bean.in_width = 80; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 16; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 16; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x37000 + feature_addr_offset, 0x55000 + feature_addr_offset, 0x500 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 4
    // layer name: MaxPool_6
    // layer type: Pooling
    // this layer is completed by using CNN engine from Round 5 to Round 5
    // input tensor names are 154
    // output tensor names are 155
    // input tensor occupy 122880 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    pool2d_bean.in_channel = 16; // how many channels of input
    pool2d_bean.in_height = 80; // height of input
    pool2d_bean.in_width = 80; // output of input
    pool2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    pool2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    pool2d_bean.kernel_size_h = 2; // size of the convolving kernel
    pool2d_bean.kernel_size_w = 2; // size of the convolving kernel
    pool2d_bean.stride = 2;
    pool2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    pool2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    pool2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    pool2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    pool2d_bean.input_signed = 1; // whether input is signed
    pool2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    pool2d_bean.count_include_pad = 0; // when non-0, include the zero-pad in the averaging calculation
    pool2d_bean.ceil_mode = 0; // when non-0, use ceil instead of floor to compute the output shape.
    pool2d_bean.avg = 0; // non-zero: avg pooling, zero: max pooling
    pool2d_bean.out_shift = 0; // the fraction of output minus the fraction of input, you can set this to a positive number for a higher precision, only valid in avgpooling
    pool2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    pool2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    pool2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    pool2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    cnn_pool2D(0x55000 + feature_addr_offset, 0x6400 + feature_addr_offset, &pool2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 5
    // layer name: Conv_7
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 6 to Round 6
    // input tensor names are 155
    // output tensor names are 156
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 288 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 16; // how many channels of input
    conv2d_bean.in_height = 40; // height of input
    conv2d_bean.in_width = 40; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 16; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x6400 + feature_addr_offset, 0x10400 + feature_addr_offset, 0x700 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 6
    // layer name: Conv_8
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 7 to Round 7
    // input tensor names are 156
    // output tensor names are 159
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 512 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 16; // how many channels of input
    conv2d_bean.in_height = 40; // height of input
    conv2d_bean.in_width = 40; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 16; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 16; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x10400 + feature_addr_offset, 0x1a400 + feature_addr_offset, 0x900 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 7
    // layer name: Conv_10
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 8 to Round 8
    // input tensor names are 159
    // output tensor names are 160
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 163840 bytes individually
    // weight occupies 1152 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 16; // how many channels of input
    conv2d_bean.in_height = 40; // height of input
    conv2d_bean.in_width = 40; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x1a400 + feature_addr_offset, 0x24400 + feature_addr_offset, 0xb00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 8
    // layer name: Conv_11
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 9 to Round 9
    // input tensor names are 160
    // output tensor names are 163
    // input tensor occupy 163840 bytes individually
    // output tensor occupy 163840 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 40; // height of input
    conv2d_bean.in_width = 40; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x24400 + feature_addr_offset, 0x4c400 + feature_addr_offset, 0x1000 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 9
    // layer name: Conv_13
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 10 to Round 10
    // input tensor names are 163
    // output tensor names are 164
    // input tensor occupy 163840 bytes individually
    // output tensor occupy 163840 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 40; // height of input
    conv2d_bean.in_width = 40; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x4c400 + feature_addr_offset, 0x74400 + feature_addr_offset, 0x1800 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 10
    // layer name: Conv_14
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 11 to Round 11
    // input tensor names are 164
    // output tensor names are 167
    // input tensor occupy 163840 bytes individually
    // output tensor occupy 163840 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 40; // height of input
    conv2d_bean.in_width = 40; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x74400 + feature_addr_offset, 0x6400 + feature_addr_offset, 0x2900 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 11
    // layer name: Conv_16
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 12 to Round 12
    // input tensor names are 167
    // output tensor names are 168
    // input tensor occupy 163840 bytes individually
    // output tensor occupy 163840 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 40; // height of input
    conv2d_bean.in_width = 40; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x6400 + feature_addr_offset, 0x2e400 + feature_addr_offset, 0x3100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 12
    // layer name: Conv_17
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 13 to Round 13
    // input tensor names are 168
    // output tensor names are 171
    // input tensor occupy 163840 bytes individually
    // output tensor occupy 163840 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 40; // height of input
    conv2d_bean.in_width = 40; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x2e400 + feature_addr_offset, 0x56400 + feature_addr_offset, 0x4200 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 13
    // layer name: MaxPool_19
    // layer type: Pooling
    // this layer is completed by using CNN engine from Round 14 to Round 14
    // input tensor names are 171
    // output tensor names are 172
    // input tensor occupy 163840 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    pool2d_bean.in_channel = 64; // how many channels of input
    pool2d_bean.in_height = 40; // height of input
    pool2d_bean.in_width = 40; // output of input
    pool2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    pool2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    pool2d_bean.kernel_size_h = 2; // size of the convolving kernel
    pool2d_bean.kernel_size_w = 2; // size of the convolving kernel
    pool2d_bean.stride = 2;
    pool2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    pool2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    pool2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    pool2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    pool2d_bean.input_signed = 1; // whether input is signed
    pool2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    pool2d_bean.count_include_pad = 0; // when non-0, include the zero-pad in the averaging calculation
    pool2d_bean.ceil_mode = 0; // when non-0, use ceil instead of floor to compute the output shape.
    pool2d_bean.avg = 0; // non-zero: avg pooling, zero: max pooling
    pool2d_bean.out_shift = 0; // the fraction of output minus the fraction of input, you can set this to a positive number for a higher precision, only valid in avgpooling
    pool2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    pool2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    pool2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    pool2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    cnn_pool2D(0x56400 + feature_addr_offset, 0, &pool2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 14
    // layer name: Conv_20
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 15 to Round 15
    // input tensor names are 172
    // output tensor names are 173
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 20; // height of input
    conv2d_bean.in_width = 20; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x4a00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 15
    // layer name: Conv_21
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 16 to Round 16
    // input tensor names are 173
    // output tensor names are 176
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 20; // height of input
    conv2d_bean.in_width = 20; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x5b00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 16
    // layer name: Conv_23
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 17 to Round 17
    // input tensor names are 176
    // output tensor names are 177
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 20; // height of input
    conv2d_bean.in_width = 20; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x6300 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 17
    // layer name: Conv_24
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 18 to Round 18
    // input tensor names are 177
    // output tensor names are 180
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 20; // height of input
    conv2d_bean.in_width = 20; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0x6400 + feature_addr_offset, 0x7400 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 18
    // layer name: MaxPool_26
    // layer type: Pooling
    // this layer is completed by using CNN engine from Round 19 to Round 19
    // input tensor names are 180
    // output tensor names are 181
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 20480 bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    pool2d_bean.in_channel = 64; // how many channels of input
    pool2d_bean.in_height = 20; // height of input
    pool2d_bean.in_width = 20; // output of input
    pool2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    pool2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    pool2d_bean.kernel_size_h = 2; // size of the convolving kernel
    pool2d_bean.kernel_size_w = 2; // size of the convolving kernel
    pool2d_bean.stride = 2;
    pool2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    pool2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    pool2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    pool2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    pool2d_bean.input_signed = 1; // whether input is signed
    pool2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    pool2d_bean.count_include_pad = 0; // when non-0, include the zero-pad in the averaging calculation
    pool2d_bean.ceil_mode = 0; // when non-0, use ceil instead of floor to compute the output shape.
    pool2d_bean.avg = 0; // non-zero: avg pooling, zero: max pooling
    pool2d_bean.out_shift = 0; // the fraction of output minus the fraction of input, you can set this to a positive number for a higher precision, only valid in avgpooling
    pool2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    pool2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    pool2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    pool2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    cnn_pool2D(0x6400 + feature_addr_offset, 0, &pool2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 19
    // layer name: Conv_27
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 20 to Round 20
    // input tensor names are 181
    // output tensor names are 182
    // input tensor occupy 20480 bytes individually
    // output tensor occupy 20480 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 10; // height of input
    conv2d_bean.in_width = 10; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x7c00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 20
    // layer name: Conv_28
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 21 to Round 21
    // input tensor names are 182
    // output tensor names are 185
    // input tensor occupy 20480 bytes individually
    // output tensor occupy 20480 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 10; // height of input
    conv2d_bean.in_width = 10; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x8d00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 21
    // layer name: Conv_30
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 22 to Round 22
    // input tensor names are 185
    // output tensor names are 186
    // input tensor occupy 20480 bytes individually
    // output tensor occupy 20480 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 10; // height of input
    conv2d_bean.in_width = 10; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x9500 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 22
    // layer name: Conv_31
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 23 to Round 23
    // input tensor names are 186
    // output tensor names are 189
    // input tensor occupy 20480 bytes individually
    // output tensor occupy 20480 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 10; // height of input
    conv2d_bean.in_width = 10; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0x10400 + feature_addr_offset, 0xa600 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 23
    // layer name: MaxPool_33
    // layer type: Pooling
    // this layer is completed by using CNN engine from Round 24 to Round 24
    // input tensor names are 189
    // output tensor names are 190
    // input tensor occupy 20480 bytes individually
    // output tensor occupy 10240 bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    pool2d_bean.in_channel = 64; // how many channels of input
    pool2d_bean.in_height = 10; // height of input
    pool2d_bean.in_width = 10; // output of input
    pool2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    pool2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    pool2d_bean.kernel_size_h = 2; // size of the convolving kernel
    pool2d_bean.kernel_size_w = 2; // size of the convolving kernel
    pool2d_bean.stride = 2;
    pool2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    pool2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    pool2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    pool2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    pool2d_bean.input_signed = 1; // whether input is signed
    pool2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    pool2d_bean.count_include_pad = 0; // when non-0, include the zero-pad in the averaging calculation
    pool2d_bean.ceil_mode = 0; // when non-0, use ceil instead of floor to compute the output shape.
    pool2d_bean.avg = 0; // non-zero: avg pooling, zero: max pooling
    pool2d_bean.out_shift = 0; // the fraction of output minus the fraction of input, you can set this to a positive number for a higher precision, only valid in avgpooling
    pool2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    pool2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    pool2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    pool2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    cnn_pool2D(0x10400 + feature_addr_offset, 0, &pool2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 24
    // layer name: Conv_34
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 25 to Round 25
    // input tensor names are 190
    // output tensor names are 191
    // input tensor occupy 10240 bytes individually
    // output tensor occupy 10240 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 5; // height of input
    conv2d_bean.in_width = 5; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0xae00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 25
    // layer name: Conv_35
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 26 to Round 26
    // input tensor names are 191
    // output tensor names are 194
    // input tensor occupy 10240 bytes individually
    // output tensor occupy 10240 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 5; // height of input
    conv2d_bean.in_width = 5; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0xbf00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 26
    // layer name: Conv_37
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 27 to Round 27
    // input tensor names are 194
    // output tensor names are 195
    // input tensor occupy 10240 bytes individually
    // output tensor occupy 10240 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 5; // height of input
    conv2d_bean.in_width = 5; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0xc700 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 27
    // layer name: Conv_38
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 28 to Round 28
    // input tensor names are 195
    // output tensor names are 198
    // input tensor occupy 10240 bytes individually
    // output tensor occupy 10240 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 5; // height of input
    conv2d_bean.in_width = 5; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0xd800 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 28
    // layer name: Conv_50
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 29 to Round 29
    // input tensor names are 198
    // output tensor names are 211
    // input tensor occupy 10240 bytes individually
    // output tensor occupy 10240 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 5; // height of input
    conv2d_bean.in_width = 5; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0xe000 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 29
    // layer name: Conv_51
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 30 to Round 30
    // input tensor names are 211
    // output tensor names are 214
    // input tensor occupy 10240 bytes individually
    // output tensor occupy 10240 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 5; // height of input
    conv2d_bean.in_width = 5; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0xf100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 30
    // layer name: Conv_53
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 31 to Round 31
    // input tensor names are 214
    // output tensor names are 215
    // input tensor occupy 10240 bytes individually
    // output tensor occupy 5440 bytes individually
    // weight occupies 3168 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 5; // height of input
    conv2d_bean.in_width = 5; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 34; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0xf900 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 31
    // layer name: Conv_54
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 32 to Round 32
    // input tensor names are 215
    // output tensor names are 216
    // input tensor occupy 5440 bytes individually
    // output tensor occupy 5440 bytes individually
    // weight occupies 1088 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 34; // how many channels of input
    conv2d_bean.in_height = 5; // height of input
    conv2d_bean.in_width = 5; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 34; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 34; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0x15400 + feature_addr_offset, 0x10600 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 32
    // layer name: Conv_45
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 33 to Round 33
    // input tensor names are 189
    // output tensor names are 205
    // input tensor occupy 20480 bytes individually
    // output tensor occupy 20480 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 10; // height of input
    conv2d_bean.in_width = 10; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 1; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 1; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 1; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x10400 + feature_addr_offset, 0, 0x10b00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 33
    // layer name: Conv_46
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 34 to Round 34
    // input tensor names are 205
    // output tensor names are 208
    // input tensor occupy 20480 bytes individually
    // output tensor occupy 20480 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 10; // height of input
    conv2d_bean.in_width = 10; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 1; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 1; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x11c00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 34
    // layer name: Conv_48
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 35 to Round 35
    // input tensor names are 208
    // output tensor names are 209
    // input tensor occupy 20480 bytes individually
    // output tensor occupy 10880 bytes individually
    // weight occupies 3168 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 10; // height of input
    conv2d_bean.in_width = 10; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 34; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 1; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 1; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 1; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 1; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x12400 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 35
    // layer name: Conv_49
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 36 to Round 36
    // input tensor names are 209
    // output tensor names are 210
    // input tensor occupy 10880 bytes individually
    // output tensor occupy 10880 bytes individually
    // weight occupies 1088 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 34; // how many channels of input
    conv2d_bean.in_height = 10; // height of input
    conv2d_bean.in_width = 10; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 34; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 34; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 1; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0x10400 + feature_addr_offset, 0x13100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 36
    // layer name: Conv_40
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 37 to Round 37
    // input tensor names are 180
    // output tensor names are 199
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 20; // height of input
    conv2d_bean.in_width = 20; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x6400 + feature_addr_offset, 0, 0x13600 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 37
    // layer name: Conv_41
    // layer type: Convolution, ReLU
    // this layer is completed by using CNN engine from Round 38 to Round 38
    // input tensor names are 199
    // output tensor names are 202
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 40960 bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 20; // height of input
    conv2d_bean.in_width = 20; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x14700 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 38
    // layer name: Conv_43
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 39 to Round 39
    // input tensor names are 202
    // output tensor names are 203
    // input tensor occupy 40960 bytes individually
    // output tensor occupy 32640 bytes individually
    // weight occupies 4224 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 20; // height of input
    conv2d_bean.in_width = 20; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 51; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 1; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0, 0x14f00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 39
    // layer name: Conv_44
    // layer type: Convolution
    // this layer is completed by using CNN engine from Round 40 to Round 40
    // input tensor names are 203
    // output tensor names are 204
    // input tensor occupy 32640 bytes individually
    // output tensor occupy 32640 bytes individually
    // weight occupies 1632 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 51; // how many channels of input
    conv2d_bean.in_height = 20; // height of input
    conv2d_bean.in_width = 20; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 51; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 51; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 1; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 1; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0, 0x6400 + feature_addr_offset, 0x16000 + weight_addr_offset, &conv2d_bean);
    /*
     * this layer end
     */
    // total_weight_size: 0x16700
}
