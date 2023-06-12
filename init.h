/*
 * @Descripttion: all parameter init
 * @version:
 * @Author: Quikziii
 * @email: quikziii@gmail.com
 * @Date: 2023-02-23
 * @LastEditors: Quikziii
 * @LastEditTime: 2023-04-29
 */
#ifndef _INIT_H
#define _INIT_H


//Weight for each layer
struct Matrix **kernel_1; //amt 64/2 , channel 3 , size 3
//struct BN *bn_1; //channel 64/2

struct Matrix **kernel_2; //amt 64/2 , channel 64/2 , size 1
//struct BN *bn_2; //channel 64/2
struct Matrix **kernel_3; //amt 64/2 , channel 64/2 , size 3
//struct BN *bn_3; //channel 64/2
struct Matrix **kernel_4; //amt 256/2 , channel 64/2 , size 1
//struct BN *bn_4; //channel 256/2
struct Matrix **resKernel_4; //amt 256/2 , channel 64/2, size 1
//struct BN *resbn_4; //channel 256/2

struct Matrix **kernel_5; //amt 64/2 , channel 256/2 , size 1
//struct BN *bn_5; //channel 64/2
struct Matrix **kernel_6; //amt 64/2 , channel 64/2 , size 3
//struct BN *bn_6; //channel 64/2
struct Matrix **kernel_7; //amt 256/2 , channel 64/2 , size 1
//struct BN *bn_7; //channel 256/2

struct Matrix **kernel_8; //amt 64/2 , channel 256/2 , size 1
//struct BN *bn_8; //channel 64/2
struct Matrix **kernel_9; //amt 64/2 , channel 64/2 , size 3
//struct BN *bn_9; //channel 64/2
struct Matrix **kernel_10; //amt 256/2 , channel 64/2, size 1
//struct BN *bn_10; //channel 256/2

struct Matrix **kernel_11; //amt 128/2 , channel 256/2, size 1
//struct BN *bn_11; //channel 128/2
struct Matrix **kernel_12; //amt 128/2 , channel 128/2,  size 3
//struct BN *bn_12; //channel 128/2
struct Matrix **kernel_13; //amt 512/2 , channel 128/2, size 1
//struct BN *bn_13; //channel 512/2
struct Matrix **resKernel_13; // amt 512/2 , channel 256/2, size 1
//struct BN *resbn_13; //channel 512/2

struct Matrix **kernel_14; //amt 128/2 , channel 512/2 , size 1
//struct BN *bn_14; //channel 128/2
struct Matrix **kernel_15; // amt 128/2 , channel 128/2 ,size 3
//struct BN *bn_15; //channel 128/2
struct Matrix **kernel_16; //amt 512/2, channel 128/2 , size 1
//struct BN *bn_16; //channel 512/2

struct Matrix **kernel_17; //amt 128/2 , channel 512/2 ,size 1
//struct BN *bn_17; //channel 128/2
struct Matrix **kernel_18; // amt 128/2 , channel 128/2 ,size 3
//struct BN *bn_18; //channel 128/2
struct Matrix **kernel_19; // amt512/2, channel 128/2 ,size 1
//struct BN *bn_19; //channel 512/2

struct Matrix **kernel_20; //amt 128/2 , channel 512/2 ,size 1
//struct BN *bn_20; //channel 128/2
struct Matrix **kernel_21; // amt 128/2 , channel 128/2 ,size 3
//struct BN *bn_21; //channel 128/2
struct Matrix **kernel_22; // amt512/2, channel 128/2 ,size 1
//struct BN *bn_22; //channel 512/2
struct Matrix **resKernel_22; // amt 512/2 , channel 512/2 , size 1
//struct BN *resbn_22; // channel 512/2

struct Matrix **kernel_23; //amt 256/2 ,channel 512/2 ,size 1
//struct BN *bn_23; //channel 256/2
struct Matrix **kernel_24; // amt 256/2 , channel 256/2 , size 3
//struct BN *bn_24; // channel 256/2
struct Matrix **kernel_25; // amt 1024/2 , channel 256/2 , size 1
//struct BN *bn_25; // channel 1024/2
struct Matrix **resKernel_25; //amt 1024/2 , channel 512/2 , size 1
//struct BN *resbn_25; //channel 1024/2

struct Matrix **kernel_26; // amt 256/2 , channel 1024/2 , size 1
//struct BN *bn_26; // channel 256/2
struct Matrix **kernel_27; //amt 256/2 , channel 256/2 , size 3
//struct BN *bn_27; // channel 256/2
struct Matrix **kernel_28; //amt 1024/2 , channel 256/2 , size 1
//struct BN *bn_28; //channel 1024/2

struct Matrix **kernel_29; // amt 256/2 , channel 1024/2 ,size 1
//struct BN *bn_29; // channel 256/2
struct Matrix **kernel_30; // amt 256/2 , channel 256/2 , size 3
//struct BN *bn_30; // channel 256/2
struct Matrix **kernel_31; // amt 1024/2 , channel 256/2 , size 1
//struct BN *bn_31; //channel 1024/2

struct Matrix **kernel_32; // amt 256/2 , channel 1024/2 , size 1
//struct BN *bn_32; // channel 256/2
struct Matrix **kernel_33; // amt 256/2 , channel 256/2 , size 3
//struct BN *bn_33; //channel 256/2
struct Matrix **kernel_34; // amt 1024/2 , channel 256/2 , size 1
//struct BN *bn_34; //channel 1024/2

struct Matrix **kernel_35; //amt 256/2 , channel 1024/2 , size 1
//struct BN *bn_35; //channel 256/2
struct Matrix **kernel_36; // amt 256/2 , channel 256/2 , size 3
//struct BN *bn_36; //channel 256/2
struct Matrix **kernel_37; // amt 1024/2 , channel 256/2 , size 1
//struct BN *bn_37; // channel 1024/2

struct Matrix **kernel_38; //amt 256/2 , channel 1024/2 , size 1
//struct BN *bn_38; //channel 256/2
struct Matrix **kernel_39; //amt 256/2 , channel 256/2 , size 3
//struct BN *bn_39; //channel 256/2
struct Matrix **kernel_40; //amt 1024/2 , channel 256/2 , size 1
//struct BN *bn_40; //channel 1024/2
struct Matrix **resKernel_40; //amt 1024/2 , channel 1024/2 , size 1
//struct BN *resbn_40; //channel 1024/2

struct Matrix **kernel_41 ; //amt 512/2 , channel 1024/2 , size 1
//struct BN *bn_41 ; //channel 512/2
struct Matrix **kernel_42 ; //amt 512/2 , channel 512/2 , size 3
//struct BN *bn_42 ; //channel 512/2
struct Matrix **kernel_43 ; //amt 1024 , channel 512/2 , size 1
//struct BN *bn_43 ; //channel 1024
struct Matrix **resKernel_43 ; //amt 1024 , channel 1024/2 , size 1
//struct BN *resbn_43; //channel 1024

struct Matrix **kernel_44; //amt 512/2 , channel 1024 , size 1
//struct BN *bn_44; //channel 512/2
struct Matrix **kernel_45; //amt 512/2 , channel 512/2 , size 3
//struct BN *bn_45; // channel 512/2
struct Matrix **kernel_46; //amt 1024 , channel 512/2 , size 1
//struct BN *bn_46; // channel 1024

struct Matrix **kernel_47; // amt 512/2 , channel 1024 , size 1
//struct BN *bn_47; // channel 512/2
struct Matrix **kernel_48; // amt 512/2 , channel 512/2 , size 3
//struct BN *bn_48; //channel 512/2
struct Matrix **kernel_49; //amt 1024 , channel 512/2 , size 1
//struct BN *bn_49; // channel 1024
struct Matrix **resKernel_49; // amt 1024 , channel 1024 , size 1
struct Matrix *weightFC; //channel 1, row predictSize , column 1024
struct Matrix *bias; //channel 1 , row predictSize , size 1

//struct of the feature for frontpropagation
//layer 0
struct Matrix **feature_0;
//layer 1
struct Matrix **feature_CONV_1; //channel 64/2 , size 32
//struct Matrix **feature_BN_1; //channel 64/2 , size 32
struct Matrix **feature_ReLU_1; //channel 64/2 , size 32
struct Matrix **feature_2; //channel 64/2 , size 16
//bottleneck 1
//layer 2
//struct Matrix **feature_CONV_2; //channel 64/2 , size 16
struct Matrix **feature_3; //channel 64/2 , size 16
//layer 3
//struct Matrix **feature_CONV_3; //channel 64/2 , size 16
struct Matrix **feature_4; //channel 64/2 , size 16
//layer 4
struct Matrix **feature_CONV_4; //channel 256/2 , size 16
//struct Matrix **feature_BN_4; //channel 256/2 , size 16
struct Matrix **feature_ReLU_4; //channel 256/2 , size 16
//bottle neck 1 residual part
struct Matrix **feature_Res_4; //channel 256/2 , size 16
struct Matrix **feature_5; //channel 256/2 , size 16
//bottle neck 2
//layer 5
//struct Matrix **feature_CONV_5; //channel 64/2 , size 16
struct Matrix **feature_6; //channel 64/2 , size 16
//layer 6
//struct Matrix **feature_CONV_6; //channel 64/2 , size 16
struct Matrix **feature_7; //channel 64/2 , size 16
//layer 7
struct Matrix **feature_CONV_7; //channel 256/2 , size 16
//struct Matrix **feature_BN_7; //channel 256/2 , size 16
struct Matrix **feature_ReLU_7; //channel 256/2 , size 16
//bottle neck 2 residual part
struct Matrix **feature_8; //channel 256/2 , size 16
//bottle neck 3
//layer 8
//struct Matrix **feature_CONV_8; //channel 64/2 , size 16
struct Matrix **feature_9; //channel 64/2 , size 16
//layer 9
//struct Matrix **feature_CONV_9; //channel 64/2 , size 16
struct Matrix **feature_10; //channel 64/2 , size 16
//layer 10
struct Matrix **feature_CONV_10; //channel 256/2 , size 16
//struct Matrix **feature_BN_10; //channel 256/2 , size 16
struct Matrix **feature_ReLU_10; //channel 256/2 , size 16
//bottle neck 3 residual part
struct Matrix **feature_11; //channel 256/2 , size 16
//bottle neck 4
//layer 11
//struct Matrix **feature_CONV_11; //channel 128/2 , size 16
struct Matrix **feature_12; //channel 128/2 , size 16
//layer 12
//struct Matrix **feature_CONV_12; //channel 128/2 , size 16
struct Matrix **feature_13; //channel 128/2 , size 16
//layer 13
struct Matrix **feature_CONV_13; //channel 512/2 , size 16
//struct Matrix **feature_BN_13; //channel 512/2 , size 16
struct Matrix **feature_ReLU_13; //channel 512/2 , size 16
//bottle neck 4 residual part
struct Matrix **feature_Res_13; //channel 512/2 , size 16
struct Matrix **feature_14; //channel 512/2 , size 16
//bottle neck 5
//layer 14
//struct Matrix **feature_CONV_14; //channel 128/2 , size 16
struct Matrix **feature_15; //channel 128/2 , size 16
//layer 15
//struct Matrix **feature_CONV_15; //channel 128/2 , size 16
struct Matrix **feature_16; //channel 128/2 , size 16
//layer 16
struct Matrix **feature_CONV_16; //channel 512/2 , size 16
//struct Matrix **feature_BN_16; //channel 512/2 , size 16
struct Matrix **feature_ReLU_16; //channel 512/2 , size 16
//bottle neck 5 residual part
struct Matrix **feature_17; //channel 512/2 , size 16
//bottle neck 6
//layer 17
//struct Matrix **feature_CONV_17; //channel 128/2 , size 16
struct Matrix **feature_18; //channel 128/2 , size 16
//layer 18
//struct Matrix **feature_CONV_18; //channel 128/2 , size 16
struct Matrix **feature_19; //channel 128/2 , size 16
//layer 19
struct Matrix **feature_CONV_19; //channel 512/2 , size 16
//struct Matrix **feature_BN_19; //channel 512/2 , size 16
struct Matrix **feature_ReLU_19; //channel 512/2 , size 16
//bottle neck 6 residual part
struct Matrix **feature_20; //channel 512/2 , size 16
//bottle neck 7
//layer 20
//struct Matrix **feature_CONV_20; //channel 128/2 , size 16
struct Matrix **feature_21; //channel 128/2 , size 16
//layer 21
//struct Matrix **feature_CONV_21; //channel 128/2 , size 8
struct Matrix **feature_22; //channel 128/2 , size 8
//layer 22
struct Matrix **feature_CONV_22; //channel 512/2 , size 8
//struct Matrix **feature_BN_22; //channel 512/2 , size 8
struct Matrix **feature_ReLU_22; //channel 512/2 , size 8
//bottle neck 7 residual part
struct Matrix **feature_Res_22; //channel 512/2 , size 8
struct Matrix **feature_23; //channel 512/2 , size 8
//bottle neck 8
//layer 23
//struct Matrix **feature_CONV_23; //channel 256/2 , size 8
struct Matrix **feature_24; //channel 256/2 , size 8
//layer 24
//struct Matrix **feature_CONV_24; //channel 256/2 , size 8
struct Matrix **feature_25; //channel 256/2 , size 8
//layer 25
struct Matrix **feature_CONV_25; //channel 1024/2 , size 8
//struct Matrix **feature_BN_25; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_25; //channel 1024/2 , size 8
//bottle neck 8 residual part
struct Matrix **feature_Res_25; //channel 1024/2 , size 8
struct Matrix **feature_26; //channel 1024/2 , size 8
//bottle neck 9
//layer 26
//struct Matrix **feature_CONV_26; //channel 256/2 , size 8
struct Matrix **feature_27; //channel 256/2 , size 8
//layer 27
//struct Matrix **feature_CONV_27; //channel 256/2 , size 8
struct Matrix **feature_28; //channel 256/2 , size 8
//layer 28
struct Matrix **feature_CONV_28; //channel 1024/2 , size 8
//struct Matrix **feature_BN_28; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_28; //channel 1024/2 , size 8
//bottle neck 9 residual part
struct Matrix **feature_29; //channel 1024/2 , size 8
//bottle neck 10
//layer 29
//struct Matrix **feature_CONV_29; //channel 256/2 , size 8
struct Matrix **feature_30; //channel 256/2 , size 8
//layer 30
//struct Matrix **feature_CONV_30; //channel 256/2 , size 8
struct Matrix **feature_31; //channel 256/2 , size 8
//layer 31
struct Matrix **feature_CONV_31; //channel 1024/2 , size 8
//struct Matrix **feature_BN_31; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_31; //channel 1024/2 , size 8
//bottle neck 10 residual part
struct Matrix **feature_32; //channel 1024/2 , size 8
//bottle neck 11
//layer 32
//struct Matrix **feature_CONV_32; //channel 256/2 , size 8
struct Matrix **feature_33; //channel 256/2 , size 8
//layer 33
//struct Matrix **feature_CONV_33; //channel 256/2 , size 8
struct Matrix **feature_34; //channel 256/2 , size 8
//layer 34
struct Matrix **feature_CONV_34; //channel 1024/2 , size 8
//struct Matrix **feature_BN_34; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_34; //channel 1024/2 , size 8
//bottle neck 11 residual part
struct Matrix **feature_35; //channel 1024/2 , size 8
//bottle neck 12
//layer 35
//struct Matrix **feature_CONV_35; //channel 256/2 , size 8
struct Matrix **feature_36; //channel 256/2 , size 8
//layer 36
//struct Matrix **feature_CONV_36; //channel 256/2 , size 8
struct Matrix **feature_37; //channel 256/2 , size 8
//layer 37
struct Matrix **feature_CONV_37; //channel 1024/2 , size 8
//struct Matrix **feature_BN_37; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_37; //channel 1024/2 , size 8
//bottle neck 12 residual part
struct Matrix **feature_38; //channel 1024/2 , size 8
//bottle neck 13
//layer 38
//struct Matrix **feature_CONV_38; //channel 256/2 , size 8
struct Matrix **feature_39; //channel 256/2 , size 8
//layer 39
//struct Matrix **feature_CONV_39; //channel 256/2 , size 4
struct Matrix **feature_40; //channel 256/2 , size 4
//layer 40
struct Matrix **feature_CONV_40; //channel 1024/2 , size 4
//struct Matrix **feature_BN_40; //channel 1024/2 , size 4
struct Matrix **feature_ReLU_40; //channel 1024/2 , size 4
//bottle neck 13 residual part
struct Matrix **feature_Res_40; //channel 1024/2 , size 4
struct Matrix **feature_41; //channel 1024/2 , size 4
//bottle neck 14
//layer 41
//struct Matrix **feature_CONV_41; //channel 512/2 , size 4
struct Matrix **feature_42; //channel 512/2 , size 4
//layer 42
//struct Matrix **feature_CONV_42; //channel 512/2 , size 4
struct Matrix **feature_43; //channel 512/2 , size 4
//layer 43
struct Matrix **feature_CONV_43; //channel 1024 , size 4
//struct Matrix **feature_BN_43; //channel 1024 , size 4
struct Matrix **feature_ReLU_43; //channel 1024 , size 4
//bottle neck 14 residual part
struct Matrix **feature_Res_43; //channel 1024 , size 4
struct Matrix **feature_44; //channel 1024 , size 4
//bottle neck 15
//layer 44
//struct Matrix **feature_CONV_44; //channel 512/2 , size 4
struct Matrix **feature_45; //channel 512/2 , size 4
//layer 45
//struct Matrix **feature_CONV_45; //channel 512/2 , size 4
struct Matrix **feature_46; //channel 5 12 , size 4
//layer 46
struct Matrix **feature_CONV_46; //channel 1024 , size 4
//struct Matrix **feature_BN_46; //channel 1024 , size 4
struct Matrix **feature_ReLU_46; //channel 1024 , size 4
//bottle neck 15 residual part
struct Matrix **feature_47; //channel 1024 , size 4
//bottle neck 16
//layer 47
//struct Matrix **feature_CONV_47; //channel 512/2 , size 4
struct Matrix **feature_48; //channel 512/2 , size 4
//layer 48
//struct Matrix **feature_CONV_48; //channel 512/2 , size 2
struct Matrix **feature_49; //channel 512/2 , size 2
//layer 49
struct Matrix **feature_CONV_49; //channel 1024 , size 2
//struct Matrix **feature_BN_49; //channel 1024 , size 2
struct Matrix **feature_ReLU_49; //channel 1024 , size 2
//bottle neck 16 residual part
struct Matrix **feature_50; //channel 1024 , size 2
//global average
struct Matrix **feature_GlobalAvg; //channel 1 , row 1024 , column 1
//full connect
struct Matrix **feature_FullConnect; //channel 1  ,row predictsize , column 1
//softmax
struct Matrix **feature_SoftMax; //channel 1 , row predictSize , column 1
//predict ans
int *predictAns; //size Batch

//struct of the gradient for back propagation
//softmax
struct Matrix **gradient_CostFunction; //channel 1 row predictSize , column 1
//full connect
struct Matrix **gradient_FullConnect; //channel 1 row 1024 , column 1
struct Matrix *gradient_FullConnect_Weight; //channel 1 row predictSize column 1024)
struct Matrix *gradient_bias;//channel 1 row predictSize column 1
//global average
struct Matrix **gradient_GlobalAverage; //channel 1024 row 2 column 2
//bottle neck 16 residualpart
struct Matrix **gradient_Res_CONV_49;//channel 1024 size 4
struct Matrix **gradient_Res_kernel_49;// amt 1024 , channel 1024 , size 1
//layer 49
struct Matrix **gradient_ReLU_49; //channel 1024 , size 2
//struct Matrix **gradient_BN_49; //channel 1024 , size 2
//float *gradient_BN_Beta_49; //size 1024
//float *gradient_BN_Gamma_49; //size 1024
struct Matrix **gradient_49;//channel 512/2 size 2
struct Matrix **gradient_kernel_49;//amt 1024 channel 512/2 size 1
//layer 48
struct Matrix **gradient_ReLU_48; //channel 512/2 , size 2
//struct Matrix **gradient_BN_48; //channel 512/2 , size 2
//float *gradient_BN_Beta_48; //size 512/2
//float *gradient_BN_Gamma_48; //size 512/2
struct Matrix **gradient_48;//channel 512/2 size 4
struct Matrix **gradient_kernel_48;//amt 512/2 channel 512/2 size 3
//layer 47
struct Matrix **gradient_ReLU_47; //channel 512/2 , size 4
//struct Matrix **gradient_BN_47; //channel 512/2 , size 4
//float *gradient_BN_Beta_47; //size 512/2
//float *gradient_BN_Gamma_47; //size 512/2
struct Matrix **gradient_47;//channel 1024 size 4
struct Matrix **gradient_kernel_47;//amt 512/2 channel 1024 size 1
//layer 46
struct Matrix **gradient_ReLU_46; //channel 1024 , size 4
//struct Matrix **gradient_BN_46; //channel 1024 , size 4
//float *gradient_BN_Beta_46; //size 1024
//float *gradient_BN_Gamma_46; //size 1024
struct Matrix **gradient_46;//channel 512/2 size 4
struct Matrix **gradient_kernel_46;//amt 1024 channel 512/2 size 1
//layer 45
//struct Matrix **gradient_BN_45; //channel 512/2 , size 4
//float *gradient_BN_Beta_45; //size 512/2
//float *gradient_BN_Gamma_45; //size 512/2
struct Matrix **gradient_45;//channel 512/2 size 4
struct Matrix **gradient_kernel_45;//amt 512/2 channel 512/2 size 3
//layer 44
//struct Matrix **gradient_BN_44; //channel 512/2 , size 4
//float *gradient_BN_Beta_44; //size 512/2
//float *gradient_BN_Gamma_44; //size 512/2
struct Matrix **gradient_44;//channel 1024 size 4
struct Matrix **gradient_kernel_44;//amt 512/2 channel 1024 size 1
//bottle neck 14 residualpart
//struct Matrix **gradient_Res_BN_43;//channel 1024 size 4
//float *gradient_Res_BN_Beta_43;//size 1024
//float *gradient_Res_BN_Gamma_43;//size 1024
struct Matrix **gradient_Res_CONV_43;//channel 1024/2 size 4
struct Matrix **gradient_Res_kernel_43;//amt 1024 channel 1024/2 size 1
//layer 43
struct Matrix **gradient_ReLU_43; //channel 1024 , size 4
//struct Matrix **gradient_BN_43; //channel 1024 , size 4
//float *gradient_BN_Beta_43; //size 1024
//float *gradient_BN_Gamma_43; //size 1024
struct Matrix **gradient_43;//channel 512/2 size 4
struct Matrix **gradient_kernel_43;//amt 1024 channel 512/2 size 1
//layer 42
//struct Matrix **gradient_BN_42; //channel 512/2 , size 4
//float *gradient_BN_Beta_42; //size 512/2
//float *gradient_BN_Gamma_42; //size 512/2
struct Matrix **gradient_42;//channel 512/2 size 4
struct Matrix **gradient_kernel_42;//amt 512/2 channel 512/2 size 3
//layer 41
//struct Matrix **gradient_BN_41; //channel 512/2 , size 4
//float *gradient_BN_Beta_41; //size 512/2
//float *gradient_BN_Gamma_41; //size 512/2
struct Matrix **gradient_41;//channel 1024/2 size 4
struct Matrix **gradient_kernel_41;//amt 512/2 channel 1024/2 size 1
//bottle neck 13 residualpart
//struct Matrix **gradient_Res_BN_40;//channel 1024/2 size 4
//float *gradient_Res_BN_Beta_40;//size 1024/2
//float *gradient_Res_BN_Gamma_40;//size 1024/2
struct Matrix **gradient_Res_CONV_40;//channel 1024/2 size 8
struct Matrix **gradient_Res_kernel_40;//amt 1024/2 size 1
//layer 40
struct Matrix **gradient_ReLU_40; //channel 1024/2 , size 4
//struct Matrix **gradient_BN_40; //channel 1024/2 , size 4
//float *gradient_BN_Beta_40; //size 1024/2
//float *gradient_BN_Gamma_40; //size 1024/2
struct Matrix **gradient_40;//channel 256/2 size 4
struct Matrix **gradient_kernel_40;//amt 1024/2 channel 256/2 size 1
//layer 39
//struct Matrix **gradient_BN_39; //channel 256/2 , size 4
//float *gradient_BN_Beta_39; //size 256/2
//float *gradient_BN_Gamma_39; //size 256/2
struct Matrix **gradient_39;//channel 256/2 size 8
struct Matrix **gradient_kernel_39;//amt 256/2 channel 256/2 size 3
//layer 38
//struct Matrix **gradient_BN_38; //channel 256/2 , size 8
//float *gradient_BN_Beta_38; //size 256/2
//float *gradient_BN_Gamma_38; //size 256/2
struct Matrix **gradient_38;//channel 1024/2 size 8
struct Matrix **gradient_kernel_38;//amt 256/2 channel 1024/2 size 1
//layer 37
struct Matrix **gradient_ReLU_37; //channel 1024/2 , size 8
//struct Matrix **gradient_BN_37; //channel 1024/2 , size 8
//float *gradient_BN_Beta_37; //size 1024/2
//float *gradient_//Weight for each layer
struct Matrix **kernel_1; //amt 64/2 , channel 3 , size 3
struct BN *bn_1; //channel 64/2

struct Matrix **kernel_2; //amt 64/2 , channel 64/2 , size 1
struct BN *bn_2; //channel 64/2
struct Matrix **kernel_3; //amt 64/2 , channel 64/2 , size 3
struct BN *bn_3; //channel 64/2
struct Matrix **kernel_4; //amt 256/2 , channel 64/2 , size 1
struct BN *bn_4; //channel 256/2
struct Matrix **resKernel_4; //amt 256/2 , channel 64/2, size 1
struct BN *resbn_4; //channel 256/2

struct Matrix **kernel_5; //amt 64/2 , channel 256/2 , size 1
struct BN *bn_5; //channel 64/2
struct Matrix **kernel_6; //amt 64/2 , channel 64/2 , size 3
struct BN *bn_6; //channel 64/2
struct Matrix **kernel_7; //amt 256/2 , channel 64/2 , size 1
struct BN *bn_7; //channel 256/2

struct Matrix **kernel_8; //amt 64/2 , channel 256/2 , size 1
struct BN *bn_8; //channel 64/2
struct Matrix **kernel_9; //amt 64/2 , channel 64/2 , size 3
struct BN *bn_9; //channel 64/2
struct Matrix **kernel_10; //amt 256/2 , channel 64/2, size 1
struct BN *bn_10; //channel 256/2

struct Matrix **kernel_11; //amt 128/2 , channel 256/2, size 1
struct BN *bn_11; //channel 128/2
struct Matrix **kernel_12; //amt 128/2 , channel 128/2,  size 3
struct BN *bn_12; //channel 128/2
struct Matrix **kernel_13; //amt 512/2 , channel 128/2, size 1
struct BN *bn_13; //channel 512/2
struct Matrix **resKernel_13; // amt 512/2 , channel 256/2, size 1
struct BN *resbn_13; //channel 512/2

struct Matrix **kernel_14; //amt 128/2 , channel 512/2 , size 1
struct BN *bn_14; //channel 128/2
struct Matrix **kernel_15; // amt 128/2 , channel 128/2 ,size 3
struct BN *bn_15; //channel 128/2
struct Matrix **kernel_16; //amt 512/2, channel 128/2 , size 1
struct BN *bn_16; //channel 512/2

struct Matrix **kernel_17; //amt 128/2 , channel 512/2 ,size 1
struct BN *bn_17; //channel 128/2
struct Matrix **kernel_18; // amt 128/2 , channel 128/2 ,size 3
struct BN *bn_18; //channel 128/2
struct Matrix **kernel_19; // amt512/2, channel 128/2 ,size 1
struct BN *bn_19; //channel 512/2

struct Matrix **kernel_20; //amt 128/2 , channel 512/2 ,size 1
struct BN *bn_20; //channel 128/2
struct Matrix **kernel_21; // amt 128/2 , channel 128/2 ,size 3
struct BN *bn_21; //channel 128/2
struct Matrix **kernel_22; // amt512/2, channel 128/2 ,size 1
struct BN *bn_22; //channel 512/2
struct Matrix **resKernel_22; // amt 512/2 , channel 512/2 , size 1
struct BN *resbn_22; // channel 512/2

struct Matrix **kernel_23; //amt 256/2 ,channel 512/2 ,size 1
struct BN *bn_23; //channel 256/2
struct Matrix **kernel_24; // amt 256/2 , channel 256/2 , size 3
struct BN *bn_24; // channel 256/2
struct Matrix **kernel_25; // amt 1024/2 , channel 256/2 , size 1
struct BN *bn_25; // channel 1024/2
struct Matrix **resKernel_25; //amt 1024/2 , channel 512/2 , size 1
struct BN *resbn_25; //channel 1024/2

struct Matrix **kernel_26; // amt 256/2 , channel 1024/2 , size 1
struct BN *bn_26; // channel 256/2
struct Matrix **kernel_27; //amt 256/2 , channel 256/2 , size 3
struct BN *bn_27; // channel 256/2
struct Matrix **kernel_28; //amt 1024/2 , channel 256/2 , size 1
struct BN *bn_28; //channel 1024/2

struct Matrix **kernel_29; // amt 256/2 , channel 1024/2 ,size 1
struct BN *bn_29; // channel 256/2
struct Matrix **kernel_30; // amt 256/2 , channel 256/2 , size 3
struct BN *bn_30; // channel 256/2
struct Matrix **kernel_31; // amt 1024/2 , channel 256/2 , size 1
struct BN *bn_31; //channel 1024/2

struct Matrix **kernel_32; // amt 256/2 , channel 1024/2 , size 1
struct BN *bn_32; // channel 256/2
struct Matrix **kernel_33; // amt 256/2 , channel 256/2 , size 3
struct BN *bn_33; //channel 256/2
struct Matrix **kernel_34; // amt 1024/2 , channel 256/2 , size 1
struct BN *bn_34; //channel 1024/2

struct Matrix **kernel_35; //amt 256/2 , channel 1024/2 , size 1
struct BN *bn_35; //channel 256/2
struct Matrix **kernel_36; // amt 256/2 , channel 256/2 , size 3
struct BN *bn_36; //channel 256/2
struct Matrix **kernel_37; // amt 1024/2 , channel 256/2 , size 1
struct BN *bn_37; // channel 1024/2

struct Matrix **kernel_38; //amt 256/2 , channel 1024/2 , size 1
struct BN *bn_38; //channel 256/2
struct Matrix **kernel_39; //amt 256/2 , channel 256/2 , size 3
struct BN *bn_39; //channel 256/2
struct Matrix **kernel_40; //amt 1024/2 , channel 256/2 , size 1
struct BN *bn_40; //channel 1024/2
struct Matrix **resKernel_40; //amt 1024/2 , channel 1024/2 , size 1
struct BN *resbn_40; //channel 1024/2

struct Matrix **kernel_41 ; //amt 512/2 , channel 1024/2 , size 1
struct BN *bn_41 ; //channel 512/2
struct Matrix **kernel_42 ; //amt 512/2 , channel 512/2 , size 3
struct BN *bn_42 ; //channel 512/2
struct Matrix **kernel_43 ; //amt 1024 , channel 512/2 , size 1
struct BN *bn_43 ; //channel 1024
struct Matrix **resKernel_43 ; //amt 1024 , channel 1024/2 , size 1
struct BN *resbn_43; //channel 1024

struct Matrix **kernel_44; //amt 512/2 , channel 1024 , size 1
struct BN *bn_44; //channel 512/2
struct Matrix **kernel_45; //amt 512/2 , channel 512/2 , size 3
struct BN *bn_45; // channel 512/2
struct Matrix **kernel_46; //amt 1024 , channel 512/2 , size 1
struct BN *bn_46; // channel 1024

struct Matrix **kernel_47; // amt 512/2 , channel 1024 , size 1
struct BN *bn_47; // channel 512/2
struct Matrix **kernel_48; // amt 512/2 , channel 512/2 , size 3
struct BN *bn_48; //channel 512/2
struct Matrix **kernel_49; //amt 1024 , channel 512/2 , size 1
struct BN *bn_49; // channel 1024
struct Matrix **resKernel_49; // amt 1024 , channel 1024 , size 1
struct Matrix *weightFC; //channel 1, row predictSize , column 1024
struct Matrix *bias; //channel 1 , row predictSize , size 1

//struct of the feature for frontpropagation
//layer 0
struct Matrix **feature_0;
//layer 1
struct Matrix **feature_CONV_1; //channel 64/2 , size 32
struct Matrix **feature_BN_1; //channel 64/2 , size 32
struct Matrix **feature_ReLU_1; //channel 64/2 , size 16
struct Matrix **feature_2; //channel 64/2 , size 16
//bottleneck 1
//layer 2
struct Matrix **feature_CONV_2; //channel 64/2 , size 16
struct Matrix **feature_3; //channel 64/2 , size 16
//layer 3
struct Matrix **feature_CONV_3; //channel 64/2 , size 16
struct Matrix **feature_4; //channel 64/2 , size 16
//layer 4
struct Matrix **feature_CONV_4; //channel 256/2 , size 16
struct Matrix **feature_BN_4; //channel 256/2 , size 16
struct Matrix **feature_ReLU_4; //channel 256/2 , size 16
//bottle neck 1 residual part
struct Matrix **feature_Res_4; //channel 256/2 , size 16
struct Matrix **feature_5; //channel 256/2 , size 16
//bottle neck 2
//layer 5
struct Matrix **feature_CONV_5; //channel 64/2 , size 16
struct Matrix **feature_6; //channel 64/2 , size 16
//layer 6
struct Matrix **feature_CONV_6; //channel 64/2 , size 16
struct Matrix **feature_7; //channel 64/2 , size 16
//layer 7
struct Matrix **feature_CONV_7; //channel 256/2 , size 16
struct Matrix **feature_BN_7; //channel 256/2 , size 16
struct Matrix **feature_ReLU_7; //channel 256/2 , size 16
//bottle neck 2 residual part
struct Matrix **feature_8; //channel 256/2 , size 16
//bottle neck 3
//layer 8
struct Matrix **feature_CONV_8; //channel 64/2 , size 16
struct Matrix **feature_9; //channel 64/2 , size 16
//layer 9
struct Matrix **feature_CONV_9; //channel 64/2 , size 16
struct Matrix **feature_10; //channel 64/2 , size 16
//layer 10
struct Matrix **feature_CONV_10; //channel 256/2 , size 16
struct Matrix **feature_BN_10; //channel 256/2 , size 16
struct Matrix **feature_ReLU_10; //channel 256/2 , size 16
//bottle neck 3 residual part
struct Matrix **feature_11; //channel 256/2 , size 16
//bottle neck 4
//layer 11
struct Matrix **feature_CONV_11; //channel 128/2 , size 16
struct Matrix **feature_12; //channel 128/2 , size 16
//layer 12
struct Matrix **feature_CONV_12; //channel 128/2 , size 16
struct Matrix **feature_13; //channel 128/2 , size 16
//layer 13
struct Matrix **feature_CONV_13; //channel 512/2 , size 16
struct Matrix **feature_BN_13; //channel 512/2 , size 16
struct Matrix **feature_ReLU_13; //channel 512/2 , size 16
//bottle neck 4 residual part
struct Matrix **feature_Res_13; //channel 512/2 , size 16
struct Matrix **feature_14; //channel 512/2 , size 16
//bottle neck 5
//layer 14
struct Matrix **feature_CONV_14; //channel 128/2 , size 16
struct Matrix **feature_15; //channel 128/2 , size 16
//layer 15
struct Matrix **feature_CONV_15; //channel 128/2 , size 16
struct Matrix **feature_16; //channel 128/2 , size 16
//layer 16
struct Matrix **feature_CONV_16; //channel 512/2 , size 16
struct Matrix **feature_BN_16; //channel 512/2 , size 16
struct Matrix **feature_ReLU_16; //channel 512/2 , size 16
//bottle neck 5 residual part
struct Matrix **feature_17; //channel 512/2 , size 16
//bottle neck 6
//layer 17
struct Matrix **feature_CONV_17; //channel 128/2 , size 16
struct Matrix **feature_18; //channel 128/2 , size 16
//layer 18
struct Matrix **feature_CONV_18; //channel 128/2 , size 16
struct Matrix **feature_19; //channel 128/2 , size 16
//layer 19
struct Matrix **feature_CONV_19; //channel 512/2 , size 16
struct Matrix **feature_BN_19; //channel 512/2 , size 16
struct Matrix **feature_ReLU_19; //channel 512/2 , size 16
//bottle neck 6 residual part
struct Matrix **feature_20; //channel 512/2 , size 16
//bottle neck 7
//layer 20
struct Matrix **feature_CONV_20; //channel 128/2 , size 16
struct Matrix **feature_21; //channel 128/2 , size 16
//layer 21
struct Matrix **feature_CONV_21; //channel 128/2 , size 8
struct Matrix **feature_22; //channel 128/2 , size 8
//layer 22
struct Matrix **feature_CONV_22; //channel 512/2 , size 8
struct Matrix **feature_BN_22; //channel 512/2 , size 8
struct Matrix **feature_ReLU_22; //channel 512/2 , size 8
//bottle neck 7 residual part
struct Matrix **feature_Res_22; //channel 512/2 , size 8
struct Matrix **feature_23; //channel 512/2 , size 8
//bottle neck 8
//layer 23
struct Matrix **feature_CONV_23; //channel 256/2 , size 8
struct Matrix **feature_24; //channel 256/2 , size 8
//layer 24
struct Matrix **feature_CONV_24; //channel 256/2 , size 8
struct Matrix **feature_25; //channel 256/2 , size 8
//layer 25
struct Matrix **feature_CONV_25; //channel 1024/2 , size 8
struct Matrix **feature_BN_25; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_25; //channel 1024/2 , size 8
//bottle neck 8 residual part
struct Matrix **feature_Res_25; //channel 1024/2 , size 8
struct Matrix **feature_26; //channel 1024/2 , size 8
//bottle neck 9
//layer 26
struct Matrix **feature_CONV_26; //channel 256/2 , size 8
struct Matrix **feature_27; //channel 256/2 , size 8
//layer 27
struct Matrix **feature_CONV_27; //channel 256/2 , size 8
struct Matrix **feature_28; //channel 256/2 , size 8
//layer 28
struct Matrix **feature_CONV_28; //channel 1024/2 , size 8
struct Matrix **feature_BN_28; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_28; //channel 1024/2 , size 8
//bottle neck 9 residual part
struct Matrix **feature_29; //channel 1024/2 , size 8
//bottle neck 10
//layer 29
struct Matrix **feature_CONV_29; //channel 256/2 , size 8
struct Matrix **feature_30; //channel 256/2 , size 8
//layer 30
struct Matrix **feature_CONV_30; //channel 256/2 , size 8
struct Matrix **feature_31; //channel 256/2 , size 8
//layer 31
struct Matrix **feature_CONV_31; //channel 1024/2 , size 8
struct Matrix **feature_BN_31; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_31; //channel 1024/2 , size 8
//bottle neck 10 residual part
struct Matrix **feature_32; //channel 1024/2 , size 8
//bottle neck 11
//layer 32
struct Matrix **feature_CONV_32; //channel 256/2 , size 8
struct Matrix **feature_33; //channel 256/2 , size 8
//layer 33
struct Matrix **feature_CONV_33; //channel 256/2 , size 8
struct Matrix **feature_34; //channel 256/2 , size 8
//layer 34
struct Matrix **feature_CONV_34; //channel 1024/2 , size 8
struct Matrix **feature_BN_34; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_34; //channel 1024/2 , size 8
//bottle neck 11 residual part
struct Matrix **feature_35; //channel 1024/2 , size 8
//bottle neck 12
//layer 35
struct Matrix **feature_CONV_35; //channel 256/2 , size 8
struct Matrix **feature_36; //channel 256/2 , size 8
//layer 36
struct Matrix **feature_CONV_36; //channel 256/2 , size 8
struct Matrix **feature_37; //channel 256/2 , size 8
//layer 37
struct Matrix **feature_CONV_37; //channel 1024/2 , size 8
struct Matrix **feature_BN_37; //channel 1024/2 , size 8
struct Matrix **feature_ReLU_37; //channel 1024/2 , size 8
//bottle neck 12 residual part
struct Matrix **feature_38; //channel 1024/2 , size 8
//bottle neck 13
//layer 38
struct Matrix **feature_CONV_38; //channel 256/2 , size 8
struct Matrix **feature_39; //channel 256/2 , size 8
//layer 39
struct Matrix **feature_CONV_39; //channel 256/2 , size 4
struct Matrix **feature_40; //channel 256/2 , size 4
//layer 40
struct Matrix **feature_CONV_40; //channel 1024/2 , size 4
struct Matrix **feature_BN_40; //channel 1024/2 , size 4
struct Matrix **feature_ReLU_40; //channel 1024/2 , size 4
//bottle neck 13 residual part
struct Matrix **feature_Res_40; //channel 1024/2 , size 4
struct Matrix **feature_41; //channel 1024/2 , size 4
//bottle neck 14
//layer 41
struct Matrix **feature_CONV_41; //channel 512/2 , size 4
struct Matrix **feature_42; //channel 512/2 , size 4
//layer 42
struct Matrix **feature_CONV_42; //channel 512/2 , size 4
struct Matrix **feature_43; //channel 512/2 , size 4
//layer 43
struct Matrix **feature_CONV_43; //channel 1024 , size 4
struct Matrix **feature_BN_43; //channel 1024 , size 4
struct Matrix **feature_ReLU_43; //channel 1024 , size 4
//bottle neck 14 residual part
struct Matrix **feature_Res_43; //channel 1024 , size 4
struct Matrix **feature_44; //channel 1024 , size 4
//bottle neck 15
//layer 44
struct Matrix **feature_CONV_44; //channel 512/2 , size 4
struct Matrix **feature_45; //channel 512/2 , size 4
//layer 45
struct Matrix **feature_CONV_45; //channel 512/2 , size 4
struct Matrix **feature_46; //channel 5 12 , size 4
//layer 46
struct Matrix **feature_CONV_46; //channel 1024 , size 4
struct Matrix **feature_BN_46; //channel 1024 , size 4
struct Matrix **feature_ReLU_46; //channel 1024 , size 4
//bottle neck 15 residual part
struct Matrix **feature_47; //channel 1024 , size 4
//bottle neck 16
//layer 47
struct Matrix **feature_CONV_47; //channel 512/2 , size 4
struct Matrix **feature_48; //channel 512/2 , size 4
//layer 48
struct Matrix **feature_CONV_48; //channel 512/2 , size 2
struct Matrix **feature_49; //channel 512/2 , size 2
//layer 49
struct Matrix **feature_CONV_49; //channel 1024 , size 2
struct Matrix **feature_BN_49; //channel 1024 , size 2
struct Matrix **feature_ReLU_49; //channel 1024 , size 2
//bottle neck 16 residual part
struct Matrix **feature_50; //channel 1024 , size 2
//global average
struct Matrix **feature_GlobalAvg; //channel 1 , row 1024 , column 1
//full connect
struct Matrix **feature_FullConnect; //channel 1  ,row predictsize , column 1
//softmax
struct Matrix **feature_SoftMax; //channel 1 , row predictSize , column 1
//predict ans
int *predictAns; //size Batch

//struct of the gradient for back propagation
//softmax
struct Matrix **gradient_CostFunction; //channel 1 row predictSize , column 1
//full connect
struct Matrix **gradient_FullConnect; //channel 1 row 1024 , column 1
struct Matrix *gradient_FullConnect_Weight; //channel 1 row predictSize column 1024)
struct Matrix *gradient_bias;//channel 1 row predictSize column 1
//global average
struct Matrix **gradient_GlobalAverage; //channel 1024 row 2 column 2
//bottle neck 16 residualpart
struct Matrix **gradient_Res_CONV_49;//channel 1024 size 4
struct Matrix **gradient_Res_kernel_49;// amt 1024 , channel 1024 , size 1
//layer 49
struct Matrix **gradient_ReLU_49; //channel 1024 , size 2
struct Matrix **gradient_BN_49; //channel 1024 , size 2
float *gradient_BN_Beta_49; //size 1024
float *gradient_BN_Gamma_49; //size 1024
struct Matrix **gradient_49;//channel 512/2 size 2
struct Matrix **gradient_kernel_49;//amt 1024 channel 512/2 size 1
//layer 48
struct Matrix **gradient_ReLU_48; //channel 512/2 , size 2
struct Matrix **gradient_BN_48; //channel 512/2 , size 2
float *gradient_BN_Beta_48; //size 512/2
float *gradient_BN_Gamma_48; //size 512/2
struct Matrix **gradient_48;//channel 512/2 size 4
struct Matrix **gradient_kernel_48;//amt 512/2 channel 512/2 size 3
//layer 47
struct Matrix **gradient_ReLU_47; //channel 512/2 , size 4
struct Matrix **gradient_BN_47; //channel 512/2 , size 4
float *gradient_BN_Beta_47; //size 512/2
float *gradient_BN_Gamma_47; //size 512/2
struct Matrix **gradient_47;//channel 1024 size 4
struct Matrix **gradient_kernel_47;//amt 512/2 channel 1024 size 1
//layer 46
struct Matrix **gradient_ReLU_46; //channel 1024 , size 4
struct Matrix **gradient_BN_46; //channel 1024 , size 4
float *gradient_BN_Beta_46; //size 1024
float *gradient_BN_Gamma_46; //size 1024
struct Matrix **gradient_46;//channel 512/2 size 4
struct Matrix **gradient_kernel_46;//amt 1024 channel 512/2 size 1
//layer 45
struct Matrix **gradient_BN_45; //channel 512/2 , size 4
float *gradient_BN_Beta_45; //size 512/2
float *gradient_BN_Gamma_45; //size 512/2
struct Matrix **gradient_45;//channel 512/2 size 4
struct Matrix **gradient_kernel_45;//amt 512/2 channel 512/2 size 3
//layer 44
struct Matrix **gradient_BN_44; //channel 512/2 , size 4
float *gradient_BN_Beta_44; //size 512/2
float *gradient_BN_Gamma_44; //size 512/2
struct Matrix **gradient_44;//channel 1024 size 4
struct Matrix **gradient_kernel_44;//amt 512/2 channel 1024 size 1
//bottle neck 14 residualpart
struct Matrix **gradient_Res_BN_43;//channel 1024 size 4
float *gradient_Res_BN_Beta_43;//size 1024
float *gradient_Res_BN_Gamma_43;//size 1024
struct Matrix **gradient_Res_CONV_43;//channel 1024/2 size 4
struct Matrix **gradient_Res_kernel_43;//amt 1024 channel 1024/2 size 1
//layer 43
struct Matrix **gradient_ReLU_43; //channel 1024 , size 4
struct Matrix **gradient_BN_43; //channel 1024 , size 4
float *gradient_BN_Beta_43; //size 1024
float *gradient_BN_Gamma_43; //size 1024
struct Matrix **gradient_43;//channel 512/2 size 4
struct Matrix **gradient_kernel_43;//amt 1024 channel 512/2 size 1
//layer 42
struct Matrix **gradient_BN_42; //channel 512/2 , size 4
float *gradient_BN_Beta_42; //size 512/2
float *gradient_BN_Gamma_42; //size 512/2
struct Matrix **gradient_42;//channel 512/2 size 4
struct Matrix **gradient_kernel_42;//amt 512/2 channel 512/2 size 3
//layer 41
struct Matrix **gradient_BN_41; //channel 512/2 , size 4
float *gradient_BN_Beta_41; //size 512/2
float *gradient_BN_Gamma_41; //size 512/2
struct Matrix **gradient_41;//channel 1024/2 size 4
struct Matrix **gradient_kernel_41;//amt 512/2 channel 1024/2 size 1
//bottle neck 13 residualpart
struct Matrix **gradient_Res_BN_40;//channel 1024/2 size 4
float *gradient_Res_BN_Beta_40;//size 1024/2
float *gradient_Res_BN_Gamma_40;//size 1024/2
struct Matrix **gradient_Res_CONV_40;//channel 1024/2 size 8
struct Matrix **gradient_Res_kernel_40;//amt 1024/2 size 1
//layer 40
struct Matrix **gradient_ReLU_40; //channel 1024/2 , size 4
struct Matrix **gradient_BN_40; //channel 1024/2 , size 4
float *gradient_BN_Beta_40; //size 1024/2
float *gradient_BN_Gamma_40; //size 1024/2
struct Matrix **gradient_40;//channel 256/2 size 4
struct Matrix **gradient_kernel_40;//amt 1024/2 channel 256/2 size 1
//layer 39
struct Matrix **gradient_BN_39; //channel 256/2 , size 4
float *gradient_BN_Beta_39; //size 256/2
float *gradient_BN_Gamma_39; //size 256/2
struct Matrix **gradient_39;//channel 256/2 size 8
struct Matrix **gradient_kernel_39;//amt 256/2 channel 256/2 size 3
//layer 38
struct Matrix **gradient_BN_38; //channel 256/2 , size 8
float *gradient_BN_Beta_38; //size 256/2
float *gradient_BN_Gamma_38; //size 256/2
struct Matrix **gradient_38;//channel 1024/2 size 8
struct Matrix **gradient_kernel_38;//amt 256/2 channel 1024/2 size 1
//layer 37
struct Matrix **gradient_ReLU_37; //channel 1024/2 , size 8
struct Matrix **gradient_BN_37; //channel 1024/2 , size 8
float *gradient_BN_Beta_37; //size 1024/2
float *gradient_BN_Gamma_37; //size 1024/2
struct Matrix **gradient_37;//channel 256/2 size 8
struct Matrix **gradient_kernel_37;//amt 1024/2 channel 256/2 size 1
//layer 36
struct Matrix **gradient_BN_36; //channel 256/2 , size 8
float *gradient_BN_Beta_36; //size 256/2
float *gradient_BN_Gamma_36; //size 256/2
struct Matrix **gradient_36;//channel 256/2 size 8
struct Matrix **gradient_kernel_36;//amt 256/2 channel 256/2 size 3
//layer 35
struct Matrix **gradient_BN_35; //channel 256/2 , size 8
float *gradient_BN_Beta_35; //size 256/2
float *gradient_BN_Gamma_35; //size 256/2
struct Matrix **gradient_35;//channel 1024/2 size 8
struct Matrix **gradient_kernel_35;//amt 256/2 channel 1024/2 size 1
//layer 34
struct Matrix **gradient_ReLU_34; //channel 1024/2 , size 8
struct Matrix **gradient_BN_34; //channel 1024/2 , size 8
float *gradient_BN_Beta_34; //size 1024/2
float *gradient_BN_Gamma_34; //size 1024/2
struct Matrix **gradient_34;//channel 256/2 size 8
struct Matrix **gradient_kernel_34;//amt 1024/2 channel 256/2 size 1
//layer 33
struct Matrix **gradient_BN_33; //channel 256/2 , size 8
float *gradient_BN_Beta_33; //size 256/2
float *gradient_BN_Gamma_33; //size 256/2
struct Matrix **gradient_33;//channel 256/2 size 8
struct Matrix **gradient_kernel_33;//amt 256/2 channel 256/2 size 3
//layer 32
struct Matrix **gradient_BN_32; //channel 256/2 , size 8
float *gradient_BN_Beta_32; //size 256/2
float *gradient_BN_Gamma_32; //size 256/2
struct Matrix **gradient_32;//channel 1024/2 size 8
struct Matrix **gradient_kernel_32;//amt 256/2 channel 1024/2 size 1
//layer 31
struct Matrix **gradient_ReLU_31; //channel 1024/2 , size 8
struct Matrix **gradient_BN_31; //channel 1024/2 , size 8
float *gradient_BN_Beta_31; //size 1024/2
float *gradient_BN_Gamma_31; //size 1024/2
struct Matrix **gradient_31;//channel 256/2 size 8
struct Matrix **gradient_kernel_31;//amt 1024/2 channel 256/2 size 1
//layer 30
struct Matrix **gradient_BN_30; //channel 256/2 , size 8
float *gradient_BN_Beta_30; //size 256/2
float *gradient_BN_Gamma_30; //size 256/2
struct Matrix **gradient_30;//channel 256/2 size 8
struct Matrix **gradient_kernel_30;//amt 256/2 channel 256/2 size 3
//layer 29
struct Matrix **gradient_BN_29; //channel 256/2 , size 8
float *gradient_BN_Beta_29; //size 256/2
float *gradient_BN_Gamma_29; //size 256/2
struct Matrix **gradient_29;//channel 1024/2 size 8
struct Matrix **gradient_kernel_29;//amt 256/2 channel 1024/2 size 1
//layer 28
struct Matrix **gradient_ReLU_28; //channel 1024/2 , size 8
struct Matrix **gradient_BN_28; //channel 1024/2 , size 8
float *gradient_BN_Beta_28; //size 1024/2
float *gradient_BN_Gamma_28; //size 1024/2
struct Matrix **gradient_28;//channel 256/2 size 8
struct Matrix **gradient_kernel_28;//amt 1024/2 channel 256/2 size 1
//layer 27
struct Matrix **gradient_BN_27; //channel 256/2 , size 8
float *gradient_BN_Beta_27; //size 256/2
float *gradient_BN_Gamma_27; //size 256/2
struct Matrix **gradient_27;//channel 256/2 size 8
struct Matrix **gradient_kernel_27;//amt 256/2 channel 256/2 size 3
//layer 26
struct Matrix **gradient_BN_26; //channel 256/2 , size 8
float *gradient_BN_Beta_26; //size 256/2
float *gradient_BN_Gamma_26; //size 256/2
struct Matrix **gradient_26;//channel 1024/2 size 8
struct Matrix **gradient_kernel_26;//amt 256/2 channel 1024/2 size 1
//bottle neck 8 residualpart
struct Matrix **gradient_Res_BN_25;//channel 1024/2 size 8
float *gradient_Res_BN_Beta_25;//size 1024/2
float *gradient_Res_BN_Gamma_25;//size 1024/2
struct Matrix **gradient_Res_CONV_25;//channel 512/2 size 8
struct Matrix **gradient_Res_kernel_25;//amt 1024/2 channel 512/2 size 1
//layer 25
struct Matrix **gradient_ReLU_25; //channel 1024/2 , size 8
struct Matrix **gradient_BN_25; //channel 1024/2 , size 8
float *gradient_BN_Beta_25; //size 1024/2
float *gradient_BN_Gamma_25; //size 1024/2
struct Matrix **gradient_25;//channel 256/2 size 8
struct Matrix **gradient_kernel_25;//amt 1024/2 channel 256/2 size 1
//layer 24
struct Matrix **gradient_BN_24; //channel 256/2 , size 8
float *gradient_BN_Beta_24; //size 256/2
float *gradient_BN_Gamma_24; //size 256/2
struct Matrix **gradient_24;//channel 256/2 size 8
struct Matrix **gradient_kernel_24;//amt 256/2 channel 256/2 size 3
//layer 23
struct Matrix **gradient_BN_23; //channel 256/2 , size 8
float *gradient_BN_Beta_23; //size 256/2
float *gradient_BN_Gamma_23; //size 256/2
struct Matrix **gradient_23;//channel 512/2 size 8
struct Matrix **gradient_kernel_23;//amt 256/2 channel 512/2 size 1
//bottle neck 7 residualpart
struct Matrix **gradient_Res_BN_22;//channel 512/2 size 8
float *gradient_Res_BN_Beta_22;//size 512/2
float *gradient_Res_BN_Gamma_22;//size 512/2
struct Matrix **gradient_Res_CONV_22;//channel 512/2 size 16
struct Matrix **gradient_Res_kernel_22;//amt 512/2 channel 512/2 size 1
//layer 22
struct Matrix **gradient_ReLU_22; //channel 512/2 , size 8
struct Matrix **gradient_BN_22; //channel 512/2 , size 8
float *gradient_BN_Beta_22; //size 512/2
float *gradient_BN_Gamma_22; //size 512/2
struct Matrix **gradient_22;//channel 128/2 size 8
struct Matrix **gradient_kernel_22;//amt 512/2 channel 128/2 size 1
//layer 21
struct Matrix **gradient_BN_21; //channel 128/2 , size 8
float *gradient_BN_Beta_21; //size 128/2
float *gradient_BN_Gamma_21; //size 128/2
struct Matrix **gradient_21;//channel 128/2 size 16
struct Matrix **gradient_kernel_21;//amt 128/2 channel 128/2 size 3
//layer 20
struct Matrix **gradient_BN_20; //channel 128/2 , size 16
float *gradient_BN_Beta_20; //size 128/2
float *gradient_BN_Gamma_20; //size 128/2
struct Matrix **gradient_20;//channel 512/2 size 16
struct Matrix **gradient_kernel_20;//amt 128/2 channel 512/2 size 1
//layer 19
struct Matrix **gradient_ReLU_19; //channel 512/2 , size 16
struct Matrix **gradient_BN_19; //channel 512/2 , size 16
float *gradient_BN_Beta_19; //size 512/2
float *gradient_BN_Gamma_19; //size 512/2
struct Matrix **gradient_19;//channel 128/2 size 16
struct Matrix **gradient_kernel_19;//amt 512/2 channel 128/2 size 1
//layer 18
struct Matrix **gradient_BN_18; //channel 128/2 , size 16
float *gradient_BN_Beta_18; //size 128/2
float *gradient_BN_Gamma_18; //size 128/2
struct Matrix **gradient_18;//channel 128/2 size 16
struct Matrix **gradient_kernel_18;//amt 128/2 channel 128/2 size 3
//layer 17
struct Matrix **gradient_BN_17; //channel 128/2 , size 16
float *gradient_BN_Beta_17; //size 128/2
float *gradient_BN_Gamma_17; //size 128/2
struct Matrix **gradient_17;//channel 512/2 size 16
struct Matrix **gradient_kernel_17;//amt 128/2 channel 512/2 size 1
//layer 16
struct Matrix **gradient_ReLU_16; //channel 512/2 , size 16
struct Matrix **gradient_BN_16; //channel 512/2 , size 16
float *gradient_BN_Beta_16; //size 512/2
float *gradient_BN_Gamma_16; //size 512/2
struct Matrix **gradient_16;//channel 128/2 size 16
struct Matrix **gradient_kernel_16;//amt 512/2 channel 128/2 size 1
//layer 15
struct Matrix **gradient_BN_15; //channel 128/2 , size 16
float *gradient_BN_Beta_15; //size 128/2
float *gradient_BN_Gamma_15; //size 128/2
struct Matrix **gradient_15;//channel 128/2 size 16
struct Matrix **gradient_kernel_15;//amt 128/2 channel 128/2 size 3
//layer 14
struct Matrix **gradient_BN_14; //channel 128/2 , size 16
float *gradient_BN_Beta_14; //size 128/2
float *gradient_BN_Gamma_14; //size 128/2
struct Matrix **gradient_14;//channel 512/2 size 16
struct Matrix **gradient_kernel_14;//amt 128/2 channel 512/2 size 1
//bottle neck 4 residualpart
struct Matrix **gradient_Res_BN_13;//channel 512/2 size 16
float *gradient_Res_BN_Beta_13;//size 512/2
float *gradient_Res_BN_Gamma_13;//size 512/2
struct Matrix **gradient_Res_CONV_13;//channel 256/2 size 16
struct Matrix **gradient_Res_kernel_13;//amt 512/2 channel 256/2 size 1
//layer 13
struct Matrix **gradient_ReLU_13; //channel 512/2 , size 16
struct Matrix **gradient_BN_13; //channel 512/2 , size 16
float *gradient_BN_Beta_13; //size 512/2
float *gradient_BN_Gamma_13; //size 512/2
struct Matrix **gradient_13;//channel 128/2 size 16
struct Matrix **gradient_kernel_13;//amt 512/2 channel 128/2 size 1
//layer 12
struct Matrix **gradient_BN_12; //channel 128/2 , size 16
float *gradient_BN_Beta_12; //size 128/2
float *gradient_BN_Gamma_12; //size 128/2
struct Matrix **gradient_12;//channel 128/2 size 16
struct Matrix **gradient_kernel_12;//amt 128/2 channel 128/2 size 3
//layer 11
struct Matrix **gradient_BN_11; //channel 128/2 , size 16
float *gradient_BN_Beta_11; //size 128/2
float *gradient_BN_Gamma_11; //size 128/2
struct Matrix **gradient_11;//channel 256/2 size 16
struct Matrix **gradient_kernel_11;//amt 128/2 channel 256/2 size 1
//layer 10
struct Matrix **gradient_ReLU_10; //channel 256/2 , size 16
struct Matrix **gradient_BN_10; //channel 256/2 , size 16
float *gradient_BN_Beta_10; //size 256/2
float *gradient_BN_Gamma_10; //size 256/2
struct Matrix **gradient_10;//channel 64/2 size 16
struct Matrix **gradient_kernel_10;//amt 256/2 channel 64/2 size 1
//layer 9
struct Matrix **gradient_BN_9; //channel 64/2 , size 16
float *gradient_BN_Beta_9; //size 64/2
float *gradient_BN_Gamma_9; //size 64/2
struct Matrix **gradient_9;//channel 64/2 size 16
struct Matrix **gradient_kernel_9;//amt 64/2 channel 64/2 size 3
//layer 8
struct Matrix **gradient_BN_8; //channel 64/2 , size 16
float *gradient_BN_Beta_8; //size 64/2
float *gradient_BN_Gamma_8; //size 64/2
struct Matrix **gradient_8;//channel 256/2 size 16
struct Matrix **gradient_kernel_8;//amt 64/2 channel 256/2 size 1
//layer 7
struct Matrix **gradient_ReLU_7; //channel 256/2 , size 16
struct Matrix **gradient_BN_7; //channel 256/2 , size 16
float *gradient_BN_Beta_7; //size 256/2
float *gradient_BN_Gamma_7; //size 256/2
struct Matrix **gradient_7;//channel 64/2 size 16
struct Matrix **gradient_kernel_7;//amt 256/2 channel 64/2 size 1
//layer 6
struct Matrix **gradient_BN_6; //channel 64/2 , size 16
float *gradient_BN_Beta_6; //size 64/2
float *gradient_BN_Gamma_6; //size 64/2
struct Matrix **gradient_6;//channel 64/2 size 16
struct Matrix **gradient_kernel_6;//amt 64/2 channel 64/2 size 3
//layer 5
struct Matrix **gradient_BN_5; //channel 64/2 , size 16
float *gradient_BN_Beta_5; //size 64/2
float *gradient_BN_Gamma_5; //size 64/2
struct Matrix **gradient_5;//channel 256/2 size 16
struct Matrix **gradient_kernel_5;//amt 64/2 channel 256/2 size 1
//bottle neck 1 residualpart
struct Matrix **gradient_Res_BN_4;//channel 256/2 size 16
float *gradient_Res_BN_Beta_4;//size 256/2
float *gradient_Res_BN_Gamma_4;//size 256/2
struct Matrix **gradient_Res_CONV_4;//channel 64/2 size 16
struct Matrix **gradient_Res_kernel_4;//amt 256/2 channel 64/2 size 1
//layer 4
struct Matrix **gradient_ReLU_4; //channel 256/2 , size 16
struct Matrix **gradient_BN_4; //channel 256/2 , size 16
float *gradient_BN_Beta_4; //size 256/2
float *gradient_BN_Gamma_4; //size 256/2
struct Matrix **gradient_4;//channel 64/2 size 16
struct Matrix **gradient_kernel_4;//amt 256/2 channel 64/2 size 1
//layer 3
struct Matrix **gradient_BN_3; //channel 64/2 , size 16
float *gradient_BN_Beta_3; //size 64/2
float *gradient_BN_Gamma_3; //size 64/2
struct Matrix **gradient_3;//channel 64/2 size 16
struct Matrix **gradient_kernel_3;//amt 64/2 channel 64/2 size 3
//layer 2
struct Matrix **gradient_BN_2; //channel 64/2 , size 16
float *gradient_BN_Beta_2; //size 64/2
float *gradient_BN_Gamma_2; //size 64/2
struct Matrix **gradient_2;//channel 64/2/2 size 16
struct Matrix **gradient_kernel_2;//amt 64/2 channel 64/2 size 1
//layer 1
struct Matrix **gradient_MaxPooling;//channel 64/2 size 32
struct Matrix **gradient_ReLU_1; //channel 64/2 , size 32
struct Matrix **gradient_BN_1; //channel 64/2 , size 32
float *gradient_BN_Beta_1; //size 64/2
float *gradient_BN_Gamma_1; //size 64/2
struct Matrix **gradient_1;//channel 3 size 32
struct Matrix **gradient_kernel_1;//amt 64/2 channel 3 size 3



/**
 * @name: Kernel_Init
 * @msg: init the gradient for both batch norm in backpropagation
 * @param {int}size          - (input)size of gradient
 * @return {float *}the gradient which already to be init
 */
float *Gradient_Init(int size);

/**
 * @name: Kernel_Init
 * @msg: init the kernel for both frontpropagation
 * @param {int}size          - (input)size of batch
 * @param {int}channel       - (input)channel of both kernel
 * @param {int}row           - (input)row size of both kernel
 * @param {int}column        - (input)column size of both kernel
 * @return {struct Matrix **}the kernel list which already to be init
 */
struct Matrix **Kernel_Init(int size , int channel , int row , int column);

/**
 * @name: Feature_Init
 * @msg: init the feature for both frontpropagation
 * @param {int}channel       - (input)channel of both kernel
 * @param {int}row           - (input)row size of both kernel
 * @param {int}column        - (input)column size of both kernel
 * @return {struct Matrix **}the kernel list which already to be init
 */
struct Matrix **Feature_Init(int channel , int row , int column);


/**
 * @name: Feature_Copy
 * @msg: copy the feature for frontpropagation
 * @param {struct Matrix **}which store the pointer of feature     - (output)feature_0
 * @param {int}index                                               - (input)the index should be get for learn
 * @param {struct Matrix **}input                                   -(input)which store all the input
 * @return {}
 */
void Feature_Copy(struct Matrix **feature,int index,struct Matrix **input);

/**
 * @name: Init_Weight
 * @msg: initial all the kernel for frontpropagation
 * @return {}
 */
void Init_Weight();

/**
 * @name: Load_Weight
 * @msg: Load the trained weight
 * @return {}
 */
void Load_Weight();

/**
 * @name: Init_Feature
 * @msg: initial all the feature for frontpropagation
 * @return {}
 */
void Init_Feature();

/**
 * @name: Init_Gradient
 * @msg: initial all the gradient for backpropagation
 * @return {}
 */
void Init_Gradient();

#endif