#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"input.h"
#include"matrix.h"
#include"batchnorm.h"
#include"frontpropagation.h"
#include"backpropagation.h"
#include"init.h"
#include"learning.h"
#include"multi.h"

int flag = 0 ;

float FrontPropagation(int index)
{
    //printf("frontpropagation start\n");
    //layer 0
    Feature_Copy(feature_0,index,input);
    //layer 1
    Multi_Convolution(feature_0,kernel_1,feature_CONV_1,1,1);
    BN_GetCoeff(feature_CONV_1,bn_1);
    BNBatch_BatchNorm(feature_CONV_1,feature_BN_1,bn_1);
    FrontBatch_ReLU(feature_BN_1,feature_ReLU_1);
    Multi_MaxPooLing(feature_ReLU_1,feature_2,2,0,2);
    //printf("layer 1 done\n");
    //bottleneck 1
    //layer 2
    Multi_Convolution(feature_2,kernel_2,feature_CONV_2,0,1);
    BN_GetCoeff(feature_CONV_2,bn_2);
    BNBatch_BatchNorm(feature_CONV_2,feature_3,bn_2);
    //printf("layer 2 done\n");
    //layer 3
    Multi_Convolution(feature_3,kernel_3,feature_CONV_3,1,1);
    BN_GetCoeff(feature_CONV_3,bn_3);
    BNBatch_BatchNorm(feature_CONV_3,feature_4,bn_3);
    //printf("layer 3 done\n");
    //layer 4
    Multi_Convolution(feature_4,kernel_4,feature_CONV_4,0,1);
    BN_GetCoeff(feature_CONV_4,bn_4);
    BNBatch_BatchNorm(feature_CONV_4,feature_BN_4,bn_4);
    FrontBatch_ReLU(feature_BN_4,feature_ReLU_4);
    //bottle neck 1 residual part
    Multi_Convolution(feature_2,resKernel_4,feature_Res_4,0,1);
    ResBatch_Sum(feature_Res_4,feature_ReLU_4,feature_Res_4);
    BN_GetCoeff(feature_Res_4,resbn_4);
    BNBatch_BatchNorm(feature_Res_4,feature_5,resbn_4);
    //printf("layer 4 done\n");
    //bottle neck 2
    //layer 5
    Multi_Convolution(feature_5,kernel_5,feature_CONV_5,0,1);
    BN_GetCoeff(feature_CONV_5,bn_5);
    BNBatch_BatchNorm(feature_CONV_5,feature_6,bn_5);
    //printf("layer 5 done\n");
    //layer 6
    Multi_Convolution(feature_6,kernel_6,feature_CONV_6,1,1);
    BN_GetCoeff(feature_CONV_6,bn_6);
    BNBatch_BatchNorm(feature_CONV_6,feature_7,bn_6);
    //printf("layer 6 done\n");
    //layer 7
    Multi_Convolution(feature_7,kernel_7,feature_CONV_7,0,1);
    BN_GetCoeff(feature_CONV_7,bn_7);
    BNBatch_BatchNorm(feature_CONV_7,feature_BN_7,bn_7);
    FrontBatch_ReLU(feature_BN_7,feature_ReLU_7);
    //bottle neck 2 residual part
    ResBatch_Sum(feature_5,feature_ReLU_7,feature_8);
    //printf("layer 7 done\n");
    //bottle neck 3
    //layer 8
    Multi_Convolution(feature_8,kernel_8,feature_CONV_8,0,1);
    BN_GetCoeff(feature_CONV_8,bn_8);
    BNBatch_BatchNorm(feature_CONV_8,feature_9,bn_8);
    //printf("layer 8 done\n");
    //layer 9
    Multi_Convolution(feature_9,kernel_9,feature_CONV_9,1,1);
    BN_GetCoeff(feature_CONV_9,bn_9);
    BNBatch_BatchNorm(feature_CONV_9,feature_10,bn_9);
    //printf("layer 9 done\n"); 
    //layer 10
    Multi_Convolution(feature_10,kernel_10,feature_CONV_10,0,1);
    BN_GetCoeff(feature_CONV_10,bn_10);
    BNBatch_BatchNorm(feature_CONV_10,feature_BN_10,bn_10);
    FrontBatch_ReLU(feature_BN_10,feature_ReLU_10);
    //bottle neck 3 residual part
    ResBatch_Sum(feature_8,feature_ReLU_10,feature_11);
    //printf("layer 10 done\n"); 
    //bottle neck 4
    //layer 11
    Multi_Convolution(feature_11,kernel_11,feature_CONV_11,0,1);
    BN_GetCoeff(feature_CONV_11,bn_11);
    BNBatch_BatchNorm(feature_CONV_11,feature_12,bn_11);
    //printf("layer 11 done\n");
    //layer 12
    Multi_Convolution(feature_12,kernel_12,feature_CONV_12,1,1);
    BN_GetCoeff(feature_CONV_12,bn_12);
    BNBatch_BatchNorm(feature_CONV_12,feature_13,bn_12);
    //printf("layer 12 done\n"); 
    //layer 13
    Multi_Convolution(feature_13,kernel_13,feature_CONV_13,0,1);
    BN_GetCoeff(feature_CONV_13,bn_13);
    BNBatch_BatchNorm(feature_CONV_13,feature_BN_13,bn_13);
    FrontBatch_ReLU(feature_BN_13,feature_ReLU_13);
    //bottle neck 4 residual part
    Multi_Convolution(feature_11,resKernel_13,feature_Res_13,0,1);
    ResBatch_Sum(feature_Res_13,feature_ReLU_13,feature_Res_13);
    BN_GetCoeff(feature_Res_13,resbn_13);
    BNBatch_BatchNorm(feature_Res_13,feature_14,resbn_13);
    //printf("layer 13 done\n"); 
    //bottle neck 5
    //layer 14
    Multi_Convolution(feature_14,kernel_14,feature_CONV_14,0,1);
    BN_GetCoeff(feature_CONV_14,bn_14);
    BNBatch_BatchNorm(feature_CONV_14,feature_15,bn_14);
    //printf("layer 14 done\n");
    //layer 15
    Multi_Convolution(feature_15,kernel_15,feature_CONV_15,1,1);
    BN_GetCoeff(feature_CONV_15,bn_15);
    BNBatch_BatchNorm(feature_CONV_15,feature_16,bn_15);
    //printf("layer 15 done\n");
    //layer 16
    Multi_Convolution(feature_16,kernel_16,feature_CONV_16,0,1);
    BN_GetCoeff(feature_CONV_16,bn_16);
    BNBatch_BatchNorm(feature_CONV_16,feature_BN_16,bn_16);
    FrontBatch_ReLU(feature_BN_16,feature_ReLU_16);
    //bottle neck 5 residual part
    ResBatch_Sum(feature_14,feature_ReLU_16,feature_17);
    //printf("layer 16 done\n"); 
    //bottle neck 6
    //layer 17
    Multi_Convolution(feature_17,kernel_17,feature_CONV_17,0,1);
    BN_GetCoeff(feature_CONV_17,bn_17);
    BNBatch_BatchNorm(feature_CONV_17,feature_18,bn_17);
    //printf("layer 17 done\n");
    //layer 18
    Multi_Convolution(feature_18,kernel_18,feature_CONV_18,1,1);
    BN_GetCoeff(feature_CONV_18,bn_18);
    BNBatch_BatchNorm(feature_CONV_18,feature_19,bn_18);
    //printf("layer 18 done\n");
    //layer 19
    Multi_Convolution(feature_19,kernel_19,feature_CONV_19,0,1);
    BN_GetCoeff(feature_CONV_19,bn_19);
    BNBatch_BatchNorm(feature_CONV_19,feature_BN_19,bn_19);
    FrontBatch_ReLU(feature_BN_19,feature_ReLU_19);
    //bottle neck 6 residual part
    ResBatch_Sum(feature_17,feature_ReLU_19,feature_20);
    //printf("layer 19 done\n");
    //bottle neck 7
    //layer 20
    Multi_Convolution(feature_20,kernel_20,feature_CONV_20,0,1);
    BN_GetCoeff(feature_CONV_20,bn_20);
    BNBatch_BatchNorm(feature_CONV_20,feature_21,bn_20);
    //printf("layer 20 done\n");
    //layer 21
    Multi_Convolution(feature_21,kernel_21,feature_CONV_21,1,2);
    BN_GetCoeff(feature_CONV_21,bn_21);
    BNBatch_BatchNorm(feature_CONV_21,feature_22,bn_21);
    //printf("layer 21 done\n");
    //layer 22
    Multi_Convolution(feature_22,kernel_22,feature_CONV_22,0,1);
    BN_GetCoeff(feature_CONV_22,bn_22);
    BNBatch_BatchNorm(feature_CONV_22,feature_BN_22,bn_22);
    FrontBatch_ReLU(feature_BN_22,feature_ReLU_22);
    //bottle neck 7 residual part
    Multi_Convolution(feature_20,resKernel_22,feature_Res_22,0,2);
    ResBatch_Sum(feature_Res_22,feature_ReLU_22,feature_Res_22);
    BN_GetCoeff(feature_Res_22,resbn_22);
    BNBatch_BatchNorm(feature_Res_22,feature_23,resbn_22);
    //printf("layer 22 done\n");
    //bottle neck 8
    //layer 23
    Multi_Convolution(feature_23,kernel_23,feature_CONV_23,0,1);
    BN_GetCoeff(feature_CONV_23,bn_23);
    BNBatch_BatchNorm(feature_CONV_23,feature_24,bn_23);
    //printf("layer 23 done\n");
    //layer 24
    Multi_Convolution(feature_24,kernel_24,feature_CONV_24,1,1);
    BN_GetCoeff(feature_CONV_24,bn_24);
    BNBatch_BatchNorm(feature_CONV_24,feature_25,bn_24);
    //printf("layer 24 done\n");
    //layer 25
    Multi_Convolution(feature_25,kernel_25,feature_CONV_25,0,1);
    BN_GetCoeff(feature_CONV_25,bn_25);
    BNBatch_BatchNorm(feature_CONV_25,feature_BN_25,bn_25);
    FrontBatch_ReLU(feature_BN_25,feature_ReLU_25);
    //bottle neck 8 residual part
    Multi_Convolution(feature_23,resKernel_25,feature_Res_25,0,1);
    ResBatch_Sum(feature_Res_25,feature_ReLU_25,feature_Res_25);
    BN_GetCoeff(feature_Res_25,resbn_25);
    BNBatch_BatchNorm(feature_Res_25,feature_26,resbn_25);
    //printf("layer 25 done\n");
    //bottle neck 9
    //layer 26
    Multi_Convolution(feature_26,kernel_26,feature_CONV_26,0,1);
    BN_GetCoeff(feature_CONV_26,bn_26);
    BNBatch_BatchNorm(feature_CONV_26,feature_27,bn_26);
    //printf("layer 26 done\n");
    //layer 27
    Multi_Convolution(feature_27,kernel_27,feature_CONV_27,1,1);
    BN_GetCoeff(feature_CONV_27,bn_27);
    BNBatch_BatchNorm(feature_CONV_27,feature_28,bn_27);
    //printf("layer 27 done\n");
    //layer 28
    Multi_Convolution(feature_28,kernel_28,feature_CONV_28,0,1);
    BN_GetCoeff(feature_CONV_28,bn_28);
    BNBatch_BatchNorm(feature_CONV_28,feature_BN_28,bn_28);
    FrontBatch_ReLU(feature_BN_28,feature_ReLU_28);
    //bottle neck 9 residual part
    ResBatch_Sum(feature_26,feature_ReLU_28,feature_29);
    //printf("layer 28 done\n");
    //bottle neck 10
    //layer 29
    Multi_Convolution(feature_29,kernel_29,feature_CONV_29,0,1);
    BN_GetCoeff(feature_CONV_29,bn_29);
    BNBatch_BatchNorm(feature_CONV_29,feature_30,bn_29);
    //printf("layer 29 done\n");
    //layer 30
    Multi_Convolution(feature_30,kernel_30,feature_CONV_30,1,1);
    BN_GetCoeff(feature_CONV_30,bn_30);
    BNBatch_BatchNorm(feature_CONV_30,feature_31,bn_30);
    //printf("layer 30 done\n");
    //layer 31
    Multi_Convolution(feature_31,kernel_31,feature_CONV_31,0,1);
    BN_GetCoeff(feature_CONV_31,bn_31);
    BNBatch_BatchNorm(feature_CONV_31,feature_BN_31,bn_31);
    FrontBatch_ReLU(feature_BN_31,feature_ReLU_31);
    //bottle neck 10 residual part
    ResBatch_Sum(feature_29,feature_ReLU_31,feature_32);
    //printf("layer 31 done\n");
    //bottle neck 11
    //layer 32
    Multi_Convolution(feature_32,kernel_32,feature_CONV_32,0,1);
    BN_GetCoeff(feature_CONV_32,bn_32);
    BNBatch_BatchNorm(feature_CONV_32,feature_33,bn_32);
    //printf("layer 32 done\n");
    //layer 33
    Multi_Convolution(feature_33,kernel_33,feature_CONV_33,1,1);
    BN_GetCoeff(feature_CONV_33,bn_33);
    BNBatch_BatchNorm(feature_CONV_33,feature_34,bn_33);
    //printf("layer 33 done\n");
    //layer 34
    Multi_Convolution(feature_34,kernel_34,feature_CONV_34,0,1);
    BN_GetCoeff(feature_CONV_34,bn_34);
    BNBatch_BatchNorm(feature_CONV_34,feature_BN_34,bn_34);
    FrontBatch_ReLU(feature_BN_34,feature_ReLU_34);
    //bottle neck 11 residual part
    ResBatch_Sum(feature_32,feature_ReLU_34,feature_35);
    //printf("layer 34 done\n");
    //bottle neck 12
    //layer 35
    Multi_Convolution(feature_35,kernel_35,feature_CONV_35,0,1);
    BN_GetCoeff(feature_CONV_35,bn_35);
    BNBatch_BatchNorm(feature_CONV_35,feature_36,bn_35);
    //printf("layer 35 done\n");
    //layer 36
    Multi_Convolution(feature_36,kernel_36,feature_CONV_36,1,1);
    BN_GetCoeff(feature_CONV_36,bn_36);
    BNBatch_BatchNorm(feature_CONV_36,feature_37,bn_36);
    //printf("layer 36 done\n");
    //layer 37
    Multi_Convolution(feature_37,kernel_37,feature_CONV_37,0,1);
    BN_GetCoeff(feature_CONV_37,bn_37);
    BNBatch_BatchNorm(feature_CONV_37,feature_BN_37,bn_37);
    FrontBatch_ReLU(feature_BN_37,feature_ReLU_37);
    //bottle neck 12 residual part
    ResBatch_Sum(feature_35,feature_ReLU_37,feature_38);
    //printf("layer 37 done\n");
    //bottle neck 13
    //layer 38
    Multi_Convolution(feature_38,kernel_38,feature_CONV_38,0,1);
    BN_GetCoeff(feature_CONV_38,bn_38);
    BNBatch_BatchNorm(feature_CONV_38,feature_39,bn_38);
    //printf("layer 38 done\n");
    //layer 39
    Multi_Convolution(feature_39,kernel_39,feature_CONV_39,1,2);
    BN_GetCoeff(feature_CONV_39,bn_39);
    BNBatch_BatchNorm(feature_CONV_39,feature_40,bn_39);
    //printf("layer 39 done\n");
    //layer 40
    Multi_Convolution(feature_40,kernel_40,feature_CONV_40,0,1);
    BN_GetCoeff(feature_CONV_40,bn_40);
    BNBatch_BatchNorm(feature_CONV_40,feature_BN_40,bn_40);
    FrontBatch_ReLU(feature_BN_40,feature_ReLU_40);
    //bottle neck 13 residual part
    Multi_Convolution(feature_38,resKernel_40,feature_Res_40,0,2);
    ResBatch_Sum(feature_Res_40,feature_ReLU_40,feature_Res_40);
    BN_GetCoeff(feature_Res_40,resbn_40);
    BNBatch_BatchNorm(feature_Res_40,feature_41,resbn_40);
    //printf("layer 40 done\n");
    //bottle neck 14
    //layer 41
    Multi_Convolution(feature_41,kernel_41,feature_CONV_41,0,1);
    BN_GetCoeff(feature_CONV_41,bn_41);
    BNBatch_BatchNorm(feature_CONV_41,feature_42,bn_41);
    //printf("layer 41 done\n");
    //layer 42
    Multi_Convolution(feature_42,kernel_42,feature_CONV_42,1,1);
    BN_GetCoeff(feature_CONV_42,bn_42);
    BNBatch_BatchNorm(feature_CONV_42,feature_43,bn_42);
    //printf("layer 42 done\n");
    //layer 43
    Multi_Convolution(feature_43,kernel_43,feature_CONV_43,0,1);
    BN_GetCoeff(feature_CONV_43,bn_43);
    BNBatch_BatchNorm(feature_CONV_43,feature_BN_43,bn_43);
    FrontBatch_ReLU(feature_BN_43,feature_ReLU_43);
    //bottle neck 14 residual part
    Multi_Convolution(feature_41,resKernel_43,feature_Res_43,0,1);
    ResBatch_Sum(feature_Res_43,feature_ReLU_43,feature_Res_43);
    BN_GetCoeff(feature_Res_43,resbn_43);
    BNBatch_BatchNorm(feature_Res_43,feature_44,resbn_43);
    //printf("layer 43 done\n");
    //bottle neck 15
    //layer 44
    Multi_Convolution(feature_44,kernel_44,feature_CONV_44,0,1);
    BN_GetCoeff(feature_CONV_44,bn_44);
    BNBatch_BatchNorm(feature_CONV_44,feature_45,bn_44);
    //printf("layer 44 done\n");
    //layer 45
    Multi_Convolution(feature_45,kernel_45,feature_CONV_45,1,1);
    BN_GetCoeff(feature_CONV_45,bn_45);
    BNBatch_BatchNorm(feature_CONV_45,feature_46,bn_45);
    //printf("layer 45 done\n");
    //layer 46
    Multi_Convolution(feature_46,kernel_46,feature_CONV_46,0,1);
    BN_GetCoeff(feature_CONV_46,bn_46);
    BNBatch_BatchNorm(feature_CONV_46,feature_BN_46,bn_46);
    FrontBatch_ReLU(feature_BN_46,feature_ReLU_46);
    //bottle neck 15 residual part
    ResBatch_Sum(feature_44,feature_ReLU_46,feature_47);
    //printf("layer 46 done\n");
    //bottle neck 16
    //layer 47
    Multi_Convolution(feature_47,kernel_47,feature_CONV_47,0,1);
    BN_GetCoeff(feature_CONV_47,bn_47);
    BNBatch_BatchNorm(feature_CONV_47,feature_48,bn_47);
    //printf("layer 47 done\n");
    //layer 48
    Multi_Convolution(feature_48,kernel_48,feature_CONV_48,1,2);
    BN_GetCoeff(feature_CONV_48,bn_48);
    BNBatch_BatchNorm(feature_CONV_48,feature_49,bn_48);
    //printf("layer 48 done\n");
    //layer 49
    Multi_Convolution(feature_49,kernel_49,feature_CONV_49,0,1);
    BN_GetCoeff(feature_CONV_49,bn_49);
    BNBatch_BatchNorm(feature_CONV_49,feature_BN_49,bn_49);
    FrontBatch_ReLU(feature_BN_49,feature_ReLU_49);
    //bottle neck 16 residual part
    Multi_Convolution(feature_47,resKernel_49,feature_50,0,2);
    ResBatch_Sum(feature_50,feature_ReLU_49,feature_50);
    //printf("layer 49 done\n");
    //global average
    FrontBatch_GlobalAverage(feature_50,feature_GlobalAvg);
    //full connect
    FrontBatch_FUllConnct(feature_GlobalAvg,weightFC,feature_FullConnect,bias);
    //softmax
    FrontBatch_Predict(feature_FullConnect,predictAns);
    FrontBatch_Softmax(feature_FullConnect,predictAns,feature_SoftMax);
    //printf("layer 50 done\n");
    //predict ans

    float error = FrontBatch_Accurancy(predictAns,testCaseLabel,index);
    //printf("frontpropagation done\n");
    return error;
}


void BackPropagation(int index)
{
    //printf("back propagation start\n");
    //struct of the gradient for back propagation
    //softmax
    Gradient_CostFunction(feature_SoftMax,testCaseLabel,index,gradient_CostFunction);
    //Matrix_Print(gradient_CostFunction[31]);
    //full connect
    Gradient_FullConnect(gradient_CostFunction,bias,gradient_bias,weightFC,feature_GlobalAvg
                            ,gradient_FullConnect_Weight,gradient_FullConnect);                    
    //global average
    Gradient_GlobalAverage(gradient_FullConnect,feature_50,gradient_GlobalAverage);
    //printf("layer 50 done\n");
    //bottle neck 16 residualpart
    Multi_GradientConvolution(gradient_GlobalAverage,resKernel_49,gradient_Res_CONV_49,feature_47
                        ,gradient_Res_kernel_49,2,0);
    //layer 49
    Gradient_ReLU(gradient_GlobalAverage,feature_BN_49,gradient_ReLU_49);
    Gradient_BatchNorm(gradient_ReLU_49,feature_CONV_49,bn_49,gradient_BN_49,feature_BN_49
                        ,gradient_BN_Beta_49,gradient_BN_Gamma_49);
    Multi_GradientConvolution(gradient_BN_49,kernel_49,gradient_49,feature_49,gradient_kernel_49,1,0);
    //printf("layer 49 done\n");
    //layer 48
    Gradient_BatchNorm(gradient_49,feature_CONV_48,bn_48,gradient_BN_48,feature_49
                        ,gradient_BN_Beta_48,gradient_BN_Gamma_48);
    Multi_GradientConvolution(gradient_BN_48,kernel_48,gradient_48,feature_48,gradient_kernel_48,2,1);
    //printf("layer 48 done\n");
    //layer 47
    Gradient_BatchNorm(gradient_48,feature_CONV_47,bn_47,gradient_BN_47,feature_48
                        ,gradient_BN_Beta_47,gradient_BN_Gamma_47);
    Multi_GradientConvolution(gradient_BN_47,kernel_47,gradient_47,feature_47,gradient_kernel_47,1,0); 
    ResBatch_Sum(gradient_47,gradient_Res_CONV_49,gradient_47);
    //printf("layer 47 done\n");
    //layer 46
    Gradient_ReLU(gradient_47,feature_BN_46,gradient_ReLU_46);
    Gradient_BatchNorm(gradient_ReLU_46,feature_CONV_46,bn_46,gradient_BN_46,feature_BN_46
                        ,gradient_BN_Beta_46,gradient_BN_Gamma_46);
    Multi_GradientConvolution(gradient_BN_46,kernel_46,gradient_46,feature_46,gradient_kernel_46,1,0);
    //printf("layer 46 done\n");
    //layer 45
    Gradient_BatchNorm(gradient_46,feature_CONV_45,bn_45,gradient_BN_45,feature_46
                        ,gradient_BN_Beta_45,gradient_BN_Gamma_45);
    Multi_GradientConvolution(gradient_BN_45,kernel_45,gradient_45,feature_45,gradient_kernel_45,1,1);
    //printf("layer 45 done\n");
    //layer 44
    Gradient_BatchNorm(gradient_45,feature_CONV_44,bn_44,gradient_BN_44,feature_45
                        ,gradient_BN_Beta_44,gradient_BN_Gamma_44);
    Multi_GradientConvolution(gradient_BN_44,kernel_44,gradient_44,feature_44,gradient_kernel_44,1,0);
    //printf("layer 44 done\n");
    //bottle neck 14 residualpart
    Gradient_BatchNorm(gradient_44,feature_Res_43,resbn_43,gradient_Res_BN_43,feature_44,
                        gradient_Res_BN_Beta_43,gradient_Res_BN_Gamma_43);
    ResBatch_Sum(gradient_Res_BN_43,gradient_47,gradient_44);
    //Matrix_Print(gradient_44[31]);
    Multi_GradientConvolution(gradient_44,resKernel_43,gradient_Res_CONV_43,feature_41
                        ,gradient_Res_kernel_43,1,0);
    //layer 43
    Gradient_ReLU(gradient_44,feature_BN_43,gradient_ReLU_43);
    Gradient_BatchNorm(gradient_ReLU_43,feature_CONV_43,bn_43,gradient_BN_43,feature_BN_43
                        ,gradient_BN_Beta_43,gradient_BN_Gamma_43);
    Multi_GradientConvolution(gradient_BN_43,kernel_43,gradient_43,feature_43,gradient_kernel_43,1,0);
    //printf("layer 43 done\n");
    //layer 42
    Gradient_BatchNorm(gradient_43,feature_CONV_42,bn_42,gradient_BN_42,feature_43
                        ,gradient_BN_Beta_42,gradient_BN_Gamma_42);
    Multi_GradientConvolution(gradient_BN_42,kernel_42,gradient_42,feature_42,gradient_kernel_42,1,1);
    //printf("layer 42 done\n");
    //layer 41
    Gradient_BatchNorm(gradient_42,feature_CONV_41,bn_41,gradient_BN_41,feature_42
                        ,gradient_BN_Beta_41,gradient_BN_Gamma_41);
    Multi_GradientConvolution(gradient_BN_41,kernel_41,gradient_41,feature_41,gradient_kernel_41,1,0);
    //printf("layer 41 done\n");
    //bottle neck 13 residualpart

    Gradient_BatchNorm(gradient_41,feature_Res_40,resbn_40,gradient_Res_BN_40,feature_41,
                        gradient_Res_BN_Beta_40,gradient_Res_BN_Gamma_40);   
    ResBatch_Sum(gradient_Res_BN_40,gradient_Res_CONV_43,gradient_41);
    //Matrix_Print(gradient_41[31]);
    Multi_GradientConvolution(gradient_41,resKernel_40,gradient_Res_CONV_40,feature_38
                        ,gradient_Res_kernel_40,2,0);
    //layer 40
    Gradient_ReLU(gradient_41,feature_BN_40,gradient_ReLU_40);
    Gradient_BatchNorm(gradient_ReLU_40,feature_CONV_40,bn_40,gradient_BN_40,feature_BN_40
                        ,gradient_BN_Beta_40,gradient_BN_Gamma_40);
    Multi_GradientConvolution(gradient_BN_40,kernel_40,gradient_40,feature_40,gradient_kernel_40,1,0);
    //printf("layer 40 done\n");
    //layer 39
    Gradient_BatchNorm(gradient_40,feature_CONV_39,bn_39,gradient_BN_39,feature_40
                        ,gradient_BN_Beta_39,gradient_BN_Gamma_39);
    Multi_GradientConvolution(gradient_BN_39,kernel_39,gradient_39,feature_39,gradient_kernel_39,2,1);
    //printf("layer 39 done\n");
    //layer 38
    Gradient_BatchNorm(gradient_39,feature_CONV_38,bn_38,gradient_BN_38,feature_39
                        ,gradient_BN_Beta_38,gradient_BN_Gamma_38);
    Multi_GradientConvolution(gradient_BN_38,kernel_38,gradient_38,feature_38,gradient_kernel_38,1,0);
    //printf("layer 38 done\n");
    //bottle neck 12 residualpart
    ResBatch_Sum(gradient_38,gradient_Res_CONV_40,gradient_38);
    //Matrix_Print(gradient_38[31]);
    //layer 37 here
    Gradient_ReLU(gradient_38,feature_BN_37,gradient_ReLU_37);
    Gradient_BatchNorm(gradient_ReLU_37,feature_CONV_37,bn_37,gradient_BN_37,feature_BN_37
                        ,gradient_BN_Beta_37,gradient_BN_Gamma_37);
    Multi_GradientConvolution(gradient_BN_37,kernel_37,gradient_37,feature_37,gradient_kernel_37,1,0);
    //printf("layer 37 done\n");
    //layer 36
    Gradient_BatchNorm(gradient_37,feature_CONV_36,bn_36,gradient_BN_36,feature_37
                        ,gradient_BN_Beta_36,gradient_BN_Gamma_36);
    Multi_GradientConvolution(gradient_BN_36,kernel_36,gradient_36,feature_36,gradient_kernel_36,1,1);
    //printf("layer 36 done\n");
    //layer 35
    Gradient_BatchNorm(gradient_36,feature_CONV_35,bn_35,gradient_BN_35,feature_36
                        ,gradient_BN_Beta_35,gradient_BN_Gamma_35);
    Multi_GradientConvolution(gradient_BN_35,kernel_35,gradient_35,feature_35,gradient_kernel_35,1,0);
    //printf("layer 35 done\n");
    //bottle neck 11 residualpart
    ResBatch_Sum(gradient_35,gradient_38,gradient_35);
    //Matrix_Print(gradient_35[31]);
    //layer 34
    Gradient_ReLU(gradient_35,feature_BN_34,gradient_ReLU_34);
    Gradient_BatchNorm(gradient_ReLU_34,feature_CONV_34,bn_34,gradient_BN_34,feature_BN_34
                        ,gradient_BN_Beta_34,gradient_BN_Gamma_34);
    Multi_GradientConvolution(gradient_BN_34,kernel_34,gradient_34,feature_34,gradient_kernel_34,1,0);
    //printf("layer 34 done\n");
    //layer 33
    Gradient_BatchNorm(gradient_34,feature_CONV_33,bn_33,gradient_BN_33,feature_34
                        ,gradient_BN_Beta_33,gradient_BN_Gamma_33);
    Multi_GradientConvolution(gradient_BN_33,kernel_33,gradient_33,feature_33,gradient_kernel_33,1,1);
    //printf("layer 33 done\n");
    //layer 32
    Gradient_BatchNorm(gradient_33,feature_CONV_32,bn_32,gradient_BN_32,feature_33
                        ,gradient_BN_Beta_32,gradient_BN_Gamma_32);
    Multi_GradientConvolution(gradient_BN_32,kernel_32,gradient_32,feature_32,gradient_kernel_32,1,0);
    //printf("layer 32 done\n");
    //bottle neck 10 residualpart
    ResBatch_Sum(gradient_32,gradient_35,gradient_32);
    //Matrix_Print(gradient_32[31]);
    //layer 31
    Gradient_ReLU(gradient_32,feature_BN_31,gradient_ReLU_31);
    Gradient_BatchNorm(gradient_ReLU_31,feature_CONV_31,bn_31,gradient_BN_31,feature_BN_31
                        ,gradient_BN_Beta_31,gradient_BN_Gamma_31);
    Multi_GradientConvolution(gradient_BN_31,kernel_31,gradient_31,feature_31,gradient_kernel_31,1,0);
    //printf("layer 31 done\n");
    //layer 30
    Gradient_BatchNorm(gradient_31,feature_CONV_30,bn_30,gradient_BN_30,feature_31
                        ,gradient_BN_Beta_30,gradient_BN_Gamma_30);
    Multi_GradientConvolution(gradient_BN_30,kernel_30,gradient_30,feature_30,gradient_kernel_30,1,1);
    //printf("layer 30 done\n");
    //layer 29
    Gradient_BatchNorm(gradient_30,feature_CONV_29,bn_29,gradient_BN_29,feature_30
                        ,gradient_BN_Beta_29,gradient_BN_Gamma_29);
    Multi_GradientConvolution(gradient_BN_29,kernel_29,gradient_29,feature_29,gradient_kernel_29,1,0);
    //printf("layer 29 done\n");
    //bottle neck 9 residualpart
    ResBatch_Sum(gradient_29,gradient_32,gradient_29);
    //Matrix_Print(gradient_29[31]);
    //layer 28
    Gradient_ReLU(gradient_29,feature_BN_28,gradient_ReLU_28);
    Gradient_BatchNorm(gradient_ReLU_28,feature_CONV_28,bn_28,gradient_BN_28,feature_BN_28
                        ,gradient_BN_Beta_28,gradient_BN_Gamma_28);
    Multi_GradientConvolution(gradient_BN_28,kernel_28,gradient_28,feature_28,gradient_kernel_28,1,0); //check here
    //printf("layer 28 done\n");
    //layer 27
    Gradient_BatchNorm(gradient_28,feature_CONV_27,bn_27,gradient_BN_27,feature_28
                        ,gradient_BN_Beta_27,gradient_BN_Gamma_27);
    Multi_GradientConvolution(gradient_BN_27,kernel_27,gradient_27,feature_27,gradient_kernel_27,1,1);
    //printf("layer 27 done\n");
    //layer 26
    Gradient_BatchNorm(gradient_27,feature_CONV_26,bn_26,gradient_BN_26,feature_27
                        ,gradient_BN_Beta_26,gradient_BN_Gamma_26);
    Multi_GradientConvolution(gradient_BN_26,kernel_26,gradient_26,feature_26,gradient_kernel_26,1,0);
    //printf("layer 26 done\n");
    //bottle neck 8 residualpart
    Gradient_BatchNorm(gradient_26,feature_Res_25,resbn_25,gradient_Res_BN_25,feature_26,
                        gradient_Res_BN_Beta_25,gradient_Res_BN_Gamma_25);
    ResBatch_Sum(gradient_Res_BN_25,gradient_29,gradient_26);
    //Matrix_Print(gradient_26[31]);
    Multi_GradientConvolution(gradient_26,resKernel_25,gradient_Res_CONV_25,feature_23,gradient_Res_kernel_25,1,0);
    //layer 25
    Gradient_ReLU(gradient_26,feature_BN_25,gradient_ReLU_25);
    Gradient_BatchNorm(gradient_ReLU_25,feature_CONV_25,bn_25,gradient_BN_25,feature_BN_25
                        ,gradient_BN_Beta_25,gradient_BN_Gamma_25);
    Multi_GradientConvolution(gradient_BN_25,kernel_25,gradient_25,feature_25,gradient_kernel_25,1,0);
    //printf("layer 25 done\n");
    //layer 24
    Gradient_BatchNorm(gradient_25,feature_CONV_24,bn_24,gradient_BN_24,feature_25
                        ,gradient_BN_Beta_24,gradient_BN_Gamma_24);
    Multi_GradientConvolution(gradient_BN_24,kernel_24,gradient_24,feature_24,gradient_kernel_24,1,1);
    //printf("layer 24 done\n");
    //layer 23
    Gradient_BatchNorm(gradient_24,feature_CONV_23,bn_23,gradient_BN_23,feature_24
                        ,gradient_BN_Beta_23,gradient_BN_Gamma_23);
    Multi_GradientConvolution(gradient_BN_23,kernel_23,gradient_23,feature_23,gradient_kernel_23,1,0);
    //printf("layer 23 done\n");
    //bottle neck 7 residualpart
    Gradient_BatchNorm(gradient_23,feature_Res_22,resbn_22,gradient_Res_BN_22,feature_23,
                        gradient_Res_BN_Beta_22,gradient_Res_BN_Gamma_22);
    ResBatch_Sum(gradient_Res_BN_22,gradient_Res_CONV_25,gradient_23);
    //Matrix_Print(gradient_23[31]);
    Multi_GradientConvolution(gradient_23,resKernel_22,gradient_Res_CONV_22,feature_20,gradient_Res_kernel_22,2,0);
    //layer 22
    Gradient_ReLU(gradient_23,feature_BN_22,gradient_ReLU_22);
    Gradient_BatchNorm(gradient_ReLU_22,feature_CONV_22,bn_22,gradient_BN_22,feature_BN_22
                        ,gradient_BN_Beta_22,gradient_BN_Gamma_22);
    Multi_GradientConvolution(gradient_BN_22,kernel_22,gradient_22,feature_22,gradient_kernel_22,1,0);
    //printf("layer 22 done\n");
    //layer 21
    Gradient_BatchNorm(gradient_22,feature_CONV_21,bn_21,gradient_BN_21,feature_22
                        ,gradient_BN_Beta_21,gradient_BN_Gamma_21);
    Multi_GradientConvolution(gradient_BN_21,kernel_21,gradient_21,feature_21,gradient_kernel_21,2,1);
    //printf("layer 21 done\n");
    //layer 20
    Gradient_BatchNorm(gradient_21,feature_CONV_20,bn_20,gradient_BN_20,feature_21
                        ,gradient_BN_Beta_20,gradient_BN_Gamma_20);
    Multi_GradientConvolution(gradient_BN_20,kernel_20,gradient_20,feature_20,gradient_kernel_20,1,0);
    //printf("layer 20 done\n");
    //bottle neck 6 residualpart
    ResBatch_Sum(gradient_20,gradient_Res_CONV_22,gradient_20);
    //Matrix_Print(gradient_20[31]);
    //layer 19
    Gradient_ReLU(gradient_20,feature_BN_19,gradient_ReLU_19);
    Gradient_BatchNorm(gradient_ReLU_19,feature_CONV_19,bn_19,gradient_BN_19,feature_BN_19
                        ,gradient_BN_Beta_19,gradient_BN_Gamma_19);
    Multi_GradientConvolution(gradient_BN_19,kernel_19,gradient_19,feature_19,gradient_kernel_19,1,0);
    //printf("layer 19 done\n");
    //layer 18
    Gradient_BatchNorm(gradient_19,feature_CONV_18,bn_18,gradient_BN_18,feature_19
                        ,gradient_BN_Beta_18,gradient_BN_Gamma_18);
    Multi_GradientConvolution(gradient_BN_18,kernel_18,gradient_18,feature_18,gradient_kernel_18,1,1);
    //printf("layer 18 done\n");
    //layer 17
    Gradient_BatchNorm(gradient_18,feature_CONV_17,bn_17,gradient_BN_17,feature_18
                        ,gradient_BN_Beta_17,gradient_BN_Gamma_17);
    Multi_GradientConvolution(gradient_BN_17,kernel_17,gradient_17,feature_17,gradient_kernel_17,1,0);
    //printf("layer 17 done\n");
    //bottle neck 5 residualpart
    ResBatch_Sum(gradient_17,gradient_20,gradient_17);
    //Matrix_Print(gradient_17[31]);
    //layer 16
    Gradient_ReLU(gradient_17,feature_BN_16,gradient_ReLU_16);
    Gradient_BatchNorm(gradient_ReLU_16,feature_CONV_16,bn_16,gradient_BN_16,feature_BN_16
                        ,gradient_BN_Beta_16,gradient_BN_Gamma_16);
    Multi_GradientConvolution(gradient_BN_16,kernel_16,gradient_16,feature_16,gradient_kernel_16,1,0);
    //printf("layer 16 done\n");
    //layer 15
    Gradient_BatchNorm(gradient_16,feature_CONV_15,bn_15,gradient_BN_15,feature_16
                        ,gradient_BN_Beta_15,gradient_BN_Gamma_15);
    Multi_GradientConvolution(gradient_BN_15,kernel_15,gradient_15,feature_15,gradient_kernel_15,1,1);
    //printf("layer 15 done\n");
    //layer 14
    Gradient_BatchNorm(gradient_15,feature_CONV_14,bn_14,gradient_BN_14,feature_15
                        ,gradient_BN_Beta_14,gradient_BN_Gamma_14);
    Multi_GradientConvolution(gradient_BN_14,kernel_14,gradient_14,feature_14,gradient_kernel_14,1,0);
    //printf("layer 14 done\n");
    //bottle neck 4 residualpart
    Gradient_BatchNorm(gradient_14,feature_Res_13,resbn_13,gradient_Res_BN_13,feature_14,
                        gradient_Res_BN_Beta_13,gradient_Res_BN_Gamma_13);
    ResBatch_Sum(gradient_Res_BN_13,gradient_17,gradient_14);
    //Matrix_Print(gradient_14[31]);
    Multi_GradientConvolution(gradient_14,resKernel_13,gradient_Res_CONV_13,feature_11,gradient_Res_kernel_13,1,0);
    //layer 13
    Gradient_ReLU(gradient_14,feature_BN_13,gradient_ReLU_13);
    Gradient_BatchNorm(gradient_ReLU_13,feature_CONV_13,bn_13,gradient_BN_13,feature_BN_13
                        ,gradient_BN_Beta_13,gradient_BN_Gamma_13);
    Multi_GradientConvolution(gradient_BN_13,kernel_13,gradient_13,feature_13,gradient_kernel_13,1,0);
    //printf("layer 13 done\n");
    //layer 12
    Gradient_BatchNorm(gradient_13,feature_CONV_12,bn_12,gradient_BN_12,feature_13
                        ,gradient_BN_Beta_12,gradient_BN_Gamma_12);
    Multi_GradientConvolution(gradient_BN_12,kernel_12,gradient_12,feature_12,gradient_kernel_12,1,1);
    //printf("layer 12 done\n");
    //layer 11
    Gradient_BatchNorm(gradient_12,feature_CONV_11,bn_11,gradient_BN_11,feature_12
                        ,gradient_BN_Beta_11,gradient_BN_Gamma_11);
    Multi_GradientConvolution(gradient_BN_11,kernel_11,gradient_11,feature_11,gradient_kernel_11,1,0);
    //printf("layer 11 done\n");
    //bottle neck 3 residualpart
    ResBatch_Sum(gradient_11,gradient_Res_CONV_13,gradient_11);
    //Matrix_Print(gradient_11[31]);
    //layer 10
    Gradient_ReLU(gradient_11,feature_BN_10,gradient_ReLU_10);
    Gradient_BatchNorm(gradient_ReLU_10,feature_CONV_10,bn_10,gradient_BN_10,feature_BN_10
                        ,gradient_BN_Beta_10,gradient_BN_Gamma_10);
    Multi_GradientConvolution(gradient_BN_10,kernel_10,gradient_10,feature_10,gradient_kernel_10,1,0);
    //printf("layer 10 done\n");
    //layer 9
    Gradient_BatchNorm(gradient_10,feature_CONV_9,bn_9,gradient_BN_9,feature_10
                        ,gradient_BN_Beta_9,gradient_BN_Gamma_9);
    Multi_GradientConvolution(gradient_BN_9,kernel_9,gradient_9,feature_9,gradient_kernel_9,1,1);
    //printf("layer 9 done\n");
    //layer 8
    Gradient_BatchNorm(gradient_9,feature_CONV_8,bn_8,gradient_BN_8,feature_9
                        ,gradient_BN_Beta_8,gradient_BN_Gamma_8);
    Multi_GradientConvolution(gradient_BN_8,kernel_8,gradient_8,feature_8,gradient_kernel_8,1,0);
    //printf("layer 8 done\n");
    //bottle neck 2 residualpart
    ResBatch_Sum(gradient_8,gradient_11,gradient_8);
    //Matrix_Print(gradient_8[31]);
    //layer 7
    Gradient_ReLU(gradient_8,feature_BN_7,gradient_ReLU_7);
    Gradient_BatchNorm(gradient_ReLU_7,feature_CONV_7,bn_7,gradient_BN_7,feature_BN_7
                        ,gradient_BN_Beta_7,gradient_BN_Gamma_7);
    Multi_GradientConvolution(gradient_BN_7,kernel_7,gradient_7,feature_7,gradient_kernel_7,1,0);
    //printf("layer 7 done\n");
    //layer 6
    Gradient_BatchNorm(gradient_7,feature_CONV_6,bn_6,gradient_BN_6,feature_7
                        ,gradient_BN_Beta_6,gradient_BN_Gamma_6);
    Multi_GradientConvolution(gradient_BN_6,kernel_6,gradient_6,feature_6,gradient_kernel_6,1,1);
    //printf("layer 6 done\n");
    //layer 5
    Gradient_BatchNorm(gradient_6,feature_CONV_5,bn_5,gradient_BN_5,feature_6
                        ,gradient_BN_Beta_5,gradient_BN_Gamma_5);
    Multi_GradientConvolution(gradient_BN_5,kernel_5,gradient_5,feature_5,gradient_kernel_5,1,0);
    //printf("layer 5 done\n");
    //bottle neck 1 residualpart
    Gradient_BatchNorm(gradient_5,feature_Res_4,resbn_4,gradient_Res_BN_4,feature_5,
                        gradient_Res_BN_Beta_4,gradient_Res_BN_Gamma_4);
    ResBatch_Sum(gradient_Res_BN_4,gradient_8,gradient_5);
    //Matrix_Print(gradient_5[31]);
    Multi_GradientConvolution(gradient_5,resKernel_4,gradient_Res_CONV_4,feature_2,gradient_Res_kernel_4,1,0);
    //layer 4
    Gradient_ReLU(gradient_5,feature_BN_4,gradient_ReLU_4);
    Gradient_BatchNorm(gradient_ReLU_4,feature_CONV_4,bn_4,gradient_BN_4,feature_BN_4
                        ,gradient_BN_Beta_4,gradient_BN_Gamma_4);
    Multi_GradientConvolution(gradient_BN_4,kernel_4,gradient_4,feature_4,gradient_kernel_4,1,0);
    //printf("layer 4 done\n");
    //layer 3
    Gradient_BatchNorm(gradient_4,feature_CONV_3,bn_3,gradient_BN_3,feature_4
                        ,gradient_BN_Beta_3,gradient_BN_Gamma_3);
    Multi_GradientConvolution(gradient_BN_3,kernel_3,gradient_3,feature_3,gradient_kernel_3,1,1);
    //printf("layer 3 done\n");
    //layer 2
    Gradient_BatchNorm(gradient_3,feature_CONV_2,bn_2,gradient_BN_2,feature_3
                        ,gradient_BN_Beta_2,gradient_BN_Gamma_2);
    Multi_GradientConvolution(gradient_BN_2,kernel_2,gradient_2,feature_2,gradient_kernel_2,1,0);
    ResBatch_Sum(gradient_2,gradient_Res_CONV_4,gradient_2);
    //Matrix_Print(gradient_22[31]);
    //printf("layer 2 done\n");
    //layer 1
    Gradient_MaxPooling(gradient_2,feature_ReLU_1,feature_2,2,0,2,gradient_MaxPooling);
    //Matrix_Print(gradient_MaxPooling[31]);
    Gradient_ReLU(gradient_MaxPooling,feature_BN_1,gradient_ReLU_1);
    Gradient_BatchNorm(gradient_ReLU_1,feature_CONV_1,bn_1,gradient_BN_1,feature_BN_1
                        ,gradient_BN_Beta_1,gradient_BN_Gamma_1);
    Multi_GradientConvolution(gradient_BN_1,kernel_1,gradient_1,feature_0,gradient_kernel_1,1,1);
    //Matrix_Print(gradient_1[31]);
    //printf("back propagation done!\n");
}

void Write_Kernel(FILE *kernelFile,struct Matrix **kernel,int kernelAmt)
{
    int channel = kernel[0]->channelSize;
    int row = kernel[0]->rowSize;
    int column = kernel[0]->columnSize;
    for(int num = 0 ; num < kernelAmt ; num++)
    {
        for(int x = 0 ; x < channel ; x ++ )
        {
            for(int y = 0 ; y < row ; y ++ )
            {
                for(int z = 0 ; z < column ; z ++ )
                {
                    fprintf(kernelFile,"%e ",kernel[num]->feature[x][y][z]);
                }
            }
        }
    }
}

void Write_Matrix(FILE *kernelFile,struct Matrix *weight,struct Matrix *bias)
{
    int channel = weight->channelSize;
    int row = weight->rowSize;
    int column = weight->columnSize;
    for(int x = 0 ; x < channel ; x ++ )
    {
        for(int y = 0 ; y < row ; y ++ )
        {
            for(int z = 0 ; z < column ; z ++ )
            {
                fprintf(kernelFile,"%e ",weight->feature[x][y][z]);
            }
        }
    }
    channel = bias->rowSize;
    for(int j = 0 ; j < channel ; j++)
    {
        fprintf(kernelFile,"%e ",bias->feature[0][j][0]);
    }
}

void Write_BN(FILE *kernelFile, struct BN *coeff)
{
    int channel = coeff->channelSize;
    for(int i = 0 ; i < channel ; i ++)
    {
        fprintf(kernelFile,"%e ",coeff->gamma[i]);
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        fprintf(kernelFile,"%e ",coeff->beta[i]);
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        fprintf(kernelFile,"%e ",coeff->runningMean[i]);
    }
    for(int i = 0 ; i < channel ; i ++) 
    {
        fprintf(kernelFile,"%e ",coeff->runningVar[i]);
    }
}

void Record_Kernel(char *fileName)
{
    FILE* kernelFile = fopen(fileName,"w");
    Write_Kernel(kernelFile,kernel_1,32);
    Write_BN(kernelFile,bn_1);
    //bottleneck 1
    Write_Kernel(kernelFile,kernel_2,32);
    Write_BN(kernelFile,bn_2);
    Write_Kernel(kernelFile,kernel_3,32);
    Write_BN(kernelFile,bn_3);
    Write_Kernel(kernelFile,kernel_4,128);
    Write_BN(kernelFile,bn_4);
    Write_Kernel(kernelFile,resKernel_4,128);
    Write_BN(kernelFile,resbn_4);
    //bottleneck 2
    Write_Kernel(kernelFile,kernel_5,32);
    Write_BN(kernelFile,bn_5);
    Write_Kernel(kernelFile,kernel_6,32);
    Write_BN(kernelFile,bn_6);
    Write_Kernel(kernelFile,kernel_7,128);
    Write_BN(kernelFile,bn_7);
    //bottleneck 3
    Write_Kernel(kernelFile,kernel_8,32);
    Write_BN(kernelFile,bn_8);
    Write_Kernel(kernelFile,kernel_9,32);
    Write_BN(kernelFile,bn_9);
    Write_Kernel(kernelFile,kernel_10,128);
    Write_BN(kernelFile,bn_10);
    //bottleneck 4
    Write_Kernel(kernelFile,kernel_11,64);
    Write_BN(kernelFile,bn_11);
    Write_Kernel(kernelFile,kernel_12,64);
    Write_BN(kernelFile,bn_12);
    Write_Kernel(kernelFile,kernel_13,256);
    Write_BN(kernelFile,bn_13);
    Write_Kernel(kernelFile,resKernel_13,256);
    Write_BN(kernelFile,resbn_13);
    //bottleneck 5
    Write_Kernel(kernelFile,kernel_14,64);
    Write_BN(kernelFile,bn_14);
    Write_Kernel(kernelFile,kernel_15,64);
    Write_BN(kernelFile,bn_15);
    Write_Kernel(kernelFile,kernel_16,256);
    Write_BN(kernelFile,bn_16);
    //bottleneck 6
    Write_Kernel(kernelFile,kernel_17,64);
    Write_BN(kernelFile,bn_17);
    Write_Kernel(kernelFile,kernel_18,64);
    Write_BN(kernelFile,bn_18);
    Write_Kernel(kernelFile,kernel_19,256);
    Write_BN(kernelFile,bn_19);
    //bottleneck 7
    Write_Kernel(kernelFile,kernel_20,64);
    Write_BN(kernelFile,bn_20);
    Write_Kernel(kernelFile,kernel_21,64);
    Write_BN(kernelFile,bn_21);
    Write_Kernel(kernelFile,kernel_22,256);
    Write_BN(kernelFile,bn_22);
    Write_Kernel(kernelFile,resKernel_22,256);
    Write_BN(kernelFile,resbn_22);
    //bottleneck 8
    Write_Kernel(kernelFile,kernel_23,128);
    Write_BN(kernelFile,bn_23);
    Write_Kernel(kernelFile,kernel_24,128);
    Write_BN(kernelFile,bn_24);
    Write_Kernel(kernelFile,kernel_25,512);
    Write_BN(kernelFile,bn_25);
    Write_Kernel(kernelFile,resKernel_25,512);
    Write_BN(kernelFile,resbn_25);
    //bottleneck 9
    Write_Kernel(kernelFile,kernel_26,128);
    Write_BN(kernelFile,bn_26);
    Write_Kernel(kernelFile,kernel_27,128);
    Write_BN(kernelFile,bn_27);
    Write_Kernel(kernelFile,kernel_28,512);
    Write_BN(kernelFile,bn_28);
    //bottleneck 10
    Write_Kernel(kernelFile,kernel_29,128);
    Write_BN(kernelFile,bn_29);
    Write_Kernel(kernelFile,kernel_30,128);
    Write_BN(kernelFile,bn_30);
    Write_Kernel(kernelFile,kernel_31,512);
    Write_BN(kernelFile,bn_31);
    //bottleneck 11
    Write_Kernel(kernelFile,kernel_32,128);
    Write_BN(kernelFile,bn_32);
    Write_Kernel(kernelFile,kernel_33,128);
    Write_BN(kernelFile,bn_33);
    Write_Kernel(kernelFile,kernel_34,512);
    Write_BN(kernelFile,bn_34);
    //bottleneck 12
    Write_Kernel(kernelFile,kernel_35,128);
    Write_BN(kernelFile,bn_35);
    Write_Kernel(kernelFile,kernel_36,128);
    Write_BN(kernelFile,bn_36);
    Write_Kernel(kernelFile,kernel_37,512);
    Write_BN(kernelFile,bn_37);
    //bottleneck 13
    Write_Kernel(kernelFile,kernel_38,128);
    Write_BN(kernelFile,bn_38);
    Write_Kernel(kernelFile,kernel_39,128);
    Write_BN(kernelFile,bn_39);
    Write_Kernel(kernelFile,kernel_40,512);  
    Write_BN(kernelFile,bn_40);
    Write_Kernel(kernelFile,resKernel_40,512);   
    Write_BN(kernelFile,resbn_40);
    //bottleneck 14
    Write_Kernel(kernelFile,kernel_41,256);
    Write_BN(kernelFile,bn_41);
    Write_Kernel(kernelFile,kernel_42,256);
    Write_BN(kernelFile,bn_42);
    Write_Kernel(kernelFile,kernel_43,1024); 
    Write_BN(kernelFile,bn_43); 
    Write_Kernel(kernelFile,resKernel_43,1024); 
    Write_BN(kernelFile,resbn_43); 
    //bottleneck 15
    Write_Kernel(kernelFile,kernel_44,256);
    Write_BN(kernelFile,bn_44);
    Write_Kernel(kernelFile,kernel_45,256);
    Write_BN(kernelFile,bn_45);
    Write_Kernel(kernelFile,kernel_46,1024);  
    Write_BN(kernelFile,bn_46);
    //bottleneck 16
    Write_Kernel(kernelFile,kernel_47,256);
    Write_BN(kernelFile,bn_47);
    Write_Kernel(kernelFile,kernel_48,256);
    Write_BN(kernelFile,bn_48);
    Write_Kernel(kernelFile,kernel_49,1024);
    Write_BN(kernelFile,bn_49);
    Write_Kernel(kernelFile,resKernel_49,1024); 
    //fc
    Write_Matrix(kernelFile,weightFC,bias);

    fclose(kernelFile);
    
}
