#include<stdio.h>
#include<stdlib.h>
#include"matrix.h"
#include"input.h"
#include"batchnorm.h"
#include"frontpropagation.h"
#include"backpropagation.h"
#include"init.h"


float *Gradient_Init(int size)
{
    float *gradient = (float*)malloc(size*sizeof(float));
    for(int num = 0 ; num < size ; num ++)
    {
        gradient[num] = 0 ;
    }
    return gradient;
}

struct Matrix **Kernel_Init(int size , int channel , int row , int column)
{
    struct Matrix **kernel = (struct Matrix **)malloc(size * sizeof(struct Matrix*));
    if(kernel == NULL)
    {
        perror("Failed to allocate memory --- Kernel List\n");
    }
    for(int num = 0 ; num < size ; num++)
    {
        kernel[num] = Matrix_Init(Random,channel,row,column);
    }
    return kernel;
}

struct Matrix **Feature_Init(int channel , int row , int column)
{
    struct Matrix **feature = (struct Matrix **)malloc(BATCH * sizeof(struct Matrix*));
    if(feature == NULL)
    {
        perror("Failed to allocate memory --- Kernel List\n");
    }
    for(int num = 0 ; num < BATCH ; num++)
    {
        feature[num] = Matrix_Init(Zero,channel,row,column);
    }
    return feature;
}

void Feature_Copy(struct Matrix **feature,int index,struct Matrix **input)
{
    int start = index*BATCH;
    for(int num = 0 ; num < BATCH ; num++)
    {
        for(int i = 0 ; i < 3 ; i++)
        {
            for(int j = 0 ; j < 32 ; j ++)
            {
                for(int k = 0 ; k < 32 ; k++)
                {
                    feature[num]->feature[i][j][k] = input[start+num]->feature[i][j][k];
                }
            }
        }
    }
}

void Init_Weight()
{
    kernel_1 = Kernel_Init(32,3,3,3); //amt 32 , channel 3 , size 3
    bn_1 = BN_Init(32);       //channel 32

    kernel_2 = Kernel_Init(32,32,1,1); //amt 32 , channel 32 , size 1
    bn_2 = BN_Init(32);    // channel 32
    kernel_3  = Kernel_Init(32,32,3,3); //amt 32 , channel 32 , size 3
    bn_3 = BN_Init(32); //channel 32
    kernel_4 = Kernel_Init(128,32,1,1); //amt 128 , channel 32 , size 1
    bn_4 = BN_Init(128); //channel 128
    resKernel_4 = Kernel_Init(128,32,1,1); //amt 128 , channel 32, size 1
    resbn_4 = BN_Init(128); //channel 128

    kernel_5 = Kernel_Init(32,128,1,1); //amt 32 , channel 128 , size 1
    bn_5 = BN_Init(32); //channel 32
    kernel_6 = Kernel_Init(32,32,3,3); //amt 32 , channel 32 , size 3
    bn_6 = BN_Init(32); //channel 32
    kernel_7 = Kernel_Init(128,32,1,1); //amt 128 , channel 32 , size 1
    bn_7 = BN_Init(128); //channel 128

    kernel_8 = Kernel_Init(32,128,1,1); //amt 32 , channel 128 , size 1
    bn_8 = BN_Init(32); //channel 32
    kernel_9 = Kernel_Init(32,32,3,3); //amt 32 , channel 32 , size 3
    bn_9 = BN_Init(32); //channel 32
    kernel_10 = Kernel_Init(128,32,1,1); //amt 128 , channel 32, size 1
    bn_10 =  BN_Init(128); //channel 128

    kernel_11 = Kernel_Init(64,128,1,1); //amt 64 , channel 128, size 1
    bn_11 =  BN_Init(64); //channel 64
    kernel_12 = Kernel_Init(64,64,3,3); //amt 64 , channel 64,  size 3
    bn_12 =  BN_Init(64); //channel 64
    kernel_13 = Kernel_Init(256,64,1,1); //amt 256 , channel 64, size 1
    bn_13 =  BN_Init(256); //channel 256
    resKernel_13 = Kernel_Init(256,128,1,1); // amt 256 , channel 128, size 1
    resbn_13 = BN_Init(256); //channel 256

    kernel_14 = Kernel_Init(64,256,1,1); //amt 64 , channel 256 , size 1
    bn_14 =  BN_Init(64); //channel 64
    kernel_15 = Kernel_Init(64,64,3,3); // amt 64 , channel 64 ,size 3
    bn_15 = BN_Init(64); //channel 64
    kernel_16 = Kernel_Init(256,64,1,1); //amt 256, channel 64 , size 1
    bn_16 =  BN_Init(256); //channel 256

    kernel_17 = Kernel_Init(64,256,1,1); //amt 64 , channel 256 ,size 1
    bn_17 = BN_Init(64); //channel 64
    kernel_18 = Kernel_Init(64,64,3,3); // amt 64 , channel 64 ,size 3
    bn_18 = BN_Init(64); //channel 64
    kernel_19 = Kernel_Init(256,64,1,1); // amt256, channel 64 ,size 1
    bn_19 = BN_Init(256); //channel 256

    kernel_20 = Kernel_Init(64,256,1,1); //amt 64 , channel 256 ,size 1
    bn_20  = BN_Init(64); //channel 64
    kernel_21 = Kernel_Init(64,64,3,3); // amt 64 , channel 64 ,size 3
    bn_21 = BN_Init(64); //channel 64
    kernel_22 = Kernel_Init(256,64,1,1); // amt256, channel 64 ,size 1
    bn_22 = BN_Init(256); //channel 256
    resKernel_22 = Kernel_Init(256,256,1,1); // amt 256 , channel 256 , size 1
    resbn_22 = BN_Init(256); // channel 256

    kernel_23 = Kernel_Init(128,256,1,1); //amt 128 ,channel 256 ,size 1
    bn_23 = BN_Init(128); //channel 128
    kernel_24 = Kernel_Init(128,128,3,3); // amt 128 , channel 128 , size 3
    bn_24 = BN_Init(128); // channel 128
    kernel_25 = Kernel_Init(512,128,1,1); // amt 512 , channel 128 , size 1
    bn_25 = BN_Init(512); // channel 512
    resKernel_25 = Kernel_Init(512,256,1,1); //amt 512 , channel 256 , size 1
    resbn_25 = BN_Init(512); //channel 512

    kernel_26 = Kernel_Init(128,512,1,1); // amt 128 , channel 512 , size 1
    bn_26 = BN_Init(128); // channel 128
    kernel_27 = Kernel_Init(128,128,3,3); //amt 128 , channel 128 , size 3
    bn_27 = BN_Init(128); // channel 128
    kernel_28 = Kernel_Init(512,128,1,1); //amt 512 , channel 128 , size 1
    bn_28 = BN_Init(512); //channel 512

    kernel_29 = Kernel_Init(128,512,1,1); // amt 128 , channel 512 ,size 1
    bn_29 = BN_Init(128); // channel 128
    kernel_30 = Kernel_Init(128,128,3,3); // amt 128 , channel 128 , size 3
    bn_30 = BN_Init(128); // channel 128
    kernel_31 = Kernel_Init(512,128,1,1); // amt 512 , channel 128 , size 1
    bn_31 = BN_Init(512); //channel 512

    kernel_32 = Kernel_Init(128,512,1,1); // amt 128 , channel 512 , size 1
    bn_32 = BN_Init(128); // channel 128
    kernel_33 = Kernel_Init(128,128,3,3); // amt 128 , channel 128 , size 3
    bn_33 = BN_Init(128); //channel 128
    kernel_34 = Kernel_Init(512,128,1,1); // amt 512 , channel 128 , size 1
    bn_34 = BN_Init(512); //channel 512

    kernel_35 = Kernel_Init(128,512,1,1); //amt 128 , channel 512 , size 1
    bn_35 = BN_Init(128); //channel 128
    kernel_36 = Kernel_Init(128,128,3,3); // amt 128 , channel 128 , size 3
    bn_36 = BN_Init(128); //channel 128
    kernel_37 = Kernel_Init(512,128,1,1); // amt 512 , channel 128 , size 1
    bn_37 = BN_Init(512); // channel 512

    kernel_38 = Kernel_Init(128,512,1,1); //amt 128 , channel 512 , size 1
    bn_38 = BN_Init(128); //channel 128
    kernel_39 = Kernel_Init(128,128,3,3); //amt 128 , channel 128 , size 3
    bn_39 = BN_Init(128); //channel 128
    kernel_40 = Kernel_Init(512,128,1,1); //amt 512 , channel 128 , size 1
    bn_40 = BN_Init(512); //channel 512
    resKernel_40 = Kernel_Init(512,512,1,1); //amt 512 , channel 512 , size 1
    resbn_40 = BN_Init(512); //channel 512

    kernel_41 = Kernel_Init(256,512,1,1); //amt 256 , channel 512 , size 1
    bn_41 = BN_Init(256) ; //channel 256
    kernel_42 = Kernel_Init(256,256,3,3) ; //amt 256 , channel 256 , size 3
    bn_42 = BN_Init(256) ; //channel 256
    kernel_43 = Kernel_Init(1024,256,1,1) ; //amt 1024 , channel 256 , size 1
    bn_43 = BN_Init(1024) ; //channel 1024
    resKernel_43 = Kernel_Init(1024,512,1,1); //amt 1024 , channel 512 , size 1
    resbn_43 = BN_Init(1024); //channel 1024

    kernel_44 = Kernel_Init(256,1024,1,1); //amt 256 , channel 1024 , size 1
    bn_44 = BN_Init(256); //channel 256
    kernel_45 = Kernel_Init(256,256,3,3); //amt 256 , channel 256 , size 3
    bn_45 = BN_Init(256); // channel 256
    kernel_46 = Kernel_Init(1024,256,1,1); //amt 1024 , channel 256 , size 1
    bn_46 = BN_Init(1024); // channel 1024

    kernel_47 = Kernel_Init(256,1024,1,1); // amt 256 , channel 1024 , size 1
    bn_47 = BN_Init(256); // channel 256
    kernel_48 = Kernel_Init(256,256,3,3); // amt 256 , channel 256 , size 3
    bn_48 = BN_Init(256); //channel 256
    kernel_49 = Kernel_Init(1024,256,1,1); //amt 1024 , channel 256 , size 1
    bn_49 = BN_Init(1024); // channel 1024
    resKernel_49 = Kernel_Init(1024,1024,1,1); // amt 1024 , channel 1024 , size 1
    weightFC = Matrix_Init(Random,1,predictSize,1024); //row predictSize , column 1024
    bias = Matrix_Init(Zero,1,predictSize,1); // row predictSize , column 1
}

void Init_Feature()
{
    //layer 0
    feature_0 = Feature_Init(3,32,32);
    //layer 1
    feature_CONV_1 = Feature_Init(32,32,32); //channel 32 , size 32
    feature_BN_1 = Feature_Init(32,32,32); //channel 32 , size 16
    feature_ReLU_1 = Feature_Init(32,32,32); //channel 32 , size 16
    feature_2 = Feature_Init(32,16,16); //channel 32 , size 16
    //bottleneck 1
    //layer 2
    feature_CONV_2 = Feature_Init(32,16,16); //channel 32 , size 16
    feature_3 = Feature_Init(32,16,16); //channel 32 , size 16
    //layer 3
    feature_CONV_3 = Feature_Init(32,16,16); //channel 32 , size 16
    feature_4 = Feature_Init(32,16,16); //channel 32 , size 16
    //layer 4
    feature_CONV_4 = Feature_Init(128,16,16); //channel 128 , size 16
    feature_BN_4 = Feature_Init(128,16,16); //channel 128 , size 16
    feature_ReLU_4 = Feature_Init(128,16,16); //channel 128 , size 16
    //bottle neck 1 residual part
    feature_Res_4 = Feature_Init(128,16,16); //channel 128 , size 16
    feature_5 = Feature_Init(128,16,16); //channel 128 , size 16
    //bottle neck 2
    //layer 5
    feature_CONV_5 = Feature_Init(32,16,16); //channel 32 , size 16
    feature_6 = Feature_Init(32,16,16); //channel 32 , size 16
    //layer 6
    feature_CONV_6 = Feature_Init(32,16,16); //channel 32 , size 16
    feature_7 = Feature_Init(32,16,16); //channel 32 , size 16
    //layer 7
    feature_CONV_7 = Feature_Init(128,16,16); //channel 128 , size 16
    feature_BN_7 = Feature_Init(128,16,16); //channel 128 , size 16
    feature_ReLU_7 = Feature_Init(128,16,16); //channel 128 , size 16
    //bottle neck 2 residual part
    feature_8 = Feature_Init(128,16,16); //channel 128 , size 16
    //bottle neck 3
    //layer 8
    feature_CONV_8 = Feature_Init(32,16,16); //channel 32 , size 16
    feature_9 = Feature_Init(32,16,16); //channel 32 , size 16
    //layer 9
    feature_CONV_9 = Feature_Init(32,16,16); //channel 32 , size 16
    feature_10 = Feature_Init(32,16,16); //channel 32 , size 16
    //layer 10
    feature_CONV_10 = Feature_Init(128,16,16); //channel 128 , size 16
    feature_BN_10 = Feature_Init(128,16,16); //channel 128 , size 16
    feature_ReLU_10 = Feature_Init(128,16,16); //channel 128 , size 16
    //bottle neck 3 residual part
    feature_11 = Feature_Init(128,16,16); //channel 128 , size 16
    //bottle neck 4
    //layer 11
    feature_CONV_11 = Feature_Init(64,16,16); //channel 64 , size 16
    feature_12 = Feature_Init(64,16,16); //channel 64 , size 16
    //layer 12
    feature_CONV_12 = Feature_Init(64,16,16); //channel 64 , size 16
    feature_13 = Feature_Init(64,16,16); //channel 64 , size 16
    //layer 13
    feature_CONV_13 = Feature_Init(256,16,16); //channel 256 , size 16
    feature_BN_13 = Feature_Init(256,16,16); //channel 256 , size 16
    feature_ReLU_13 = Feature_Init(256,16,16); //channel 256 , size 16
    //bottle neck 4 residual part
    feature_Res_13 = Feature_Init(256,16,16); //channel 256 , size 16
    feature_14 = Feature_Init(256,16,16); //channel 256 , size 16
    //bottle neck 5
    //layer 14
    feature_CONV_14 = Feature_Init(64,16,16); //channel 64 , size 16
    feature_15 = Feature_Init(64,16,16); //channel 64 , size 16
    //layer 15
    feature_CONV_15 = Feature_Init(64,16,16); //channel 64 , size 16
    feature_16 = Feature_Init(64,16,16); //channel 64 , size 16
    //layer 16
    feature_CONV_16 = Feature_Init(256,16,16); //channel 256 , size 16
    feature_BN_16 = Feature_Init(256,16,16); //channel 256 , size 16
    feature_ReLU_16 = Feature_Init(256,16,16); //channel 256 , size 16
    //bottle neck 5 residual part
    feature_17 = Feature_Init(256,16,16); //channel 256 , size 16
    //bottle neck 6
    //layer 17
    feature_CONV_17 = Feature_Init(64,16,16); //channel 64 , size 16
    feature_18 = Feature_Init(64,16,16); //channel 64 , size 16
    //layer 18
    feature_CONV_18 = Feature_Init(64,16,16); //channel 64 , size 16
    feature_19 = Feature_Init(64,16,16); //channel 64 , size 16
    //layer 19
    feature_CONV_19 = Feature_Init(256,16,16); //channel 256 , size 16
    feature_BN_19 = Feature_Init(256,16,16); //channel 256 , size 16
    feature_ReLU_19 = Feature_Init(256,16,16); //channel 256 , size 16
    //bottle neck 6 residual part
    feature_20 = Feature_Init(256,16,16); //channel 256 , size 16
    //bottle neck 7
    //layer 20
    feature_CONV_20 = Feature_Init(64,16,16); //channel 64 , size 16
    feature_21 = Feature_Init(64,16,16); //channel 64 , size 16
    //layer 21
    feature_CONV_21 = Feature_Init(64,8,8); //channel 64 , size 8
    feature_22 = Feature_Init(64,8,8); //channel 64 , size 8
    //layer 22
    feature_CONV_22 = Feature_Init(256,8,8); //channel 256 , size 8
    feature_BN_22 = Feature_Init(256,8,8); //channel 256 , size 8
    feature_ReLU_22 = Feature_Init(256,8,8); //channel 256 , size 8
    //bottle neck 7 residual part
    feature_Res_22 = Feature_Init(256,8,8); //channel 256 , size 8
    feature_23 = Feature_Init(256,8,8); //channel 256 , size 8
    //bottle neck 8
    //layer 23
    feature_CONV_23 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_24 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 24
    feature_CONV_24 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_25 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 25
    feature_CONV_25 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_BN_25 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_ReLU_25 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 8 residual part
    feature_Res_25 = Feature_Init(512,8,8); //channel 512,size 8
    feature_26 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 9
    //layer 26
    feature_CONV_26 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_27 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 27
    feature_CONV_27 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_28 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 28
    feature_CONV_28 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_BN_28 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_ReLU_28 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 9 residual part
    feature_29 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 10
    //layer 29
    feature_CONV_29 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_30 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 30
    feature_CONV_30 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_31 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 31
    feature_CONV_31 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_BN_31 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_ReLU_31 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 10 residual part
    feature_32 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 11
    //layer 32
    feature_CONV_32 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_33 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 33
    feature_CONV_33 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_34 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 34
    feature_CONV_34 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_BN_34 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_ReLU_34 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 11 residual part
    feature_35 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 12
    //layer 35
    feature_CONV_35 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_36 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 36
    feature_CONV_36 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_37 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 37
    feature_CONV_37 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_BN_37 = Feature_Init(512,8,8); //channel 512 , size 8
    feature_ReLU_37 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 12 residual part
    feature_38 = Feature_Init(512,8,8); //channel 512 , size 8
    //bottle neck 13
    //layer 38
    feature_CONV_38 = Feature_Init(128,8,8); //channel 128 , size 8
    feature_39 = Feature_Init(128,8,8); //channel 128 , size 8
    //layer 39
    feature_CONV_39 = Feature_Init(128,4,4); //channel 128 , size 4
    feature_40 = Feature_Init(128,4,4); //channel 128 , size 4
    //layer 40
    feature_CONV_40 = Feature_Init(512,4,4); //channel 512 , size 4
    feature_BN_40 = Feature_Init(512,4,4); //channel 512 , size 4
    feature_ReLU_40 = Feature_Init(512,4,4); //channel 512 , size 4
    //bottle neck 13 residual part
    feature_Res_40 = Feature_Init(512,4,4); //channel 512 , size 4
    feature_41 = Feature_Init(512,4,4); //channel 512 , size 4
    //bottle neck 14
    //layer 41
    feature_CONV_41 = Feature_Init(256,4,4); //channel 256 , size 4
    feature_42 = Feature_Init(256,4,4); //channel 256 , size 4
    //layer 42
    feature_CONV_42 = Feature_Init(256,4,4); //channel 256 , size 4
    feature_43 = Feature_Init(256,4,4); //channel 256 , size 4
    //layer 43
    feature_CONV_43 = Feature_Init(1024,4,4); //channel 1024 , size 4
    feature_BN_43 = Feature_Init(1024,4,4); //channel 1024 , size 4
    feature_ReLU_43 = Feature_Init(1024,4,4); //channel 1024 , size 4
    //bottle neck 14 residual part
    feature_Res_43 = Feature_Init(1024,4,4); //channel 1024 , size 4
    feature_44 = Feature_Init(1024,4,4); //channel 1024 , size 4
    //bottle neck 15
    //layer 44
    feature_CONV_44 = Feature_Init(256,4,4); //channel 256 , size 4
    feature_45 = Feature_Init(256,4,4); //channel 256 , size 4
    //layer 45
    feature_CONV_45 = Feature_Init(256,4,4); //channel 256 , size 4
    feature_46 = Feature_Init(256,4,4); //channel 256 , size 4
    //layer 46
    feature_CONV_46 = Feature_Init(1024,4,4); //channel 1024 , size 4
    feature_BN_46 = Feature_Init(1024,4,4); //channel 1024 , size 4
    feature_ReLU_46 = Feature_Init(1024,4,4); //channel 1024 , size 4
    //bottle neck 15 residual part
    feature_47 = Feature_Init(1024,4,4); //channel 1024 , size 4
    //bottle neck 16
    //layer 47
    feature_CONV_47 = Feature_Init(256,4,4); //channel 256 , size 4
    feature_48 = Feature_Init(256,4,4); //channel 256 , size 4
    //layer 48
    feature_CONV_48 = Feature_Init(256,2,2); //channel 256 , size 2
    feature_49 = Feature_Init(256,2,2); //channel 256 , size 2
    //layer 49
    feature_CONV_49 = Feature_Init(1024,2,2); //channel 1024 , size 2
    feature_BN_49 = Feature_Init(1024,2,2); //channel 1024 , size 2
    feature_ReLU_49 = Feature_Init(1024,2,2); //channel 1024 , size 2
    //bottle neck 16 residual part
    feature_50 = Feature_Init(1024,2,2); //channel 1024 , size 2
    //global average
    feature_GlobalAvg = Feature_Init(1,1024,1); //channel 1 , row 1024 , column 1
    //full connect
    feature_FullConnect = Feature_Init(1,predictSize,1); //channel 1  ,row predictsize , column 1
    //softmax
    feature_SoftMax = Feature_Init(1,predictSize,1); //channel 1 , row predictSize , column 1
    //predict ans
    predictAns = (int *)malloc(BATCH*sizeof(int)); //size Batch
}

void Init_Gradient()
{
    //struct of the gradient for back propagation
    //softmax
    gradient_CostFunction = Feature_Init(1,predictSize,1); //channel 1 row predictSize , column 1
    //full connect
    gradient_FullConnect = Feature_Init(1,1024,1); //channel 1 row 1024 , column 1
    gradient_FullConnect_Weight = Matrix_Init(Zero,1,predictSize,1024); //channel 1 row predictSize column 1024)
    gradient_bias = Matrix_Init(Zero,1,predictSize,1);//channel 1 row predictSize column 1
    //global average
    gradient_GlobalAverage = Feature_Init(1024,2,2); //channel 1024 row 2 column 2
    //bottle neck 16 residualpart
    gradient_Res_CONV_49 = Feature_Init(1024,4,4);//channel 1024 size 4
    gradient_Res_kernel_49 = Kernel_Init(1024,1024,1,1);// amt 1024 , channel 1024 , size 1
    //layer 49
    gradient_ReLU_49 = Feature_Init(1024,2,2); //channel 1024 , size 2
    gradient_BN_49 = Feature_Init(1024,2,2); //channel 1024 , size 2
    gradient_BN_Beta_49 = Gradient_Init(1024); //size 1024
    gradient_BN_Gamma_49 = Gradient_Init(1024); //size 1024
    gradient_49 = Feature_Init(256,2,2);//channel 256 size 2
    gradient_kernel_49 = Kernel_Init(1024,256,1,1);//amt 1024 channel 256 size 1
    //layer 48
    gradient_BN_48 = Feature_Init(256,2,2); //channel 256 , size 2
    gradient_BN_Beta_48 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_48 = Gradient_Init(256); //size 256
    gradient_48 = Feature_Init(256,4,4);//channel 256 size 4
    gradient_kernel_48 = Kernel_Init(256,256,3,3);//amt 256 channel 256 size 3
    //layer 47
    gradient_BN_47 = Feature_Init(256,4,4); //channel 256 , size 4
    gradient_BN_Beta_47 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_47 = Gradient_Init(256); //size 256
    gradient_47 = Feature_Init(1024,4,4);//channel 1024 size 4
    gradient_kernel_47 = Kernel_Init(256,1024,1,1);//amt 256 channel 1024 size 1
    //layer 46
    gradient_ReLU_46 = Feature_Init(1024,4,4); //channel 1024 , size 4
    gradient_BN_46 = Feature_Init(1024,4,4); //channel 1024 , size 4
    gradient_BN_Beta_46 = Gradient_Init(1024); //size 1024
    gradient_BN_Gamma_46 = Gradient_Init(1024); //size 1024
    gradient_46 = Feature_Init(256,4,4);//channel 256 size 4
    gradient_kernel_46 = Kernel_Init(1024,256,1,1);//amt 1024 channel 256 size 1
    //layer 45
    gradient_BN_45 = Feature_Init(256,4,4); //channel 256 , size 4
    gradient_BN_Beta_45 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_45 = Gradient_Init(256); //size 256
    gradient_45 = Feature_Init(256,4,4);//channel 256 size 4
    gradient_kernel_45 = Kernel_Init(256,256,3,3);//amt 256 channel 256 size 3
    //layer 44
    gradient_BN_44 = Feature_Init(256,4,4); //channel 256 , size 4
    gradient_BN_Beta_44 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_44 = Gradient_Init(256); //size 256
    gradient_44 = Feature_Init(1024,4,4);//channel 1024 size 4
    gradient_kernel_44 = Kernel_Init(256,1024,1,1);//amt 256 channel 1024 size 1
    //bottle neck 14 residualpart
    gradient_Res_BN_43 = Feature_Init(1024,4,4);//channel 1024 size 4
    gradient_Res_BN_Beta_43 = Gradient_Init(1024);//size 1024
    gradient_Res_BN_Gamma_43 = Gradient_Init(1024);//size 1024
    gradient_Res_CONV_43 = Feature_Init(512,4,4);//channel 512 size 4
    gradient_Res_kernel_43 = Kernel_Init(1024,512,1,1);//amt 1024 channel 512 size 1
    //layer 43
    gradient_ReLU_43 = Feature_Init(1024,4,4); //channel 1024 , size 4
    gradient_BN_43 = Feature_Init(1024,4,4); //channel 1024 , size 4
    gradient_BN_Beta_43 = Gradient_Init(1024); //size 1024
    gradient_BN_Gamma_43 = Gradient_Init(1024); //size 1024
    gradient_43 = Feature_Init(256,4,4);//channel 256 size 4
    gradient_kernel_43 = Kernel_Init(1024,256,1,1);//amt 1024 channel 256 size 1
    //layer 42
    gradient_BN_42 = Feature_Init(256,4,4); //channel 256 , size 4
    gradient_BN_Beta_42 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_42 = Gradient_Init(256); //size 256
    gradient_42 = Feature_Init(256,4,4);//channel 256 size 4
    gradient_kernel_42 = Kernel_Init(256,256,3,3);//amt 256 channel 256 size 3
    //layer 41
    gradient_BN_41 = Feature_Init(256,4,4); //channel 256 , size 4
    gradient_BN_Beta_41 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_41 = Gradient_Init(256); //size 256
    gradient_41 = Feature_Init(512,4,4);//channel 512 size 4
    gradient_kernel_41 = Kernel_Init(256,512,1,1);//amt 256 channel 512 size 1
    //bottle neck 13 residualpart
    gradient_Res_BN_40 = Feature_Init(512,4,4);//channel 512 size 4
    gradient_Res_BN_Beta_40 = Gradient_Init(512);//size 512
    gradient_Res_BN_Gamma_40 = Gradient_Init(512);//size 512
    gradient_Res_CONV_40 = Feature_Init(512,8,8);//channel 512 size 8
    gradient_Res_kernel_40 = Kernel_Init(512,512,1,1);//amt 512 channel 512 size 1
    //layer 40
    gradient_ReLU_40 = Feature_Init(512,4,4); //channel 512 , size 4
    gradient_BN_40 = Feature_Init(512,4,4); //channel 512 , size 4
    gradient_BN_Beta_40 = Gradient_Init(512); //size 512
    gradient_BN_Gamma_40 = Gradient_Init(512); //size 512
    gradient_40 = Feature_Init(128,4,4);//channel 128 size 4
    gradient_kernel_40 = Kernel_Init(512,128,1,1);//amt 512 channel 128 size 1
    //layer 39
    gradient_BN_39 = Feature_Init(128,4,4); //channel 128 , size 4
    gradient_BN_Beta_39 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_39 = Gradient_Init(128); //size 128
    gradient_39 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_39 = Kernel_Init(128,128,3,3);//amt 128 channel 128 size 3
    //layer 38
    gradient_BN_38 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_38 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_38 = Gradient_Init(128); //size 128
    gradient_38 = Feature_Init(512,8,8);//channel 512 size 8
    gradient_kernel_38 = Kernel_Init(128,512,1,1);//amt 128 channel 512 size 1
    //layer 37
    gradient_ReLU_37 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_37 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_Beta_37 = Gradient_Init(512); //size 512
    gradient_BN_Gamma_37 = Gradient_Init(512); //size 512
    gradient_37 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_37 = Kernel_Init(512,128,1,1);//amt 512 channel 128 size 1
    //layer 36
    gradient_BN_36 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_36 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_36 = Gradient_Init(128); //size 128
    gradient_36 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_36 = Kernel_Init(128,128,3,3);//amt 128 channel 128 size 3
    //layer 35
    gradient_BN_35 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_35 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_35 = Gradient_Init(128); //size 128
    gradient_35 = Feature_Init(512,8,8);//channel 512 size 8
    gradient_kernel_35 = Kernel_Init(128,512,1,1);//amt 128 channel 512 size 1
    //layer 34
    gradient_ReLU_34 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_34 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_Beta_34 = Gradient_Init(512); //size 512
    gradient_BN_Gamma_34 = Gradient_Init(512); //size 512
    gradient_34 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_34 = Kernel_Init(512,128,1,1);//amt 512 channel 128 size 1
    //layer 33
    gradient_BN_33 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_33 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_33 = Gradient_Init(128); //size 128
    gradient_33 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_33 = Kernel_Init(128,128,3,3);//amt 128 channel 128 size 3
    //layer 32
    gradient_BN_32 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_32 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_32 = Gradient_Init(128); //size 128
    gradient_32 = Feature_Init(512,8,8);//channel 512 size 8
    gradient_kernel_32 = Kernel_Init(128,512,1,1);//amt 128 channel 512 size 1
    //layer 31
    gradient_ReLU_31 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_31 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_Beta_31 = Gradient_Init(512); //size 512
    gradient_BN_Gamma_31 = Gradient_Init(512); //size 512
    gradient_31 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_31 = Kernel_Init(512,128,1,1);//amt 512 channel 128 size 1
    //layer 30
    gradient_BN_30 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_30 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_30 = Gradient_Init(128); //size 128
    gradient_30 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_30 = Kernel_Init(128,128,3,3);//amt 128 channel 128 size 3
    //layer 29
    gradient_BN_29 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_29 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_29 = Gradient_Init(128); //size 128
    gradient_29 = Feature_Init(512,8,8);//channel 512 size 8
    gradient_kernel_29 = Kernel_Init(128,512,1,1);//amt 128 channel 512 size 1
    //layer 28
    gradient_ReLU_28 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_28 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_Beta_28 = Gradient_Init(512); //size 512
    gradient_BN_Gamma_28 = Gradient_Init(512); //size 512
    gradient_28 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_28 = Kernel_Init(512,128,1,1);//amt 512 channel 128 size 1
    //layer 27
    gradient_BN_27 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_27 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_27 = Gradient_Init(128); //size 128
    gradient_27 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_27 = Kernel_Init(128,128,3,3);//amt 128 channel 128 size 3
    //layer 26
    gradient_BN_26 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_26 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_26 = Gradient_Init(128); //size 128
    gradient_26 = Feature_Init(512,8,8);//channel 512 size 8
    gradient_kernel_26 = Kernel_Init(128,512,1,1);//amt 128 channel 512 size 1
    //bottle neck 8 residualpart
    gradient_Res_BN_25 = Feature_Init(512,8,8);//channel 512 size 8
    gradient_Res_BN_Beta_25 = Gradient_Init(512);//size 512
    gradient_Res_BN_Gamma_25 = Gradient_Init(512);//size 512
    gradient_Res_CONV_25 = Feature_Init(256,8,8);//channel 256 size 8
    gradient_Res_kernel_25 = Kernel_Init(512,256,1,1);//amt 512 channel 256 size 1
    //layer 25
    gradient_ReLU_25 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_25 = Feature_Init(512,8,8); //channel 512 , size 8
    gradient_BN_Beta_25 = Gradient_Init(512); //size 512
    gradient_BN_Gamma_25 = Gradient_Init(512); //size 512
    gradient_25 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_25 = Kernel_Init(512,128,1,1);//amt 512 channel 128 size 1
    //layer 24
    gradient_BN_24 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_24 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_24 = Gradient_Init(128); //size 128
    gradient_24 = Feature_Init(128,8,8);//channel 128 size 8
    gradient_kernel_24 = Kernel_Init(128,128,3,3);//amt 128 channel 128 size 3
    //layer 23
    gradient_BN_23 = Feature_Init(128,8,8); //channel 128 , size 8
    gradient_BN_Beta_23 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_23 = Gradient_Init(128); //size 128
    gradient_23 = Feature_Init(256,8,8);//channel 256 size 8
    gradient_kernel_23 = Kernel_Init(128,256,1,1);//amt 128 channel 256 size 1
    //bottle neck 7 residualpart
    gradient_Res_BN_22 = Feature_Init(256,8,8);//channel 256 size 8
    gradient_Res_BN_Beta_22 = Gradient_Init(256);//size 256
    gradient_Res_BN_Gamma_22 = Gradient_Init(256);//size 256
    gradient_Res_CONV_22 = Feature_Init(256,16,16);//channel 256 size 16
    gradient_Res_kernel_22 = Kernel_Init(256,256,1,1);//amt 256 channel 256 size 1
    //layer 22
    gradient_ReLU_22 = Feature_Init(256,8,8); //channel 256 , size 8
    gradient_BN_22 = Feature_Init(256,8,8); //channel 256 , size 8
    gradient_BN_Beta_22 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_22 = Gradient_Init(256); //size 256
    gradient_22 = Feature_Init(64,8,8);//channel 64 size 8
    gradient_kernel_22 = Kernel_Init(256,64,1,1);//amt 256 channel 64 size 1
    //layer 21
    gradient_BN_21 = Feature_Init(64,8,8); //channel 64 , size 8
    gradient_BN_Beta_21 = Gradient_Init(64); //size 64
    gradient_BN_Gamma_21 = Gradient_Init(64); //size 64
    gradient_21 = Feature_Init(64,16,16);//channel 64 size 16
    gradient_kernel_21 = Kernel_Init(64,64,3,3);//amt 64 channel 64 size 3
    //layer 20
    gradient_BN_20 = Feature_Init(64,16,16); //channel 64 , size 16
    gradient_BN_Beta_20 = Gradient_Init(64); //size 64
    gradient_BN_Gamma_20 = Gradient_Init(64); //size 64
    gradient_20 = Feature_Init(256,16,16);//channel 256 size 16
    gradient_kernel_20 = Kernel_Init(64,256,1,1);//amt 64 channel 256 size 1
    //layer 19
    gradient_ReLU_19 = Feature_Init(256,16,16); //channel 256 , size 16
    gradient_BN_19 = Feature_Init(256,16,16); //channel 256 , size 16
    gradient_BN_Beta_19 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_19 = Gradient_Init(256); //size 256
    gradient_19 = Feature_Init(64,16,16);//channel 64 size 16
    gradient_kernel_19 = Kernel_Init(256,64,1,1);//amt 256 channel 64 size 1
    //layer 18
    gradient_BN_18 = Feature_Init(64,16,16); //channel 64 , size 16
    gradient_BN_Beta_18 = Gradient_Init(64); //size 64
    gradient_BN_Gamma_18 = Gradient_Init(64); //size 64
    gradient_18 = Feature_Init(64,16,16);//channel 64 size 16
    gradient_kernel_18 = Kernel_Init(64,64,3,3);//amt 64 channel 64 size 3
    //layer 17
    gradient_BN_17 = Feature_Init(64,16,16); //channel 64 , size 16
    gradient_BN_Beta_17 = Gradient_Init(64); //size 64
    gradient_BN_Gamma_17 = Gradient_Init(64); //size 64
    gradient_17 = Feature_Init(256,16,16);//channel 256 size 16
    gradient_kernel_17 = Kernel_Init(64,256,1,1);//amt 64 channel 256 size 1
    //layer 16
    gradient_ReLU_16 = Feature_Init(256,16,16); //channel 256 , size 16
    gradient_BN_16 = Feature_Init(256,16,16); //channel 256 , size 16
    gradient_BN_Beta_16 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_16 = Gradient_Init(256); //size 256
    gradient_16 = Feature_Init(64,16,16);//channel 64 size 16
    gradient_kernel_16 = Kernel_Init(256,64,1,1);//amt 256 channel 64 size 1
    //layer 15
    gradient_BN_15 = Feature_Init(64,16,16); //channel 64 , size 16
    gradient_BN_Beta_15 = Gradient_Init(64); //size 64
    gradient_BN_Gamma_15 = Gradient_Init(64); //size 64
    gradient_15 = Feature_Init(64,16,16);//channel 64 size 16
    gradient_kernel_15 = Kernel_Init(64,64,3,3);//amt 64 channel 64 size 3
    //layer 14
    gradient_BN_14 = Feature_Init(64,16,16); //channel 64 , size 16
    gradient_BN_Beta_14 = Gradient_Init(64); //size 64
    gradient_BN_Gamma_14 = Gradient_Init(64); //size 64
    gradient_14 = Feature_Init(256,16,16);//channel 256 size 16
    gradient_kernel_14 = Kernel_Init(64,256,1,1);//amt 64 channel 256 size 1
    //bottle neck 4 residualpart
    gradient_Res_BN_13 = Feature_Init(256,16,16);//channel 256 size 16
    gradient_Res_BN_Beta_13 = Gradient_Init(256);//size 256
    gradient_Res_BN_Gamma_13 = Gradient_Init(256);//size 256
    gradient_Res_CONV_13 = Feature_Init(128,16,16);//channel 128 size 16
    gradient_Res_kernel_13 = Kernel_Init(256,128,1,1);//amt 256 channel 128 size 1
    //layer 13
    gradient_ReLU_13 = Feature_Init(256,16,16); //channel 256 , size 16
    gradient_BN_13 = Feature_Init(256,16,16); //channel 256 , size 16
    gradient_BN_Beta_13 = Gradient_Init(256); //size 256
    gradient_BN_Gamma_13 = Gradient_Init(256); //size 256
    gradient_13 = Feature_Init(64,16,16);//channel 64 size 16
    gradient_kernel_13 = Kernel_Init(256,64,1,1);//amt 256 channel 64 size 1
    //layer 12
    gradient_BN_12 = Feature_Init(64,16,16); //channel 64 , size 16
    gradient_BN_Beta_12 = Gradient_Init(64); //size 64
    gradient_BN_Gamma_12 = Gradient_Init(64); //size 64
    gradient_12 = Feature_Init(64,16,16);//channel 64 size 16
    gradient_kernel_12 = Kernel_Init(64,64,3,3);//amt 64 channel 64 size 3
    //layer 11
    gradient_BN_11 = Feature_Init(64,16,16); //channel 64 , size 16
    gradient_BN_Beta_11 = Gradient_Init(64); //size 64
    gradient_BN_Gamma_11 = Gradient_Init(64); //size 64
    gradient_11 = Feature_Init(128,16,16);//channel 128 size 16
    gradient_kernel_11 = Kernel_Init(64,128,1,1);//amt 64 channel 128 size 1
    //layer 10
    gradient_ReLU_10 = Feature_Init(128,16,16); //channel 128 , size 16
    gradient_BN_10 = Feature_Init(128,16,16); //channel 128 , size 16
    gradient_BN_Beta_10 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_10 = Gradient_Init(128); //size 128
    gradient_10 = Feature_Init(32,16,16);//channel 32 size 16
    gradient_kernel_10 = Kernel_Init(128,32,1,1);//amt 128 channel 32 size 1
    //layer 9
    gradient_BN_9 = Feature_Init(32,16,16); //channel 32 , size 16
    gradient_BN_Beta_9 = Gradient_Init(32); //size 32
    gradient_BN_Gamma_9 = Gradient_Init(32); //size 32
    gradient_9 = Feature_Init(32,16,16);//channel 32 size 16
    gradient_kernel_9 = Kernel_Init(32,32,3,3);//amt 32 channel 32 size 3
    //layer 8
    gradient_BN_8 = Feature_Init(32,16,16); //channel 32 , size 16
    gradient_BN_Beta_8 = Gradient_Init(32); //size 32
    gradient_BN_Gamma_8 = Gradient_Init(32); //size 32
    gradient_8 = Feature_Init(128,16,16);//channel 128 size 16
    gradient_kernel_8 = Kernel_Init(32,128,1,1);//amt 32 channel 128 size 1
    //layer 7
    gradient_ReLU_7 = Feature_Init(128,16,16); //channel 128 , size 16
    gradient_BN_7 = Feature_Init(128,16,16); //channel 128 , size 16
    gradient_BN_Beta_7 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_7 = Gradient_Init(128); //size 128
    gradient_7 = Feature_Init(32,16,16);//channel 32 size 16
    gradient_kernel_7 = Kernel_Init(128,32,1,1);//amt 128 channel 32 size 1
    //layer 6
    gradient_BN_6 = Feature_Init(32,16,16); //channel 32 , size 16
    gradient_BN_Beta_6 = Gradient_Init(32); //size 32
    gradient_BN_Gamma_6 = Gradient_Init(32); //size 32
    gradient_6 = Feature_Init(32,16,16);//channel 32 size 16
    gradient_kernel_6 = Kernel_Init(32,32,3,3);//amt 32 channel 32 size 3
    //layer 5
    gradient_BN_5 = Feature_Init(32,16,16); //channel 32 , size 16
    gradient_BN_Beta_5 = Gradient_Init(32); //size 32
    gradient_BN_Gamma_5 = Gradient_Init(32); //size 32
    gradient_5 = Feature_Init(128,16,16);//channel 128 size 16
    gradient_kernel_5 = Kernel_Init(32,128,1,1);//amt 32 channel 128 size 1
    //bottle neck 1 residualpart
    gradient_Res_BN_4 = Feature_Init(128,16,16);//channel 128 size 16
    gradient_Res_BN_Beta_4 = Gradient_Init(128);//size 128
    gradient_Res_BN_Gamma_4 = Gradient_Init(128);//size 128
    gradient_Res_CONV_4 = Feature_Init(32,16,16);//channel 32 size 16
    gradient_Res_kernel_4 = Kernel_Init(128,32,1,1);//amt 128 channel 32 size 1
    //layer 4
    gradient_ReLU_4 = Feature_Init(128,16,16); //channel 128 , size 16
    gradient_BN_4 = Feature_Init(128,16,16); //channel 128 , size 16
    gradient_BN_Beta_4 = Gradient_Init(128); //size 128
    gradient_BN_Gamma_4 = Gradient_Init(128); //size 128
    gradient_4 = Feature_Init(32,16,16);//channel 32 size 16
    gradient_kernel_4 = Kernel_Init(128,32,1,1);//amt 128 channel 32 size 1
    //layer 3
    gradient_BN_3 = Feature_Init(32,16,16); //channel 32 , size 16
    gradient_BN_Beta_3 = Gradient_Init(32); //size 32
    gradient_BN_Gamma_3 = Gradient_Init(32); //size 32
    gradient_3 = Feature_Init(32,16,16);//channel 32 size 16
    gradient_kernel_3 = Kernel_Init(32,32,3,3);//amt 32 channel 32 size 3
    //layer 2
    gradient_BN_2 = Feature_Init(32,16,16); //channel 32 , size 16
    gradient_BN_Beta_2 = Gradient_Init(32); //size 32
    gradient_BN_Gamma_2 = Gradient_Init(32); //size 32
    gradient_2 = Feature_Init(32,16,16);//channel 32 size 16
    gradient_kernel_2 = Kernel_Init(32,32,1,1);//amt 32 channel 32 size 1
    //layer 1
    gradient_MaxPooling = Feature_Init(32,32,32);//channel 32 size 32
    gradient_ReLU_1 = Feature_Init(32,32,32); //channel 32 , size 16
    gradient_BN_1 = Feature_Init(32,32,32); //channel 32 , size 16
    gradient_BN_Beta_1 = Gradient_Init(32); //size 32
    gradient_BN_Gamma_1 = Gradient_Init(32); //size 32
    gradient_1 = Feature_Init(3,32,32);//channel 3 size 32
    gradient_kernel_1 = Kernel_Init(32,3,3,3);//amt 32 channel 3 size 3
}

void Read_Kernel(FILE *textfile,struct Matrix **kernel,int kernelAmt)
{
    float number;
    int channel = kernel[0]->channelSize;
    int row = kernel[0]->rowSize;
    int column = kernel[0]->columnSize;
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        for(int i = 0 ; i < channel ; i++)
        {
            for(int j = 0 ; j < row ; j++)
            {
                for(int k = 0 ; k < column ;k++)
                {
                    fscanf(textfile,"%f",&number);
                    kernel[num]->feature[i][j][k] = number;  
                }
            }
        }
    }
}

void Read_Weight(FILE *textfile,struct Matrix *weight,struct Matrix *bias)
{
    float number;
    int channel = weight->channelSize;
    int row = weight->rowSize;
    int column = weight->columnSize;
    for(int i = 0 ; i < channel ; i++)
    {
        for(int j = 0 ; j < row ; j++)
        {
            for(int k = 0 ; k < column ;k++)
            {
                fscanf(textfile,"%f",&number);
                weight->feature[i][j][k] = number;  
            }
        }
    }
    for(int num = 0 ; num < row ; num ++)
    {
        fscanf(textfile,"%f",&number);
        bias->feature[0][num][0] = number;     
    }
}

void Read_BN(FILE *textfile,struct BN *coeff)
{
    float number;
    int channel = coeff->channelSize;
    for(int i = 0 ; i < channel ; i++)
    {
        fscanf(textfile,"%f",&number);
        coeff->gamma[i] = number;
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        fscanf(textfile,"%f",&number);
        coeff->beta[i] = number;     
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        fscanf(textfile,"%e",&number);
        coeff->runningMean[i] = number;     
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        fscanf(textfile,"%f",&number);
        coeff->runningVar[i] = number;     
    }
}

void Load_Weight()
{
    FILE    *textfile;
    textfile = fopen("kernelwithBN.txt", "r");   
    Read_Kernel(textfile,kernel_1,32);
    Read_BN(textfile,bn_1);
    //bottleneck 1
    Read_Kernel(textfile,kernel_2,32);
    Read_BN(textfile,bn_2);
    Read_Kernel(textfile,kernel_3,32);
    Read_BN(textfile,bn_3);
    Read_Kernel(textfile,kernel_4,128);
    Read_BN(textfile,bn_4);
    Read_Kernel(textfile,resKernel_4,128);
    Read_BN(textfile,resbn_4);
    //bottleneck 2
    Read_Kernel(textfile,kernel_5,32);
    Read_BN(textfile,bn_5);
    Read_Kernel(textfile,kernel_6,32);
    Read_BN(textfile,bn_6);
    Read_Kernel(textfile,kernel_7,128);
    Read_BN(textfile,bn_7);
    //bottleneck 3
    Read_Kernel(textfile,kernel_8,32);
    Read_BN(textfile,bn_8);
    Read_Kernel(textfile,kernel_9,32);
    Read_BN(textfile,bn_9);
    Read_Kernel(textfile,kernel_10,128);
    Read_BN(textfile,bn_10);
    //bottleneck 4
    Read_Kernel(textfile,kernel_11,64);
    Read_BN(textfile,bn_11);
    Read_Kernel(textfile,kernel_12,64);
    Read_BN(textfile,bn_12);
    Read_Kernel(textfile,kernel_13,256);
    Read_BN(textfile,bn_13);
    Read_Kernel(textfile,resKernel_13,256);
    Read_BN(textfile,resbn_13);
    //bottleneck 5
    Read_Kernel(textfile,kernel_14,64);
    Read_BN(textfile,bn_14);
    Read_Kernel(textfile,kernel_15,64);
    Read_BN(textfile,bn_15);
    Read_Kernel(textfile,kernel_16,256);
    Read_BN(textfile,bn_16);
    //bottleneck 6
    Read_Kernel(textfile,kernel_17,64);
    Read_BN(textfile,bn_17);
    Read_Kernel(textfile,kernel_18,64);
    Read_BN(textfile,bn_18);
    Read_Kernel(textfile,kernel_19,256);
    Read_BN(textfile,bn_19);
    //bottleneck 7
    Read_Kernel(textfile,kernel_20,64);
    Read_BN(textfile,bn_20);
    Read_Kernel(textfile,kernel_21,64);
    Read_BN(textfile,bn_21);
    Read_Kernel(textfile,kernel_22,256);
    Read_BN(textfile,bn_22);
    Read_Kernel(textfile,resKernel_22,256);
    Read_BN(textfile,resbn_22);
    //bottleneck 8
    Read_Kernel(textfile,kernel_23,128);
    Read_BN(textfile,bn_23);
    Read_Kernel(textfile,kernel_24,128);
    Read_BN(textfile,bn_24);
    Read_Kernel(textfile,kernel_25,512);
    Read_BN(textfile,bn_25);
    Read_Kernel(textfile,resKernel_25,512);
    Read_BN(textfile,resbn_25);
    //bottleneck 9
    Read_Kernel(textfile,kernel_26,128);
    Read_BN(textfile,bn_26);
    Read_Kernel(textfile,kernel_27,128);
    Read_BN(textfile,bn_27);
    Read_Kernel(textfile,kernel_28,512);
    Read_BN(textfile,bn_28);
    //bottleneck 10
    Read_Kernel(textfile,kernel_29,128);
    Read_BN(textfile,bn_29);
    Read_Kernel(textfile,kernel_30,128);
    Read_BN(textfile,bn_30);
    Read_Kernel(textfile,kernel_31,512);
    Read_BN(textfile,bn_31);
    //bottleneck 11
    Read_Kernel(textfile,kernel_32,128);
    Read_BN(textfile,bn_32);
    Read_Kernel(textfile,kernel_33,128);
    Read_BN(textfile,bn_33);
    Read_Kernel(textfile,kernel_34,512);
    Read_BN(textfile,bn_34);
    //bottleneck 12
    Read_Kernel(textfile,kernel_35,128);
    Read_BN(textfile,bn_35);
    Read_Kernel(textfile,kernel_36,128);
    Read_BN(textfile,bn_36);
    Read_Kernel(textfile,kernel_37,512);
    Read_BN(textfile,bn_37);
    //bottleneck 13
    Read_Kernel(textfile,kernel_38,128);
    Read_BN(textfile,bn_38);
    Read_Kernel(textfile,kernel_39,128);
    Read_BN(textfile,bn_39);
    Read_Kernel(textfile,kernel_40,512);  
    Read_BN(textfile,bn_40);
    Read_Kernel(textfile,resKernel_40,512);   
    Read_BN(textfile,resbn_40);
    //bottleneck 14
    Read_Kernel(textfile,kernel_41,256);
    Read_BN(textfile,bn_41);
    Read_Kernel(textfile,kernel_42,256);
    Read_BN(textfile,bn_42);
    Read_Kernel(textfile,kernel_43,1024); 
    Read_BN(textfile,bn_43); 
    Read_Kernel(textfile,resKernel_43,1024); 
    Read_BN(textfile,resbn_43); 
    //bottleneck 15
    Read_Kernel(textfile,kernel_44,256);
    Read_BN(textfile,bn_44);
    Read_Kernel(textfile,kernel_45,256);
    Read_BN(textfile,bn_45);
    Read_Kernel(textfile,kernel_46,1024);  
    Read_BN(textfile,bn_46);
    //bottleneck 16
    Read_Kernel(textfile,kernel_47,256);
    Read_BN(textfile,bn_47);
    Read_Kernel(textfile,kernel_48,256);
    Read_BN(textfile,bn_48);
    Read_Kernel(textfile,kernel_49,1024);
    Read_BN(textfile,bn_49);
    Read_Kernel(textfile,resKernel_49,1024); 
    //fc
    Read_Weight(textfile,weightFC,bias);

    fclose(textfile);
    
}

