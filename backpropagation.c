#include"matrix.h"
#include"batchnorm.h"
#include"backpropagation.h"

void Back_ToZero(float *gradient,int size)
{
    for(int i = 0 ; i < size ; i ++)
    {
        gradient[i] = 0;
    }
}

void Back_Descent(struct Matrix *kernel,struct Matrix *gradient)
{
    int channel = kernel->channelSize;
    int row = kernel->rowSize;
    int column = kernel->columnSize;
    for(int i = 0 ; i < channel ; i ++)
    {
        for(int j = 0 ; j < row ; j ++)
        {
            for(int k = 0 ; k < column ; k++)
            {
                kernel->feature[i][j][k] = kernel->feature[i][j][k] - LEARNINGRATE*gradient->feature[i][j][k];
            }
        }
                
    }
}

void Back_BatchNorm_Descent(struct BN *weight,float *gradientBeta,float *gradientGamma)
{
    int channel = weight->channelSize;
    for(int i = 0 ; i < channel ; i ++)
    {
        weight->beta[i] = weight->beta[i] - LEARNINGRATE*gradientBeta[i];  
        weight->gamma[i] = weight->gamma[i] - LEARNINGRATE*gradientGamma[i];                 
    }
}

void Back_CostFunction(struct Matrix *input,int testCaseLabel,struct Matrix *output)
{
    //input = (1 X predictSize X 1)
    //output = (1 X predictSize X 1)
    //checked
    int row = input->rowSize ; 
    for(int j = 0 ; j < row ; j++)
    {
        output->feature[0][j][0] = input->feature[0][j][0];
        if(j == testCaseLabel)
        {
            output->feature[0][j][0] -= 1 ;
        }
        output->feature[0][j][0] /=BATCH ;
    }
}

void Gradient_CostFunction(struct Matrix **input,int *testCaseLabel , int index , struct Matrix **output)
{
    //checked
    int start = index*BATCH;
    for(int num = 0 ; num < BATCH ; num ++)
    {
        Back_CostFunction(input[num],testCaseLabel[start],output[num]);
        start++;
    }
}

void Back_FullConnect_Bias(struct Matrix *lastTermGradeint,struct Matrix *gradient)
{
    //lastTermGradient = (1 X predictSize X 1)
    Matrix_Sum(gradient,lastTermGradeint,gradient);
}

void Gradient_FullConnect_Bias(struct Matrix **lastTermGradeint,struct Matrix *bias,struct Matrix *gradient)
{
    Matrix_ToZero(gradient);
    for(int num = 0 ; num < BATCH ; num++)
    {
        Back_FullConnect_Bias(lastTermGradeint[num],gradient);
    }
    Back_Descent(bias,gradient);
}

void Back_FullConnect_Weight(struct Matrix *lastTermGradient ,struct Matrix *variable, struct Matrix *gradient)
{
    //lastTermGradient = (1 X predictSize X 1)
    //variable =  (1 X rowSize X 1)
    //gradient = (1 X predictSize X rowSize)
    int outputSize = lastTermGradient->rowSize; //10
    int inputSize = variable->rowSize; //2048
    for(int k = 0 ; k < outputSize ; k ++)
    {
        for(int j = 0 ; j < inputSize ; j ++)
        {
            gradient->feature[0][k][j] += (lastTermGradient->feature[0][k][0]*variable->feature[0][j][0]);
        }
    }
}

void Gradient_FullConnect_Weight(struct Matrix **lastTermGradient ,struct Matrix **variable,struct Matrix *weight, struct Matrix *gradient)
{
    Matrix_ToZero(gradient);
    for(int num = 0 ; num < BATCH ; num++)
    {
        Back_FullConnect_Weight(lastTermGradient[num],variable[num],gradient);
    }
    Back_Descent(weight,gradient);
}

void Back_FullConnect_Variable(struct Matrix *lastTermGradient ,struct Matrix *weight, struct Matrix *gradient)
{
    //lastTermGradient = (1 X predictSize X 1)
    //gradient =  (1 X rowSize X 1)
    //weight = (1 X predictSize X rowSize)
    int row = lastTermGradient->rowSize; //predictsize
    int column = weight->columnSize; // rowsize
    for(int j = 0 ; j < row ; j ++)
    {
        for(int k = 0 ; k < column ; k ++)
        {
            gradient->feature[0][k][0] += (lastTermGradient->feature[0][j][0] * weight->feature[0][j][k]);
        }
    }
}

void Gradient_FullConnect_Variable(struct Matrix **lastTermGradient ,struct Matrix *weight, struct Matrix **gradient)
{
    for(int num = 0 ; num < BATCH ; num++)
    {
        Matrix_ToZero(gradient[num]);
        Back_FullConnect_Variable(lastTermGradient[num],weight,gradient[num]);
    }
}

void Gradient_FullConnect(struct Matrix **lastTermGradient ,struct Matrix *bias ,struct Matrix *gradientBias
                        ,struct Matrix *weight , struct Matrix **variable , struct Matrix *gradientWeight
                        ,struct Matrix **gradient)
{
    //checked
    Gradient_FullConnect_Variable(lastTermGradient,weight,gradient);
    Gradient_FullConnect_Bias(lastTermGradient,bias,gradientBias);
    Gradient_FullConnect_Weight(lastTermGradient,variable,weight,gradientWeight);
}

void Back_GlobalAverage(struct Matrix *lastTermGradient ,struct Matrix *variable, struct Matrix *gradient)
{
    //lastTermGradient = (1 X rowSize X 1)
    //variable = (rowSize X row X column)
    //gradient = (rowSize X row X column)
    int channel = lastTermGradient->rowSize ; 
    int row = variable->rowSize ; 
    int column = variable->columnSize ; 
    for(int i = 0 ; i < channel ; i ++ )
    {
        float temp = lastTermGradient->feature[0][i][0] / (row*column);
        for(int j = 0 ; j < row ; j ++ )
        {
            for(int k = 0 ; k < column ; k++)
            {
                gradient->feature[i][j][k] =  temp; 
            }
        }
    }
}

void Gradient_GlobalAverage(struct Matrix **lastTermGradient ,struct Matrix **variable, struct Matrix **gradient)
{
    for(int num = 0 ; num < BATCH ; num ++)
    {
        Matrix_ToZero(gradient[num]);
        Back_GlobalAverage(lastTermGradient[num],variable[num],gradient[num]);
    }
}

void Back_MaxPooling(struct Matrix *lastTermGradient ,struct Matrix *variable ,struct Matrix *output,int poolingSize,int paddingAmt,int stride,struct Matrix *gradient)
{
    //gradient same size with variable
    //Matrix_Check(lastTermGradient); // 32 x 16 x 16
    //Matrix_Check(variable); //32 x 32 x 32
    //Matrix_Check(output); // 32 x 16 x 16
   // Matrix_Check(gradient); // 32 x 32 x 32
    int channel = variable->channelSize;
    int size = lastTermGradient->rowSize;
    Matrix_ToZero(gradient);
    for(int i = 0 ; i < channel ; i ++)
    {   
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0 ; k < size ; k++ )
            {
                float maximun = output->feature[i][j][k]; //because only the maximun term gradient will be 1 else = 0
                for(int y = 0 ; y < poolingSize ; y ++)
                {
                    for(int z = 0 ; z < poolingSize ; z++ )
                    {
                        int rowLoc = j*stride+y-paddingAmt;
                        int columnLoc = k*stride+z-paddingAmt;
                        //calculate the column is padding or not
                        if ((rowLoc < 0 || rowLoc >= variable->rowSize ) || (columnLoc < 0 || columnLoc >= variable->columnSize))
                        {
                            continue;
                        }
                        if(variable->feature[i][rowLoc][columnLoc] == maximun)
                        {
                            gradient->feature[i][rowLoc][columnLoc] = lastTermGradient->feature[i][j][k];
                        }
                    }
                }
            }
        }
    }
}

void Gradient_MaxPooling(struct Matrix **lastTermGradient ,struct Matrix **variable ,struct Matrix **output,int poolingSize,int paddingAmt,int stride,struct Matrix **gradient)
{
    for(int num = 0 ; num < BATCH ; num ++)
    {
        Back_MaxPooling(lastTermGradient[num],variable[num],output[num],poolingSize,paddingAmt,stride,gradient[num]);
    }
}

void Back_ReLU(struct Matrix *lastTermGradient , struct Matrix *variable, struct Matrix *gradient)
{
    int channel = variable->channelSize ;
    int row = variable->rowSize ; 
    int column = variable->columnSize ;
    for(int i = 0 ; i < channel ; i ++)
    {
        for(int j = 0 ; j < row ; j ++)
        {
            for(int k = 0 ; k < column ; k++)
            {
                if(variable->feature[i][j][k]>0)
                {
                    gradient->feature[i][j][k] = lastTermGradient->feature[i][j][k];
                }
            }
        }
    }
}

void Gradient_ReLU(struct Matrix **lastTermGradient , struct Matrix **variable, struct Matrix **gradient)
{
    for(int num = 0 ; num < BATCH ; num ++)
    {
        Matrix_ToZero(gradient[num]);
        Back_ReLU(lastTermGradient[num],variable[num],gradient[num]);
    }
}


void Gradient_BatchNorm_Variable(struct Matrix **lastTermGradient , struct Matrix **variable,struct BN *coeff,struct Matrix **gradient)
{
    int channel = variable[0]->channelSize;
    int row = variable[0]->rowSize;
    int column = variable[0]->columnSize;
    int amount = row*column*BATCH;
    for(int num = 0 ; num < BATCH ; num++)
    {
        Matrix_ToZero(gradient[num]);
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        float gradientMean = 0;
        float gradientVariance = 0;
        float gamma = coeff->gamma[i];
        float mean = coeff->mean[i];
        float para = pow(coeff->variance[i]+EPSILON,0.5); //(sqrt(var+epsilon))
        for(int num = 0 ; num < BATCH ; num++)
        {
            for(int j = 0 ; j < row ; j ++)
            {
                for(int k = 0 ; k < column ; k++)
                {
                    //dl/dx_hat
                    gradient[num]->feature[i][j][k] = lastTermGradient[num]->feature[i][j][k]*gamma;
                    //dl/dmean = -summation(dl/dx_hat)/(sigma^2+epsilon)^1/2
                    gradientMean += gradient[num]->feature[i][j][k];
                    //dl/dsigma = summantion(dl/dx_hat*(xi-mean)/(2(sigma^2+epsilon)^3/2)
                    gradientVariance += gradient[num]->feature[i][j][k]*(variable[num]->feature[i][j][k]-mean)/para;
                    //dl/dx_hat/(sigma^2+epsilon)^1/2
                    gradient[num]->feature[i][j][k] = gradient[num]->feature[i][j][k]/para;
                }
            }
        }
        for(int num = 0 ; num < BATCH ; num++)
        {
            for(int j = 0 ; j < row ; j ++)
            {
                for(int k = 0 ; k < column ; k++)
                {
                    gradient[num]->feature[i][j][k] = gradient[num]->feature[i][j][k] - (gradientMean + (variable[num]->feature[i][j][k]- mean) / para * gradientVariance) / amount / para;
                }
            }
        }
    }
}

void Gradient_BatchNorm_Weight(struct Matrix **lastTermGradient ,struct BN *coeff,
                          struct Matrix **variable,float *gradientBeta,float *gradientGamma)
{
    int channel = coeff->channelSize;
    int row = lastTermGradient[0]->rowSize;
    int column = lastTermGradient[0]->columnSize;
    int size = row*column;
    for(int i = 0 ; i < channel ; i++)
    {
        gradientBeta[i] = 0 ;
        gradientGamma[i] = 0 ;
        float mean = coeff->mean[i];
        float variance = coeff->variance[i];
        float para = pow((variance+EPSILON),0.5);
        for(int num = 0 ; num < BATCH ; num ++ )
        {
            for(int j = 0 ; j < row ; j++)
            {
                for(int k = 0 ; k < column ; k++)
                {
                    gradientBeta[i] += lastTermGradient[num]->feature[i][j][k];
                    //variable[i][j]][k] = Gamma[i] * variableHat[i][j][k] + Beta[i]
                    float variable_Hat = (variable[num]->feature[i][j][k] - mean)/para ;
                    gradientGamma[i] += (lastTermGradient[num]->feature[i][j][k] * variable_Hat);
                }
            }
        }
    }
    Back_BatchNorm_Descent(coeff,gradientBeta,gradientGamma);
}

void Gradient_BatchNorm(struct Matrix **lastTermGradient , struct Matrix **inputVariable,struct BN *coeff
                        , struct Matrix **gradient ,struct Matrix **outputVariable,float *gradientBeta,float *gradientGamma )
{
    Gradient_BatchNorm_Variable(lastTermGradient,inputVariable,coeff,gradient);
    Gradient_BatchNorm_Weight(lastTermGradient,coeff,inputVariable,gradientBeta,gradientGamma);
}

void Back_Convolution_Variable(struct Matrix *lastTermGradient ,struct Matrix **kernel ,struct Matrix *gradient,int stride, int paddingAmt)
{
    int channel = gradient->channelSize ;
    int size = lastTermGradient->rowSize;
    int kernelSize = kernel[0]->rowSize ;
    int kernelAmt = lastTermGradient->channelSize;
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0 ; k < size ; k++ )
            {
                for(int i = 0 ; i < channel ; i ++)
                {   
                    for(int y = 0 ; y < kernelSize ; y ++)
                    {
                        for(int z = 0 ; z < kernelSize ; z++ )
                        {
                            int rowLoc = j*stride+y-paddingAmt;
                            int columnLoc = k*stride+z-paddingAmt;
                            //calculate the column is padding or not
                            if ((rowLoc < 0 || rowLoc >= gradient->rowSize ) || (columnLoc < 0 || columnLoc >= gradient->columnSize ))
                            {
                                //use to ignore the padding term
                                continue;
                            }
                            //dL/dX = dL/dy * dy/dx (only the kernel which sliding and fix the input will affect the gradient )
                            gradient->feature[i][rowLoc][columnLoc] += (lastTermGradient->feature[num][j][k] *
                                                                         kernel[num]->feature[i][y][z]);
                        }
                    }
                }
            }
        }
    }   
}

void Gradient_Convolution_Variable(struct Matrix **lastTermGradient ,struct Matrix **kernel ,struct Matrix **gradient,int stride, int paddingAmt)
{
    for(int num = 0 ; num < BATCH ; num ++)
    {
        Matrix_ToZero(gradient[num]);
        Back_Convolution_Variable(lastTermGradient[num],kernel,gradient[num],stride,paddingAmt);
    }
}

void Back_Convolution_Kernel(struct Matrix *lastTermGradient ,struct Matrix **gradient,struct Matrix *variable ,int stride, int paddingAmt)
{
    int channel = variable->channelSize ;
    int size = lastTermGradient->rowSize;
    int kernelSize = gradient[0]->rowSize ;
    int kernelAmt = lastTermGradient->channelSize;
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0 ; k < size ; k++ )
            {
                for(int i = 0 ; i < channel ; i ++)
                {   
                    for(int y = 0 ; y < kernelSize ; y ++)
                    {
                        for(int z = 0 ; z < kernelSize ; z++ )
                        {
                            int rowLoc = j*stride+y-paddingAmt;
                            int columnLoc = k*stride+z-paddingAmt;
                            //calculate the column is padding or not
                            if ((rowLoc < 0 || rowLoc >= variable->rowSize ) || (columnLoc < 0 || columnLoc >= variable->columnSize ))
                            {
                                //use to ignore the padding term
                                continue;
                            }
                            //dL/dTheta = dL/dy * dy/dTheta (only the kernel which sliding and fix the input will affect the gradient )
                            gradient[num]->feature[i][y][z] += (lastTermGradient->feature[num][j][k] * 
                                                                variable->feature[i][rowLoc][columnLoc]);
                        }
                    }
                }
            }
        }
    } 
}

void Gradient_Convolution_Kernel(struct Matrix **lastTermGradient ,struct Matrix **gradient,struct Matrix **variable ,int stride, int paddingAmt)
{
    for(int num = 0 ; num < BATCH ; num ++)
    {
        Back_Convolution_Kernel(lastTermGradient[num],gradient,variable[num],stride,paddingAmt);
    }
}

void Gradient_Convolution(struct Matrix **lastTermGradient ,struct Matrix **kernel ,struct Matrix **gradient,
                           struct Matrix **variable ,struct Matrix **gradientKernel, int stride, int paddingAmt)
{
    int kernelAmt = lastTermGradient[0]->channelSize;
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        Matrix_ToZero(gradientKernel[num]);
    }
    Gradient_Convolution_Variable(lastTermGradient,kernel,gradient,stride,paddingAmt);
    Gradient_Convolution_Kernel(lastTermGradient,gradientKernel,variable,stride,paddingAmt);
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        Back_Descent(kernel[num],gradientKernel[num]);
    }
}