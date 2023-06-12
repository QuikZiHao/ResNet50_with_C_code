#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"matrix.h"

struct Matrix *Matrix_Init(struct Matrix *(*func)(int ,int,int),int channel ,int row,int column)
{
    return (*func)(channel,row,column);
}

struct Matrix *Zero(int channel , int row , int column)
{
    struct Matrix *output = (struct Matrix *)malloc(sizeof(struct Matrix));
    if (output == NULL)
    {
        perror("Failed to allocate memory --- struct Matrix\n");
    }
    output->channelSize = channel;
    output->rowSize = row;
    output->columnSize = column;
    output->feature = (float ***)malloc(channel * sizeof(float**));
    if (output->feature == NULL)
    {
        perror("Failed to allocate memory --- 3D Matrix \n");
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        output->feature[i] = (float **)malloc(row * sizeof(float*));
        if (output->feature[i] == NULL)
        {
            perror("Failed to allocate memory --- 2D Matrix \n");
        }
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        for(int j = 0 ; j < row ; j ++)
        {
            output->feature[i][j] = (float *)malloc( column * sizeof(float));
            if (output->feature[i][j] == NULL)
            {
                perror("Failed to allocate memory --- column Matrix \n");
            }
            for(int k = 0 ; k < column ; k++)
            {
                output->feature[i][j][k] = 0;
            }
        }
    }
    return output;
}

struct Matrix *Random(int channel , int row , int column)
{
    struct Matrix *output = (struct Matrix *)malloc(sizeof(struct Matrix));
    if (output == NULL)
    {
        perror("Failed to allocate memory --- struct Matrix\n");
    }
    output->channelSize = channel;
    output->rowSize = row;
    output->columnSize = column;
    output->feature = (float ***)malloc(channel * sizeof(float**));
    if (output->feature == NULL)
    {
        perror("Failed to allocate memory --- 3D Matrix \n");
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        output->feature[i] = (float **)malloc(row * sizeof(float*));
        if (output->feature[i] == NULL)
        {
            perror("Failed to allocate memory --- 2D Matrix \n");
        }
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        for(int j = 0 ; j < row ; j ++)
        {
            output->feature[i][j] = (float *)malloc( column * sizeof(float));
            if (output->feature[i][j] == NULL)
            {
                perror("Failed to allocate memory --- column Matrix \n");
            }
            for(int k = 0 ; k < column ; k++)
            {
                output->feature[i][j][k] = (((float)rand())/RAND_MAX -0.5); //random number betweeen -.1~.1
            }
        }
    }
    return output;
}

void Matrix_Print(struct Matrix *input)
{
    printf("channel size = %d , size = %d * %d\n" , input->channelSize , input->rowSize , input->columnSize);
    for(int i = 0 ; i < input->channelSize ; i ++)
    {
        printf("channel = %d \n",i);
        for(int j = 0 ; j < input->rowSize ; j ++ )
        {
            for(int k = 0 ; k< input->columnSize ; k ++)
            {
                //printf("%7.4f  ", input->feature[i][j][k]);
                printf("%E  ", input->feature[i][j][k]);
            }
            printf("\n");
        }
    }
}

void Matrix_Show(struct Matrix *input)
{
    for(int i = 0 ; i < input->channelSize ; i++)
        {
            for (int j = 0 ; j < input->rowSize; j++)
            {
                for (int k = 0 ; k < input->columnSize ; k++)
                {
                    printf("%.0f ",input->feature[i][j][k]);
                }
            }
        }
        printf("\n");
}

void Matrix_Check(struct Matrix *input)
{
    printf("channel = %d size = %d X %d\n",input->channelSize,input->rowSize,input->columnSize);
}

void Matrix_Convolution(struct Matrix *input,struct Matrix **kernel,struct Matrix *output ,int paddingAmt,int stride)
{
    int channel = input->channelSize ;
    int outputSize = output->rowSize;
    int kernelSize = kernel[0]->rowSize ;
    int kernelAmt = output->channelSize;
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        for(int j = 0 ; j < outputSize ; j++)
        {
            for(int k = 0 ; k < outputSize ; k++ )
            {
                float temp = 0; // store the temporary after convolution give the value back to the output
                for(int i = 0 ; i < channel ; i ++)
                {   
                    for(int y = 0 ; y < kernelSize ; y ++)
                    {
                        for(int z = 0 ; z < kernelSize ; z++ )
                        {
                            int rowLoc = j*stride+y-paddingAmt;
                            int columnLoc = k*stride+z-paddingAmt;
                            //calculate the column is padding or not
                            if ((rowLoc < 0 || rowLoc >= input->rowSize ) || (columnLoc < 0 || columnLoc >= input->columnSize ))
                            {
                                //use to ignore the padding term
                                continue;
                            }
                            temp += (input->feature[i][rowLoc][columnLoc] * kernel[num]->feature[i][y][z]);
                        }
                    }
                }
                output->feature[num][j][k] = temp;
            }
        }
    }   
}

void Matrix_Resize(struct Matrix *input,int channel,int row , int column)
{
    if((channel == input->channelSize)&&(row == input->rowSize)&&(column == input->columnSize))
    {

    }
    else
    {
        for(int i = 0 ; i < input->channelSize ; i ++)
        {
            for(int j = 0 ; j < input->rowSize ; j++)
            {
                free(input->feature[i][j]);
            }
            free(input->feature[i]);
        }
        free(input->feature);
        input->channelSize = channel;
        input->rowSize= row ;
        input->columnSize = column ;
        input->feature = Matrix_Form(channel,row,column);
    }
}

float ***Matrix_Form(int channel,int row , int column)
{
    float ***output = (float***)malloc(channel*sizeof(float**));
    if (output == NULL)
    {
        printf("Failed to allocate memory --- 3D Matrix \n");
        output = (float ***)malloc(channel * sizeof(float**));
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        output[i] = (float **)malloc(row * sizeof(float*));
        if (output[i] == NULL)
        {
            printf("Failed to allocate memory --- 2D Matrix \n");
            output[i] = (float **)malloc(row * sizeof(float*));
        }
    }
    for(int i = 0 ; i < channel ; i ++)
    {
        for(int j = 0 ; j < row ; j ++)
        {
            output[i][j] = (float *)malloc(column * sizeof(float));
            if (output[i][j] == NULL)
            {
                printf("Failed to allocate memory --- column Matrix \n");
                output[i][j] = (float *)malloc(column * sizeof(float));
            }
            for(int k = 0 ; k < column ; k++)
            {
                output[i][j][k] = 0;
            }
        }
    }
    return output;
}

void Matrix_Multiply(struct Matrix *input,struct Matrix *mulTerm,struct Matrix *output)
{
    int channel = input->channelSize;
    int row = input->rowSize;
    int column = input->columnSize;
    int duration = mulTerm->columnSize;
    for(int i = 0 ; i < channel ; i++)
    {
        for(int x = 0 ; x < duration ; x++)
        {
            for(int j = 0 ; j < row ; j ++)
            {   
                float temp = 0;
                for(int k = 0 ; k < column ; k++)
                {
                    //printf("var = %f, mulTerm = %f ", input->feature[i][k][0],coeff[j][k]);
                    temp = temp + input->feature[i][j][k] * mulTerm->feature[i][k][x];
                }  
                output->feature[i][j][x] = temp;  
            }
        }
    }
}

void Matrix_Sum(struct Matrix *input,struct Matrix *addTerm,struct Matrix *output)
{
    int channel = input->channelSize;
    int row = input->rowSize;
    int column = input->columnSize;
    for(int i = 0 ; i < channel ; i++)
    {
        for(int j = 0 ; j < row ; j++)
        {
            for(int k = 0 ; k < column ; k++)
            {
                output->feature[i][j][k] = input->feature[i][j][k] + addTerm->feature[i][j][k];
            }
        }
    }
}

void Matrix_ToZero(struct Matrix *input)
{
    int channel = input->channelSize;
    int row = input->rowSize;
    int column = input->columnSize;
    for(int i = 0 ; i < channel ; i++)
    {
        for(int j = 0 ; j < row ; j++)
        {
            for(int k = 0 ; k < column ; k++)
            {
                input->feature[i][j][k] = 0;
            }
        }
    }
}