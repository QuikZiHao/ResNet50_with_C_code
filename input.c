#include<stdio.h>
#include<stdlib.h>
#include"input.h"
#include"matrix.h"

void ReadFile(void (*func)(struct Matrix **, int *),struct Matrix **input, int *testCaseLabel)
{
    return (*func)(input,testCaseLabel);
}

void Cifar10(struct Matrix **input,int *testCaseLabel)
{
    predictSize = 10;
    InitInput(input);
    unsigned char* buffer = (unsigned char*)malloc(BIN+1);
    FILE *fp;
    fp = fopen("Cifar10TrainingSet.bin","rb");  // r for read, b for binary
    size_t bytes_read;
    int count = 0 ;
    printf("--------------------loading  CIFAR 10--------------------- \n");
    while (count < TESTSIZE) 
    {
        bytes_read = fread(buffer, 1, BIN+1 , fp);
        unsigned int temp = buffer[0];
        testCaseLabel[count] = temp;
        int bufferLoc = 1;
        for(int channel = 0 ; channel < input[count]->channelSize ; channel ++ )
        {
            for(int row = 0 ; row < input[count]->rowSize ; row ++)
            {
                for(int column = 0 ; column < input[count]->columnSize ; column++)
                {
                    unsigned int temp = buffer[bufferLoc++];
                    input[count]->feature[channel][row][column] = temp/255.0;
                }
            }
        }
        count ++;
    }
    printf("--------------------done load  CIFAR 10--------------------- \n");
    fclose(fp);
    free(buffer);
}

void Cifar100(struct Matrix **input,int *testCaseLabel)
{
    predictSize = 100;
    InitInput(input);
    unsigned char* buffer = (unsigned char*)malloc(BIN+2);
    FILE *fp;
    fp = fopen("Cifar100TrainingSet.bin","rb");  // r for read, b for binary
    size_t bytes_read;
    int count = 0 ;
    printf("-------------------- loading CIFAR 100--------------------- \n");
    while (count < TESTSIZE) 
    {
        bytes_read = fread(buffer, 1, BIN+2 , fp);
        unsigned int temp = buffer[0];
        testCaseLabel[count] = temp;
        int bufferLoc = 2;
        for(int channel = 0 ; channel < input[count]->channelSize ; channel ++ )
        {
            for(int row = 0 ; row < input[count]->rowSize ; row ++)
            {
                for(int column = 0 ; column < input[count]->columnSize ; column++)
                {
                    unsigned int temp = buffer[bufferLoc++];
                    input[count]->feature[channel][row][column] = temp/255.0;
                }
            }
        }
        count ++;
    }
    printf("--------------------done load CIFAR 100--------------------- \n");
    fclose(fp);
    free(buffer);
}

void InitInput(struct Matrix **input)
{
    for (int i = 0 ; i < TESTSIZE; i++)
    {
        input[i] = Matrix_Init(Zero,3,32,32);
    }
    //Matrix_Print(input[1]);
    //printf("init done\n");
}

void Write_Error(char *fileName , int iterateTime , float accurancy , float loss,double elapsed_time)
{
    FILE* errorFile = fopen(fileName,"a");
    fprintf(errorFile,"%d,%f,%f,%lf\n",iterateTime,accurancy,loss,elapsed_time);
    fclose(errorFile);
}