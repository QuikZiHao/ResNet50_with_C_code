#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h>
#include <pthread.h>
#include"input.h"
#include"input.c"
#include"matrix.h"
#include"matrix.c"
#include"batchnorm.h"
#include"batchnorm.c"
#include"frontpropagation.h"
#include"frontpropagation.c"
#include"backpropagation.h"
#include"backpropagation.c"
#include"init.h"
#include"init.c"
#include"learning.h"
#include"learning.c"
#include"multi.h"
#include"multi.c"

#define EPOCHS 5


int main()
{
    printf("hello\n");
    input = (struct Matrix **)malloc(TESTSIZE * sizeof(struct Matrix *));
    testCaseLabel = (int *)malloc(TESTSIZE * sizeof(int));
    ReadFile(Cifar10,input,testCaseLabel);
    Init_Weight();
    Load_Weight();
    printf("kernel init done\n");
    Init_Feature();
    printf("Feature init done\n");
    Init_Gradient();
    printf("Gradient init done\n");
    int count = 0 ;
    int size = TESTSIZE/BATCH;
    clock_t start_time = clock();
    printf("iterate start\n");
    for(int epoch = 0 ; epoch < EPOCHS ; epoch ++)
    {
        printf("epoch %d\n---------------\n",epoch+1);
        float total_Accurancy = 0;
        float total_CostFunction = 0;
        for(int count = 0 ; count < size ; count ++)
        {
            clock_t epoch_time = clock();
            float accurancy = FrontPropagation(count);
            float costFunction = FrontBatch_LossFunction(feature_SoftMax,testCaseLabel,count);
            printf("batch : %d | accurancy : %f  | cost function :%f \n",count,accurancy,costFunction);
            if(count%10 == 0)
            {
                Record_Kernel("Cifar10Kernel.txt");
            }
            BackPropagation(count);
            total_Accurancy += accurancy;
            total_CostFunction += costFunction;
            clock_t epoch_time_end = clock();
            double elapsed_time = (double) (epoch_time_end - epoch_time) / CLOCKS_PER_SEC;
            printf("time use: %f\n",elapsed_time);
            Write_Error("cifar10model_Error.txt",epoch*size+count,accurancy,costFunction,elapsed_time);
        }
        total_Accurancy = total_Accurancy/size;
        total_CostFunction = total_CostFunction/size;
        printf("Train loss : %f ,Train acc : %f \n",total_CostFunction,total_Accurancy);
        Record_Kernel("Cifar10Kernel.txt");
    }
    clock_t end_time = clock();
    double elapsed_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    printf("running time for the trainning:%f s\n", elapsed_time);
    printf("done\n");
    return 0;
}
