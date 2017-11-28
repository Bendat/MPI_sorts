#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include <time.h>

// MPI Functions
int MPI_Sort_oddeven(int n, double * a, int root, MPI_Comm comm);
int MPI_Sort_bucket(int n, double * a, double max, int root, MPI_Comm comm);
int MPI_Sort_ranking(int n, double * a, int root, MPI_Comm comm);
int MPI_Sort_direct(int n, double * a, int root, MPI_Comm comm);

int MPI_Exchange(int n, double * a, int rank1, int rank2, MPI_Comm comm);
int MPI_Is_Sorted(int n, double * a, int root, MPI_Comm comm);

// Helper Functions
int is_sorted(int n, double * a);
void fillWithRandom(int n, double *a, int max, int rank);
double * merge_array(int n, double * a, int max, double * b);
void merge_sort(int n, double * a);
void swap(double * a, double * b);

//global helpers to avoid modifying function signatures
double static totalCommTime = 0, totalCompTime= 0, totalExecutionTime = 0;

int main(int argc, char * argv[]) {
    int size,
        rank,
        result,
        i,
        testIterations = 10;
    int n = 100000; // number of elements
    double max = 10.0; // range for element value
    double * a, processorTime, rootProcessorTime;
    MPI_Init( & argc, & argv);
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    MPI_Comm_size(MPI_COMM_WORLD, & size);

    a = (double * ) calloc(n, sizeof(double));
    
    totalExecutionTime = MPI_Wtime();
    for(i = 0; i < testIterations; i++){
        // r takes too long
        if (i > 0 && argv[1][0] == 'r'){
            break;
        }
        if(rank == 0)printf("\n\n#############################################\nStarting %s sort on system of size %d \n#############################################\n\n",argv[1], size);

        MPI_Barrier(MPI_COMM_WORLD);
        printf("Starting %s sort on %d\n", argv[1], rank);

        if (rank == 0) {
           fillWithRandom(n, a, max, rank);
        }

        processorTime = MPI_Wtime();

        if(rank == 0){
            rootProcessorTime = MPI_Wtime();
        }
        switch(argv[1][0]){
            case 'd':
                result = MPI_Sort_direct(n, a, 0, MPI_COMM_WORLD);
                break;
            case 'o':
                result = MPI_Sort_oddeven(n, a, 0, MPI_COMM_WORLD);
                break;
            case 'r':
                result = MPI_Sort_ranking(n, a, 0, MPI_COMM_WORLD);
                break;
            case 'b':
                result = MPI_Sort_bucket(n, a, max, 0, MPI_COMM_WORLD);
                break;
            default:
                result = MPI_Sort_direct(n, a, 0, MPI_COMM_WORLD);
                break;
        }
        if (result != MPI_SUCCESS) {
            return result;
        }
        processorTime = MPI_Wtime() - processorTime;
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0){
            rootProcessorTime = MPI_Wtime() - rootProcessorTime;
            
        }
        printf("Is sorted as seen by rank %d: %d\n", rank,is_sorted(n, a));
    }

    totalExecutionTime = MPI_Wtime() - totalExecutionTime;
    double globalCommTime = 0, globalCompTime = 0;

    MPI_Reduce(&totalCommTime, &globalCommTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalCompTime, &globalCompTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        rootProcessorTime = MPI_Wtime() - rootProcessorTime;
        printf("\n***************************************************************\n");
        printf("Average computation time: %lf \nAverage communication time: %lf\n", 
            round(1000*(globalCompTime/testIterations)/size)/1000, (globalCommTime/testIterations)/size);
        printf("sort %s completed averaging %lf sec for exectution.\n",argv[1],
            round(1000*(globalCompTime/testIterations)/size)/1000+ (globalCommTime/testIterations)/size);
        printf("***************************************************************\n");
    }

    MPI_Finalize();

}

void fillWithRandom(int n, double * a, int max, int rank){
    int i;
    //initialise the array with random values, then scatter to all processors
    srand(((unsigned) time(NULL) + rank));
    for (i = 0; i < n; i++) {
        a[i] = ((double) rand() / RAND_MAX) * max;
        //printf( "Initial: %f\n", a[i] );
    }
}

int is_sorted(int n, double * a){
    int i,
        temp = a[0];
    for(i = 1; i < n; i++){
        if(temp > a[i]){
            return -1;
        }
        temp = a[i];
    }
    return 1;
}

int MPI_Sort_ranking(int n, double * a, int root, MPI_Comm comm) {

    int rank,
        size,
        i,
        j, 
        * ranking, 
        * overallRanking;

    double  compTime = 0,
            commTime = 0,
            time1, 
            * b;

    MPI_Comm_rank(comm, & rank);
    MPI_Comm_size(comm, & size);

    ranking = (int * ) calloc(n / size, sizeof(int));
    overallRanking = (int * ) calloc(n, sizeof(int));
    b = (double * ) calloc(n, sizeof(double));

    time1 = MPI_Wtime();
    MPI_Bcast( & a[0], n, MPI_DOUBLE, root, comm);
    time1 = MPI_Wtime() - time1;
    commTime += time1;

    time1 = MPI_Wtime();
    for (i = 0; i < n / size; i++) {
        ranking[i] = 0;

        for (j = 0; j < n; j++) {
            if (a[j] < a[i + rank * n / size]){
                ranking[i]++;
            }
        }
    }
    time1 = MPI_Wtime() - time1;
    compTime += time1;


    time1 = MPI_Wtime();
    MPI_Gather( & ranking[0], n / size, MPI_INT, & overallRanking[0], n / size, MPI_INT, root, comm);
    time1 = MPI_Wtime() - time1;
    commTime += time1;

    time1 = MPI_Wtime();
    if (rank == root) {
        // restore isSorted in b
        for (i = 0; i < n; i++) {
            b[overallRanking[i]] = a[i];
        }
        // move b to a
        for (i = 0; i < n; i++) {
            a[i] = b[i];
        }
    }

    time1 = MPI_Wtime() - time1;
    compTime += time1;
    totalCommTime += commTime;
    totalCompTime += compTime;

    printf("processor %d communicated for %lf and computed for %lf \n", rank, commTime, totalCompTime);

    return MPI_SUCCESS;
}


int MPI_Sort_oddeven(int n, double * a, int root, MPI_Comm comm) {

    int size, 
        rank, 
        result, 
        i, 
        sorted_result;

    double  compTime = 0, 
            commTime = 0, 
            time1,
            * local_a;

    // get rank and size of communicator
    MPI_Comm_rank(comm, & rank);
    MPI_Comm_size(comm, & size);

    local_a = (double * ) calloc(n / size, sizeof(double));

    time1 = MPI_Wtime();
    result = MPI_Scatter(a, n / size, MPI_DOUBLE, local_a, n / size, MPI_DOUBLE, root, comm);

    if (result != MPI_SUCCESS) {
        return result;
    }

    time1 = MPI_Wtime() - time1;
    commTime += time1;

    time1 = MPI_Wtime();
    merge_sort(n / size, local_a);

    // alternate between odd and even phases
    for (i = 0; i < size; i++) {
        if ((i + rank) % 2 == 0) {
            printf("rank %d is running against rank %d\n", rank, rank+1);
            if (rank < size - 1) {
                result = MPI_Exchange(n / size, local_a, rank, rank + 1, comm);
                if (result != MPI_SUCCESS) {
                    return result;
                }
            }
        } else {
            if (rank > 0) {
                result = MPI_Exchange(n / size, local_a, rank - 1, rank, comm);
                if (result != MPI_SUCCESS) {
                    return result;
                }
            }
        }

        MPI_Barrier(comm);

        if (MPI_Is_Sorted(n/size, local_a, root, comm)){
            break;
        }

    }
    time1 = MPI_Wtime() - time1;
    compTime += time1;

    time1 = MPI_Wtime();
    result = MPI_Gather(local_a, n / size, MPI_DOUBLE, a, n / size, MPI_DOUBLE, root, comm);
    time1 = MPI_Wtime() - time1;

    commTime += time1;
    totalCompTime += compTime;
    totalCommTime += commTime;

    printf("processor %d communicated for %lf and computed for %lf \n", rank, commTime, compTime);

    return result;
}

int MPI_Sort_bucket(int n, double * a, double max, int root, MPI_Comm comm) {

    int rank, 
        size, 
        i, 
        result,
        count, 
        * overallCount,
        * disp;
    double compTime = 0, 
           commTime = 0, 
           time1,
           * bucket;

    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    MPI_Comm_size(MPI_COMM_WORLD, & size);

    bucket = (double * ) calloc(n, sizeof(double));
    overallCount = (int * ) calloc(size, sizeof(int));
    disp = (int * ) calloc(size, sizeof(int));

    time1 = MPI_Wtime();
    MPI_Bcast( & a[0], n, MPI_DOUBLE, root, comm);
    time1 = MPI_Wtime() - time1;
    commTime += time1;

    time1 = MPI_Wtime();
    count = 0;
    for (i = 0; i < n; i++) {
        if ((max * rank / size <= a[i]) && (a[i] < (rank + 1) * max / size)) {
            bucket[count++] = a[i];
        }
    }

    merge_sort(count, bucket);
    time1 = MPI_Wtime() - time1;
    compTime += time1;


    time1 = MPI_Wtime();
    result = MPI_Gather( & count, 1, MPI_INT, overallCount, 1, MPI_INT, root, comm);
    if (result != MPI_SUCCESS) {
        return result;
    }
    time1 = MPI_Wtime() - time1;
    commTime += time1;

    time1 = MPI_Wtime();
    if (rank == 0) {
        disp[0] = 0;
        for (i = 0; i < size - 1; i++) {
            disp[i + 1] = disp[i] + overallCount[i];
        }
    }
    time1 = MPI_Wtime() - time1;
    compTime += time1;

    time1 = MPI_Wtime();
    result = MPI_Gatherv( &bucket[0], count, MPI_DOUBLE, &a[0], overallCount, disp, MPI_DOUBLE, root, comm);
    if (result != MPI_SUCCESS) {
        return result;
    }
    time1 = MPI_Wtime() - time1;
    commTime += time1;

    totalCompTime += compTime;
    totalCommTime += commTime;
    printf("processor %d communicated for %lf and computed for %lf \n", rank, commTime, compTime);

    return MPI_SUCCESS;
}

int MPI_Is_Sorted(int n, double *a, int root, MPI_Comm comm){

    int size, 
        rank, 
        i, 
        isSorted=1;

    double  *first, 
            *last;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    first = (double *) calloc(size, sizeof(double));
    last = (double *) calloc(size, sizeof(double));

    MPI_Gather(&a[0],1,MPI_DOUBLE, first, 1, MPI_DOUBLE, root, comm);
    MPI_Gather(&a[n-1],1,MPI_DOUBLE, last, 1, MPI_DOUBLE, root, comm);

    if (rank ==0){
        for (i=0;i<size-1;i++){
            if (last[i] > first[i+1]){
                 isSorted = 0;
                 break;
            }
        }
    }

    MPI_Bcast(&isSorted, 1, MPI_INT, root, comm);

    return isSorted;
}

int MPI_Sort_direct(int n, double * a, int root, MPI_Comm comm) {

    int size, 
        rank, 
        result,
        i;

    double  compTime = 0, 
            commTime = 0, 
            time1, 
            * scattered_array;

    MPI_Comm_size(comm, & size);
    MPI_Comm_rank(comm, & rank);

    scattered_array = (double * ) calloc(n * size, sizeof(double));

    time1 = MPI_Wtime();
    result = MPI_Scatter(a, n / size, MPI_DOUBLE, scattered_array, n / size, MPI_DOUBLE, root, comm);
    if (result != MPI_SUCCESS) {
        return result;
    }
    time1 = MPI_Wtime() - time1;
    commTime += time1;

    time1 = MPI_Wtime();

    merge_sort(n / size, scattered_array);

    time1 = MPI_Wtime() - time1;
    compTime += time1;

    time1 = MPI_Wtime();
    result = MPI_Gather(scattered_array, n / size, MPI_DOUBLE, a, n / size, MPI_DOUBLE, root, comm);
    if (result != MPI_SUCCESS) {
        return result;
    }
    time1 = MPI_Wtime() - time1;
    commTime += time1;

    time1 = MPI_Wtime();
    if (rank == 0) {
        double * tmp_array = a;
        for (i = 1; i < size; i++) {
            tmp_array = merge_array(i * (n / size), tmp_array, n / size, a + (i * (n / size)));
        }
    }
    
    time1 = MPI_Wtime() - time1;
    compTime += time1;

    totalCompTime += compTime;
    totalCommTime += commTime;

    printf("processor %d communicated for %lf and computed for %lf \n", rank, commTime, compTime);

    return MPI_SUCCESS;
}

int MPI_Exchange(int n, double * a, int rank1, int rank2, MPI_Comm comm) {

    int rank, 
        size, 
        i, 
        result, 
        tag1 = 0, 
        tag2 = 2;

    double * b = (double * ) calloc(n, sizeof(double));
    double * c;

    MPI_Status status;
    MPI_Comm_rank(comm, & rank);
    MPI_Comm_size(comm, & size);


    if (rank == rank1) {
        result = MPI_Send( & a[0], n, MPI_DOUBLE, rank2, tag1, comm);

        if (result != MPI_SUCCESS) {
            return result;
        }

        result = MPI_Recv( & b[0], n, MPI_DOUBLE, rank2, tag2, comm, & status);

        if (result != MPI_SUCCESS) {
            return result;
        }

        c = merge_array(n, a, n, b);

        for (i = 0; i < n; i++) {
            a[i] = c[i];
        }
    } else if (rank == rank2) {
        result = MPI_Recv( & b[0], n, MPI_DOUBLE, rank1, tag1, comm, & status);
        
        if (result != MPI_SUCCESS) {
            return result;
        }

        result = MPI_Send( & a[0], n, MPI_DOUBLE, rank1, tag2, comm);
        if (result != MPI_SUCCESS) {
            return result;
        }

        c = merge_array(n, a, n, b);

        for (i = 0; i < n; i++) {
            a[i] = c[i + n];
        }
    }

    return MPI_SUCCESS;
}


double * merge_array(int n, double * a, int max, double * b) {

    int i,
        j,
        k;

    double * c = (double * ) calloc(n + max, sizeof(double));
    i = j = k = 0;
    while((i < n) && (j < max)){
        if (a[i] <= b[j]) {
            c[k++] = a[i++];
        } else {
            c[k++] = b[j++];
        }
    }
    for (i = j = k = 0; (i < n) && (j < max);)
        if (a[i] <= b[j]) {
            c[k++] = a[i++];
        } else {
            c[k++] = b[j++];
        }
    if (i == n) {
        for (; j < max;) {
            c[k++] = b[j++];
        }
    } else {
        for (; i < n;) {
            c[k++] = a[i++];
        }
    }
    return c;
}

void merge_sort(int n, double * a) {

    double * c;
    int i;

    if (n <= 1) {
        return;
    }

    if (n == 2) {
        if (a[0] > a[1]){ 
            swap( & a[0], & a[1]);
        }
        return;
    }

    merge_sort(n / 2, a);
    merge_sort(n - n / 2, a + n / 2);

    c = merge_array(n / 2, a, n - n / 2, a + n / 2);

    for (i = 0; i < n; i++) {
        a[i] = c[i];
    }

}

void swap(double * a, double * b) {
    double temp;
    temp = * a; 
    * a = * b; 
    * b = temp;
}