CS 179: GPU Computing
Assignment 1



Question 1: Common errors (20pts)
--------------------------------------------------------
--------------------------------------------------------

1.1 ---------------------

When the pointer a is initialized, the asterisk is simply indicating that a is a
pointer - it is not treated as a dereference operator here. So int *a = 3 is 
actually setting the pointer itself to 3, rather than setting the value which 
the pointer points to to 3. Fixing this would require splitting int *a = 3 into
two lines as follows:

int *a;
*a = 3;


1.2 ---------------------

b is not properly declared as a pointer; the program will interpret int* a,b as
a being a pointer, and b being an integer. Fixing this simply requires adding
an asterisk in front of b, producing the declaration int* a,*b;


1.3  ---------------------

malloc(1000) will only allocate space for 1000 bytes, but a single integer will
take more space than one byte. To ensure that the array a can actually store
1000 integers, malloc needs to account for the size of an integer. The fixed 
initialization would look like this:

int i, *a = (int*) malloc(1000*sizeof(int));


1.4 ---------------------

This approach creates an array of three integer pointers, but does not create
space for the second dimension of the array. A proper initialization would
require a malloc call for each row, like so:

void test4(){
    int i, **a = (int**) malloc(3*sizeof(int*));
    for(i = 0; i < 3; i++)
        a[i] = (int*) malloc(100*sizeof(int));
    a[1][1] = 5;
}


1.5 ---------------------

The function is checking the pointer value of a, rather than the value that a is
pointing to. Replacing every instance of a with *a (after the first *a) would 
fix this.



Question 2: Parallelization (30pts)
--------------------------------------------------------
--------------------------------------------------------

2.1 ---------------------

y_1[n] will be faster as its calculation does not rely on other values of y_1,
like y_2[n] does. This means individual values of y_1 can be computed in
parallel, whereas each value of y_2 must be computed sequentially.


2.2---------------------

As the hint states, if c is close to 1, then 1-c is close to 0. This means
that the contribution of y[n-1] to y[n] is minimal; expanding the recurrence
relation reveals that contributions of further y[n-k] terms are increasingly
miniscule in comparison to the contribution of each x[n-k+1] term. Since we
only need an approximation of y[n], we can ignore the minimal contributions of
the y terms and take only the first two or three x terms, like so:
y[n] ~ c*x[n] + (1-c)(c*x[n-1]) + (1-c)^2(c*x[n-2]))





