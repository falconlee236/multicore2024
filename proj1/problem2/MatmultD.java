package proj1.problem2;

import java.util.Scanner;

public class MatmultD { //125231132
    private static final int THREAD_NUM = 4;
    private static final Scanner sc = new Scanner(System.in);
    public static void main(String [] args) {
        int thread_no=0;
        if (args.length==1) thread_no = Integer.valueOf(args[0]);
        else thread_no = THREAD_NUM;

        int a[][]=readMatrix();
        int b[][]=readMatrix();

        MatmultDMResult result_matrix = new MatmultDMResult(a.length, b[0].length);
        long startTime = System.currentTimeMillis();
        MatmultDThread[] threads = new MatmultDThread[thread_no];

        for (int i = 0; i < thread_no; i++) {
            threads[i] = new MatmultDThread(i, thread_no, a, b, result_matrix);
            threads[i].start();
        }

        for (int i = 0; i < thread_no; i++) {
            try {threads[i].join();}
            catch (InterruptedException ignored) {}
        }
        long endTime = System.currentTimeMillis();
        printMatrix(result_matrix.matrix);
        System.out.printf("[thread_no]:%2d , [Time]:%4d ms\n", thread_no, endTime-startTime);
    }

    public static int[][] readMatrix() {
        int rows = sc.nextInt();
        int cols = sc.nextInt();
        int[][] result = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = sc.nextInt();
            }
        }
        return result;
    }

    public static void printMatrix(int[][] mat) {
        System.out.println("Matrix["+mat.length+"]["+mat[0].length+"]");
        int rows = mat.length;
        int columns = mat[0].length;
        int sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                System.out.printf("%4d " , mat[i][j]);
                sum+=mat[i][j];
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("Matrix Sum = " + sum + "\n");
    }
}

class MatmultDMResult {
    int[][] matrix;
    public MatmultDMResult(int row, int column) {
        this.matrix = new int[row][column];
    }
    synchronized void setResult(int row, int column, int result) {
        matrix[row][column] = result;
    }
}

class MatmultDThread extends Thread {
    final int thread_id, thread_num;
    final int a[][], b[][];
    MatmultDMResult res_mat;
    public MatmultDThread(int thread_id, int thread_num, int a[][], int b[][],
                          MatmultDMResult res_mat) {
        this.thread_id = thread_id;
        this.thread_num = thread_num;
        this.a = a;
        this.b = b;
        this.res_mat = res_mat;
    }

    @Override
    public void run() { //a[m][n], b[n][p]
        long startTime = System.currentTimeMillis();
        int n = a[0].length;
        int m = a.length;
        int p = b[0].length;

        for (int i = thread_id; i < m * p; i += thread_num) {
            int col = i % m;
            int row = (i - col) / m;

            int res = 0;
            for(int k = 0; k < n; k++) {
                res += a[row][k] * b[k][col];
            }

            res_mat.setResult(row, col, res);
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("thread_no: %d\nCalculation Time: %d ms\n", thread_id, endTime-startTime);
    }
}
