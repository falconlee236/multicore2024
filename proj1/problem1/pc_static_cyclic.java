package proj1.problem1;

public class pc_static_cyclic {
    private static final int NUM_START = 0;
    private static int NUM_END = 200000;
    private static int NUM_THREADS = 4;
    private static final int TASK_SIZE = 10;

    public static void main (String[] args) {
        long startTime = System.currentTimeMillis();
        if (args.length == 2) {
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }
        pc_static_cyclic_counter counter = new pc_static_cyclic_counter();
        pc_static_cyclic_thread[] threads = new pc_static_cyclic_thread[NUM_THREADS];
        for (int i = 0; i < NUM_THREADS; i++) {
            threads[i] = new pc_static_cyclic_thread(i, NUM_THREADS, TASK_SIZE, NUM_START, NUM_END, counter);
            threads[i].start();
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            try {threads[i].join();}
            catch (InterruptedException ignored) {}
        }
        long endTime = System.currentTimeMillis();
        long timeDiff = endTime - startTime;
        System.out.println("Program Execution Time : " + timeDiff + "ms");
        System.out.println("1..." + (NUM_END-1) + " prime# counter=" + counter.prime_num);
    }
}

class pc_static_cyclic_counter {
    int prime_num = 0;
    synchronized void addCount() {
        this.prime_num += 1;
    }
}

class pc_static_cyclic_thread extends Thread {
    int thread_id, thread_num, task_size, num_start, num_end;
    pc_static_cyclic_counter counter;
    public pc_static_cyclic_thread(int thread_id, int thread_num, int task_size, int num_start, int num_end,
            pc_static_cyclic_counter counter) {
        this.thread_id = thread_id;
        this.thread_num = thread_num;
        this.task_size = task_size;
        this.num_start = num_start;
        this.num_end = num_end;
        this.counter = counter;
    }

    @Override
    public void run() {
        long startTime = System.currentTimeMillis();
        for (int i = 0; (i * thread_num * task_size) < num_end; i++) {
            for (int j = 0; j < task_size; j++) {
                int num = (i * thread_num * task_size) + (thread_id * task_size) + j;
                if (num < num_end) {
                    if (isPrime(num)) counter.addCount();
                }
                else break;
            }
        }
        long endTime = System.currentTimeMillis();
        long timeDiff = endTime - startTime;
        System.out.println("Thread " + thread_id + " Execution Time : " + timeDiff + "ms");
    }

    private static boolean isPrime(int x) {
        int i;
        if (x<=1) return false;
        for (i=2; i<x; i++) {
            if (x%i == 0) return false;
        }
        return true;
    }
}