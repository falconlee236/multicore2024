package proj1.problem1;

public class pc_dynamic {
    private static final int NUM_START = 0;
    private static int NUM_END = 200000;
    private static int NUM_THREADS = 4;
    private static final int TASK_SIZE = 10;

    public static void main (String[] args) {
        if (args.length == 2) {
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }

        pc_dynamic_counter counter = new pc_dynamic_counter();
        pc_dynamic_task tasks = new pc_dynamic_task(NUM_START, NUM_END, TASK_SIZE);
        pc_dynamic_thread[] threads = new pc_dynamic_thread[NUM_THREADS];

        long startTime = System.currentTimeMillis();
        for (int i = 0; i < NUM_THREADS; i++) {
            threads[i] = new pc_dynamic_thread(i + 1, NUM_END, TASK_SIZE, tasks, counter);
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

class pc_dynamic_counter {
    int prime_num = 0;
    synchronized void add_count() {this.prime_num += 1;}
}

class pc_dynamic_task {
    final int num_start, num_end, task_size;
    int current_task;

    public pc_dynamic_task(int num_start, int num_end, int task_size) {
        this.num_start = num_start;
        this.num_end = num_end;
        this.task_size = task_size;
        this.current_task = num_start;
    }

    synchronized int getTask() {
        if (current_task >= num_end) return -1;
        current_task += task_size;
        return current_task - task_size;
    }
}

class pc_dynamic_thread extends Thread {
    final int thread_num, num_end, task_size;
    pc_dynamic_task tasks;
    pc_dynamic_counter counter;
    public pc_dynamic_thread(int thread_num, int num_end, int task_size,
                             pc_dynamic_task tasks, pc_dynamic_counter counter) {
        this.thread_num = thread_num;
        this.num_end = num_end;
        this.task_size = task_size;
        this.tasks = tasks;
        this.counter = counter;
    }

    @Override
    public void run() {
        long startTime = System.currentTimeMillis();
        while (true) {
            int task = tasks.getTask();
            if (task == -1) break;
            for (int i = task; i < Math.min(task + task_size, num_end); i++) {
                if (isPrime(i)) counter.add_count();
            }
        }
        long endTime = System.currentTimeMillis();
        long timeDiff = endTime - startTime;
        System.out.println("Thread " + thread_num + " Execution Time : " + timeDiff + "ms");
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