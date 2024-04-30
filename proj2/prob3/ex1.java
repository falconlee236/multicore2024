package proj2.prob3;

import java.util.concurrent.ArrayBlockingQueue;

public class ex1 {
    public static void main(String[] args) {
        ArrayBlockingQueue<Integer> queue = new ArrayBlockingQueue<>(10);

        // Producer
        new Thread(() -> {
            int element = 0;
            while (true){
                try {
                    queue.put(element);
                    System.out.printf("Producer added %d to queue\n", element);
                    element++;
                    Thread.sleep(1000);
                } catch (InterruptedException ignored) {}
            }
        }).start();

        // Consumer
        new Thread(() -> {
            while (true){
                try {
                    Integer element = queue.take();
                    System.out.printf("Consumer removed %d from queue\n", element);
                    Thread.sleep(2000);
                } catch (InterruptedException ignored) {}
            }
        }).start();
    }
}
