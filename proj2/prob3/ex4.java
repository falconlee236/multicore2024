package proj2.prob3;

import java.util.concurrent.CyclicBarrier;

public class ex4 {
    public static void main(String[] args) {
        final CyclicBarrier barrier = new CyclicBarrier(3, () -> {
            System.out.println("Barrier called");
        });

        for (int i = 0; i < 3; i++) {
            int id = i;
            new Thread(() -> {
                while(true){
                    try {
                        System.out.printf("Thread %d: Start\n", id);
                        Thread.sleep((long) (Math.random() * 1000));
                        System.out.printf("Thread %d: Reach Barrier\n", id);
                        barrier.await();
                    } catch (Exception ignored) {}
                }
            }).start();
        }
    }
}
