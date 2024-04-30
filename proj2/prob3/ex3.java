package proj2.prob3;

import java.util.concurrent.atomic.AtomicInteger;

public class ex3 {
    public static void main(String[] args) {
        AtomicInteger counter = new AtomicInteger(0);

        for(int i = 0; i < 5; i++){
            int id = i;
            new Thread(()->{
                try{
                    while (true){
                        System.out.printf("Thread %d: is get %d from counter\n", id, counter.get());
                        Thread.sleep(1000);
                        counter.set(counter.get() + 1);
                        System.out.printf("Thread %d: is set %d to counter\n", id, counter.get());
                        Thread.sleep(1000);
                        System.out.printf("Thread %d: is called getAndAdd(1), return = %d\n", id, counter.getAndAdd(1));
                        Thread.sleep(1000);
                        System.out.printf("Thread %d: is called addAndGet(1) return = %d\n", id, counter.addAndGet(1));
                        Thread.sleep(1000);
                    }
                } catch (InterruptedException ignored) {}
            }).start();
        }
    }
}
