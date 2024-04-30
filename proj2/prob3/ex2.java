package proj2.prob3;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ex2 {
    public static void main(String[] args) {
        StringBuilder message = new StringBuilder();
        ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock(true);
        Lock readLock = rwLock.readLock();
        Lock writeLock = rwLock.writeLock();

        // Writer
        new Thread(() -> {
            int element = 0;
            while (true){
                try{
                    writeLock.lock();
                    message.append(element);
                    System.out.println("write " + element);
                    Thread.sleep(2000);
                    element++;
                } catch (InterruptedException ignored) {}
                finally {
                    writeLock.unlock();
                }
            }
        }).start();;

        // reader
        new Thread(() -> {
            while(true){
                if (rwLock.isWriteLocked()){
                    System.out.println("I'll take the lock from Write");
                }
                readLock.lock();
                System.out.println("Read Thread messages is \n" + message);
                readLock.unlock();
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException ignored) {}
            }
        }).start();
    }
}
