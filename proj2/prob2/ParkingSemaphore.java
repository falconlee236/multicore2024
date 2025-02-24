package proj2.prob2;

import java.util.concurrent.Semaphore;

class ParkingGarage_SP {
    private final Semaphore parkingLot;
    public ParkingGarage_SP(int places) {
        if (places < 0)
            places = 0;
        parkingLot = new Semaphore(places);
    }
    public void enter() { // enter parking garage
        try {
            parkingLot.acquire();
        } catch (InterruptedException e) {}
    }
    public void leave() { // leave parking garage
        parkingLot.release();
    }
    public int getPlaces()
    {
        return parkingLot.availablePermits();
    }
}


class Car_SP extends Thread {
    private final ParkingGarage_SP parkingGarage;
    public Car_SP(String name, ParkingGarage_SP p) {
        super(name);
        this.parkingGarage = p;
        start();
    }

    private void tryingEnter()
    {
        System.out.println(getName()+": trying to enter");
    }


    private void justEntered()
    {
        System.out.println(getName()+": just entered");

    }

    private void aboutToLeave()
    {
        System.out.println(getName()+":                                     about to leave");
    }

    private void Left()
    {
        System.out.println(getName()+":                                     have been left");
    }

    public void run() {
        while (true) {
            try {
                sleep((int)(Math.random() * 10000)); // drive before parking
            } catch (InterruptedException e) {}
            tryingEnter();
            parkingGarage.enter();
            justEntered();
            try {
                sleep((int)(Math.random() * 20000)); // stay within the parking garage
            } catch (InterruptedException e) {}
            aboutToLeave();
            parkingGarage.leave();
            Left();

        }
    }
}


public class ParkingSemaphore {
    public static void main(String[] args){
        ParkingGarage_SP parkingGarage = new ParkingGarage_SP(7);
        for (int i=1; i<= 10; i++) {
            Car_SP c = new Car_SP("Car "+i, parkingGarage);
        }
    }
}
