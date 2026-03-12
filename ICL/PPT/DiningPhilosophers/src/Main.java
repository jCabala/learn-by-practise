import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

class Settings {
  static final int NUM_ITERATIONS = 100;
  static final int NUM_PHILOSOPHERS = 5;
  static final int TIME_BOUND = 30;
}

class Philosopher implements Runnable {
  private final Fork left;
  private final Fork right;
  private final int id;

  Philosopher(Fork _left, Fork _right, int _id) {
      this.left = _left;
      this.right = _right;
      this.id = _id;
  }

  public void think() throws InterruptedException {
    System.out.println("[" + name() + "]: Is thinking...");
    this.sleep();
    System.out.println("[" + name() + "]: Stopped thinking...");
  }

  public void eat() throws InterruptedException {
    System.out.println("[" + name()+ "]: Is hungry.");
    left.pickUp();
    this.sleep();
    right.pickUp();

    System.out.println("[" + name() + "]: Got the forks and is eating...");
    this.sleep();
    System.out.println("[" + name() + "]: Finished eating");

    left.putDown();
    right.putDown();
    System.out.println("[" + name() + "]: Put down the forks.");
  }

  private void sleep() throws InterruptedException {
    Thread.sleep(new Random().nextLong(Settings.TIME_BOUND));
  }

  private String name() {
    return "Philosopher-" + id;
  }

  @Override
  public void run() {
    try {
      for (int i = 0; i < Settings.NUM_ITERATIONS; i++) {
        this.think();
        this.eat();
      }
    } catch (InterruptedException e) {
      throw new RuntimeException();
    }
  }
}

class Fork {
  private final Lock lock;
  private final int id;

  Fork(int _id) {
    this.lock = new ReentrantLock();
    this.id = _id;
  }

  public void pickUp() {
    lock.lock();
    System.out.println("[" + this.name() +"]: Got picked up.");
  }

  public void putDown() {
    lock.unlock();
    System.out.println("[" + this.name() +"]: Got put down.");
  }

  private String name() {
    return "Fork-" + id;
  }
}

public class Main {
  public static void main(String[] args) {
    List<Fork> forks = new ArrayList<>();
    for (int i = 0; i <= Settings.NUM_PHILOSOPHERS; i++) {
      forks.add(new Fork(i));
    }

    List<Philosopher> philosophers = new ArrayList<>();
    for (int i = 0; i <= Settings.NUM_PHILOSOPHERS; i++) {
      Philosopher philosopher = new Philosopher(
          forks.get(i),
          forks.get((i + 1) % Settings.NUM_PHILOSOPHERS),
          i
      );
      philosophers.add(philosopher);
    }

    // Start threads
    System.out.println("Starting simulation...");
    List<Thread> threads = philosophers.stream().map(Thread::new).toList();
    threads.forEach(Thread::start);

    // Wait for the end
    threads.forEach(thread -> {
      try {
        thread.join();
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
    });
    System.out.println("Simulation finished!");
  }
}